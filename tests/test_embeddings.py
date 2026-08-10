import io
import json
import urllib.error
import urllib.request

import pytest

from movienight.embeddings import DIM, MAX_BATCH, EmbeddingClient, EmbeddingError


class FakeTransport:
    """Records calls and replays scripted responses, so no network is needed."""

    def __init__(self, script=None):
        self.batches = []
        self.script = list(script or [])
        self.sleeps = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        if self.script:
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return [[0.1] * DIM for _ in texts]


def make(transport, script=None):
    client = EmbeddingClient.__new__(EmbeddingClient)
    client._post = transport
    client._sleep = transport.sleeps.append
    client._max_retries = 5
    return client


def test_splits_into_batches_of_at_most_eight():
    t = FakeTransport()
    client = make(t)
    vectors = client.embed([f"doc {i}" for i in range(20)])
    assert len(vectors) == 20
    assert all(len(v) == DIM for v in vectors)
    assert [len(b) for b in t.batches] == [8, 8, 4]
    assert all(len(b) <= MAX_BATCH for b in t.batches)


def test_preserves_input_order_across_batches():
    class Ordered(FakeTransport):
        def __call__(self, texts):
            self.batches.append(list(texts))
            return [[float(int(t.split()[-1]))] * DIM for t in texts]

    t = Ordered()
    vectors = make(t).embed([f"doc {i}" for i in range(11)])
    assert [v[0] for v in vectors] == [float(i) for i in range(11)]


def test_retries_on_rate_limit_then_succeeds():
    t = FakeTransport(script=[EmbeddingError("429"), EmbeddingError("429"), None])
    client = make(t)
    vectors = client.embed(["a", "b"])
    assert len(vectors) == 2
    assert len(t.batches) == 3           # two failures then success
    assert t.sleeps == sorted(t.sleeps)  # backoff is non-decreasing
    assert t.sleeps[-1] > t.sleeps[0]


def test_gives_up_after_max_retries():
    t = FakeTransport(script=[EmbeddingError("429")] * 9)
    with pytest.raises(EmbeddingError):
        make(t).embed(["a"])


def test_empty_input_makes_no_calls():
    t = FakeTransport()
    assert make(t).embed([]) == []
    assert t.batches == []


# --- Tests that drive the REAL _post (real JSON parsing + real sort-by-index) ---
# These monkeypatch urllib.request.urlopen so no network call ever happens.


class FakeHTTPResponse:
    """Mimics the context-manager object returned by urllib.request.urlopen."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._body


def real_client():
    return EmbeddingClient(host="https://example.databricks.com",
                            auth_headers={"Authorization": "Bearer x"})


def test_post_reorders_shuffled_indices(monkeypatch):
    """The endpoint may return rows out of order; _post must sort by index."""
    payload = {
        "data": [
            {"index": 2, "embedding": [2.0] * DIM},
            {"index": 0, "embedding": [0.0] * DIM},
            {"index": 1, "embedding": [1.0] * DIM},
        ]
    }
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: FakeHTTPResponse(payload),
    )
    client = real_client()
    vectors = client._post(["a", "b", "c"])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


def test_post_preserves_order_when_index_field_is_absent(monkeypatch):
    """Some endpoints omit `index` entirely. .get('index', 0) makes every key
    equal, so Python's *stable* sort must leave the original response order
    untouched rather than scrambling it."""
    payload = {
        "data": [
            {"embedding": [0.0] * DIM},
            {"embedding": [1.0] * DIM},
            {"embedding": [2.0] * DIM},
        ]
    }
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: FakeHTTPResponse(payload),
    )
    client = real_client()
    vectors = client._post(["a", "b", "c"])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


def test_post_raises_embedding_error_on_http_error(monkeypatch):
    """The 429-rate-limit path: urlopen raises HTTPError, _post must wrap it
    in EmbeddingError with the status code visible in the message."""
    err = urllib.error.HTTPError(
        url="https://example.databricks.com/serving-endpoints/x/invocations",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=io.BytesIO(b'{"error_code":"REQUEST_LIMIT_EXCEEDED"}'),
    )

    def raise_it(req, timeout=None):
        raise err

    monkeypatch.setattr(urllib.request, "urlopen", raise_it)
    client = real_client()
    with pytest.raises(EmbeddingError) as exc_info:
        client._post(["a"])
    assert "429" in str(exc_info.value)


def test_no_sleep_after_final_failed_attempt():
    """A terminal failure should not sleep after the last attempt - there is
    no retry to wait for, so that sleep only wastes up to 60s for nothing."""
    max_retries = 5
    t = FakeTransport(script=[EmbeddingError("429")] * max_retries)
    client = make(t)
    client._max_retries = max_retries
    with pytest.raises(EmbeddingError):
        client.embed(["a"])
    assert len(t.sleeps) == max_retries - 1
