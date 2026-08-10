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
