"""Embedding client for the Foundation Model API.

Measured against this Free Edition workspace on 2026-08-09:
  batch 4, 2s gap  -> 6/6 OK
  batch 8, 2s gap  -> 6/6 OK   (~2.9 docs/sec)
  batch 16, 3s gap -> 0/5, all 429 REQUEST_LIMIT_EXCEEDED
  batch 256        -> 400 BAD_REQUEST

So MAX_BATCH is 8, and the limit's exact window is unknown - which is why the
backoff is adaptive rather than a fixed sleep. It degrades instead of failing
if the budget behaves differently on another day.
"""

import json
import time
import urllib.error
import urllib.request

MAX_BATCH = 8
DIM = 1024
BASE_GAP = 2.0
DEFAULT_ENDPOINT = "databricks-gte-large-en"


class EmbeddingError(Exception):
    """Embedding failed after retries. Callers should stop, not silently skip."""


class EmbeddingClient:
    def __init__(self, host, auth_headers, endpoint=DEFAULT_ENDPOINT,
                 max_retries=6, timeout=120):
        self._url = f"{host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"
        self._headers = {**auth_headers, "Content-Type": "application/json"}
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = time.sleep

    def _post(self, texts):
        req = urllib.request.Request(
            self._url, data=json.dumps({"input": list(texts)}).encode(),
            headers=self._headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                body = json.loads(r.read().decode())
            rows = sorted(body["data"], key=lambda d: d.get("index", 0))
            return [row["embedding"] for row in rows]
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:200]
            raise EmbeddingError(f"HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise EmbeddingError(f"{type(e).__name__}: {e}") from e

    def _post_with_backoff(self, batch):
        delay = BASE_GAP
        last = None
        for attempt in range(self._max_retries):
            try:
                return self._post(batch)
            except EmbeddingError as e:
                last = e
                if attempt + 1 < self._max_retries:
                    self._sleep(delay)
                    delay = min(delay * 2, 60.0)
        raise EmbeddingError(f"gave up after {self._max_retries} attempts: {last}")

    def embed(self, texts):
        texts = list(texts)
        if not texts:
            return []
        out = []
        for start in range(0, len(texts), MAX_BATCH):
            batch = texts[start:start + MAX_BATCH]
            out.extend(self._post_with_backoff(batch))
            if start + MAX_BATCH < len(texts):
                self._sleep(BASE_GAP)     # stay under the budget on the happy path
        return out

    def embed_one(self, text):
        return self.embed([text])[0]


def client_from_workspace(workspace_client, endpoint=DEFAULT_ENDPOINT):
    return EmbeddingClient(
        host=workspace_client.config.host,
        auth_headers=workspace_client.config.authenticate(),
        endpoint=endpoint,
    )
