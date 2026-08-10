"""Capture real TMDB responses as test fixtures. Run once."""
import json
import pathlib

from databricks.sdk import WorkspaceClient
from movienight.tmdb_client import TMDBClient, read_token

OUT = pathlib.Path(__file__).parent.parent / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

client = TMDBClient(read_token(WorkspaceClient()))
for movie_id in (550, 27205, 862):
    bundle = client.movie_bundle(movie_id)
    for part, payload in bundle.items():
        (OUT / f"{part.replace('detail', 'movie')}_{movie_id}.json").write_text(
            json.dumps(payload, indent=2)
        )
    print(f"captured {movie_id}")
