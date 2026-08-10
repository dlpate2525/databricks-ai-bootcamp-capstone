"""Apply sql/*.sql in order. Idempotent - safe to re-run."""
import pathlib
import sys

from movienight.db import connection

SQL_DIR = pathlib.Path(__file__).parent.parent / "sql"


def main():
    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        print("no SQL files found")
        return 1
    with connection() as conn:
        for path in files:
            print(f"applying {path.name} ...", end=" ")
            conn.execute(path.read_text())
            conn.commit()
            print("ok")

        counts = {}
        for table in ["users", "groups", "group_members", "movies",
                      "movie_embeddings", "ratings", "watchlist_items",
                      "recommendations"]:
            row = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
            counts[table] = row["n"]
    print("\nrow counts:")
    for table, n in counts.items():
        print(f"  {table:20s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
