"""Compose one embeddable text document per movie.

TMDB overviews average ~295 characters, which retrieves poorly on its own:
a query like "funny sci-fi that isn't too violent" finds nothing to match
against. Keywords carry most of the tone signal ("nihilism" vs "friendship"),
so they are always included when present.

Pure text in, pure text out. No I/O, so this is cheap to iterate on - and
because embeddings are gated on doc_hash, changing this format only re-embeds
the movies whose text actually changed.
"""

import hashlib

MAX_REVIEWS = 3
REVIEW_CHARS = 500


def compose_document(record, reviews=None):
    lines = []

    year = f" ({record['release_year']})" if record.get("release_year") else ""
    lines.append(f"{record['title']}{year}.")

    if record.get("tagline"):
        lines.append(record["tagline"])

    if record.get("genres"):
        lines.append("Genres: " + ", ".join(record["genres"]) + ".")

    if record.get("overview"):
        lines.append(record["overview"])

    if record.get("keywords"):
        lines.append("Themes: " + ", ".join(record["keywords"]) + ".")

    people = []
    if record.get("cast"):
        people.append("Starring " + ", ".join(record["cast"]))
    if record.get("director"):
        people.append(f"Directed by {record['director']}")
    if people:
        lines.append(". ".join(people) + ".")

    facts = []
    if record.get("runtime"):
        facts.append(f"Runtime {record['runtime']} minutes")
    if record.get("certification"):
        facts.append(f"Rated {record['certification']}")
    if facts:
        lines.append(". ".join(facts) + ".")

    for review in (reviews or [])[:MAX_REVIEWS]:
        text = (review or "").strip()
        if text:
            lines.append(text[:REVIEW_CHARS])

    return "\n".join(lines)


def doc_hash(document):
    """Stable content hash. Gates re-embedding, so it must depend only on text."""
    return hashlib.sha256(document.encode("utf-8")).hexdigest()
