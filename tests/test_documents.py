from movienight.documents import compose_document, doc_hash

FULL = {
    "movie_id": 550, "title": "Fight Club", "release_year": 1999,
    "tagline": "Mischief. Mayhem. Soap.",
    "overview": "An insomniac office worker forms an underground fight club.",
    "runtime": 139, "certification": "R",
    "genres": ["Drama", "Thriller"],
    "keywords": ["nihilism", "rage and hate", "dual identity"],
    "cast": ["Edward Norton", "Brad Pitt"], "director": "David Fincher",
    "poster_path": "/x.jpg", "vote_average": 8.4, "vote_count": 2000,
    "popularity": 60.0,
}


def test_document_contains_every_signal_field():
    doc = compose_document(FULL)
    for expected in ["Fight Club", "1999", "Mischief. Mayhem. Soap.",
                     "Drama", "Thriller", "insomniac", "nihilism",
                     "rage and hate", "Edward Norton", "David Fincher",
                     "139", "R"]:
        assert expected in doc, f"{expected!r} missing from composed document"


def test_document_omits_empty_sections_without_stray_labels():
    sparse = {**FULL, "tagline": "", "keywords": [], "cast": [],
              "director": None, "certification": None, "runtime": None}
    doc = compose_document(sparse)
    assert "Themes:" not in doc
    assert "Starring" not in doc
    assert "Rated" not in doc
    assert "Runtime" not in doc
    assert "Fight Club" in doc          # the real content survives


def test_reviews_are_included_and_capped_at_three():
    doc = compose_document(FULL, reviews=["one", "two", "three", "four"])
    assert "one" in doc and "three" in doc
    assert "four" not in doc


def test_review_excerpts_are_truncated():
    doc = compose_document(FULL, reviews=["x" * 900])
    assert "x" * 400 in doc
    assert "x" * 700 not in doc


def test_hash_is_stable_and_sensitive():
    a = compose_document(FULL)
    assert doc_hash(a) == doc_hash(compose_document(FULL))
    assert len(doc_hash(a)) == 64
    changed = compose_document({**FULL, "overview": "Something else entirely."})
    assert doc_hash(a) != doc_hash(changed)
