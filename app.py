"""Movie Night Planner - Streamlit UI.

Contains no SQL and no HTTP beyond posters. Everything goes through
movienight/*, so the logic stays testable outside Streamlit.

Tool calls are rendered inline as the agent makes them. That is deliberate:
the capstone asks to demonstrate an agent taking actions, and a visible
add_to_watchlist(...) -> ok is the evidence.
"""

import sys
from pathlib import Path

# The whole repo is synced as the app source and app.py's working directory
# is the source root, so `import movienight` should already work. This is
# cheap insurance in case the deployed working directory ever differs.
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st
from databricks.sdk import WorkspaceClient

from movienight.agent import chat_llm_from_workspace, run_agent
from movienight.db import connection, get_pool
from movienight.embeddings import client_from_workspace
from movienight.tools import ToolContext

st.set_page_config(page_title="Movie Night Planner", page_icon="🎬",
                    layout="wide")
POSTER = "https://image.tmdb.org/t/p/w185"


@st.cache_resource
def ensure_pool():
    """Streamlit runs each session in its own thread; get_pool() has an
    unguarded check-then-create singleton. st.cache_resource is
    process-wide and thread-safe, so this guarantees get_pool() is only
    ever invoked once no matter how many sessions race to start."""
    return get_pool()


@st.cache_resource
def workspace():
    return WorkspaceClient()


@st.cache_resource
def embedder():
    return client_from_workspace(workspace())


@st.cache_resource
def llm():
    return chat_llm_from_workspace(workspace())


ensure_pool()


def current_user(conn):
    """Identity comes from the Apps proxy header, never from user input."""
    email = (st.context.headers.get("X-Forwarded-Email")
              if hasattr(st, "context") else None) or "you@example.com"
    row = conn.execute(
        """
        INSERT INTO users (email, display_name, is_demo)
        VALUES (%(e)s, %(n)s, false)
        ON CONFLICT (email) DO UPDATE SET display_name = EXCLUDED.display_name
        RETURNING user_id, display_name
        """,
        {"e": email, "n": email.split("@")[0]},
    ).fetchone()
    conn.commit()
    return row


def ensure_membership(conn, group_id, user_id):
    conn.execute(
        "INSERT INTO group_members (group_id, user_id) VALUES (%(g)s, %(u)s) "
        "ON CONFLICT DO NOTHING", {"g": group_id, "u": user_id})
    conn.commit()


try:
    with connection() as conn:
        me = current_user(conn)
        groups = conn.execute(
            "SELECT group_id, name FROM groups ORDER BY group_id").fetchall()

        st.title("🎬 Movie Night Planner")

        with st.sidebar:
            st.caption(f"Signed in as **{me['display_name']}**")
            if not groups:
                st.warning("No groups yet. Run scripts/bootstrap_db.py.")
                st.stop()
            names = {g["name"]: g["group_id"] for g in groups}
            chosen = st.selectbox("Group", list(names))
            group_id = names[chosen]
            ensure_membership(conn, group_id, me["user_id"])

            members = conn.execute(
                "SELECT u.display_name, u.is_demo FROM group_members gm "
                "JOIN users u USING (user_id) WHERE gm.group_id = %(g)s "
                "ORDER BY u.display_name", {"g": group_id}).fetchall()
            st.write("**Members**")
            for m in members:
                st.write(("· " + m["display_name"]) +
                          (" _(demo)_" if m["is_demo"] else ""))

        ctx = ToolContext(conn=conn, embedder=embedder(),
                           group_id=group_id, user_id=me["user_id"])

        chat_tab, rate_tab, list_tab = st.tabs(
            ["Ask the agent", "Browse & rate", "Watchlist"])

        with chat_tab:
            if "history" not in st.session_state:
                st.session_state.history = []

            for entry in st.session_state.history:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])

            prompt = st.chat_input(
                "e.g. a funny sci-fi movie that isn't too violent, under two hours")
            if prompt:
                st.session_state.history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            steps = run_agent(llm(), ctx, prompt)
                        except Exception as exc:
                            st.error(f"Agent error: {exc}")
                            steps = []
                    for step in steps:
                        if step.kind == "tool_call":
                            status = (step.result or {}).get("status", "?")
                            icon = "✅" if status == "ok" else "⚠️"
                            with st.expander(
                                    f"{icon} `{step.name}` — {status}", expanded=False):
                                st.write("**Arguments**")
                                st.json(step.arguments)
                                st.write("**Result**")
                                st.json(step.result)
                        else:
                            st.markdown(step.content)
                            st.session_state.history.append(
                                {"role": "assistant", "content": step.content})

        with rate_tab:
            if not members:
                st.info("No members yet.")
            else:
                who = st.selectbox(
                    "Rating as", members,
                    format_func=lambda m: m["display_name"], key="rating_as")
                search = st.text_input("Find a movie by title")
                if search:
                    found = conn.execute(
                        "SELECT movie_id, title, release_year, poster_path, runtime, "
                        "certification FROM movies WHERE title ILIKE %(q)s "
                        "ORDER BY popularity DESC LIMIT 12",
                        {"q": f"%{search}%"}).fetchall()
                    if not found:
                        st.info("No movies found yet — the catalog may still be loading.")
                    for chunk_start in range(0, len(found), 4):
                        for col, movie in zip(st.columns(4),
                                               found[chunk_start:chunk_start + 4]):
                            with col:
                                if movie["poster_path"]:
                                    st.image(POSTER + movie["poster_path"])
                                st.caption(f"**{movie['title']}** ({movie['release_year']})")
                                score = st.slider("Score", 1, 10, 7,
                                                   key=f"s{movie['movie_id']}")
                                if st.button("Save", key=f"b{movie['movie_id']}"):
                                    uid = conn.execute(
                                        "SELECT user_id FROM users WHERE display_name=%(n)s",
                                        {"n": who["display_name"]}).fetchone()["user_id"]
                                    conn.execute(
                                        "INSERT INTO ratings (user_id, movie_id, score) "
                                        "VALUES (%(u)s, %(m)s, %(s)s) "
                                        "ON CONFLICT (user_id, movie_id) DO UPDATE SET "
                                        "score = EXCLUDED.score, rated_at = now()",
                                        {"u": uid, "m": movie["movie_id"], "s": score})
                                    conn.commit()
                                    st.success(f"Saved {score}/10")

        with list_tab:
            items = conn.execute(
                "SELECT w.movie_id, m.title, m.release_year, m.poster_path, "
                "w.status, w.reason FROM watchlist_items w "
                "JOIN movies m USING (movie_id) WHERE w.group_id = %(g)s "
                "ORDER BY w.added_at DESC", {"g": group_id}).fetchall()
            if not items:
                st.info("Nothing queued yet — ask the agent for a recommendation.")
            for item in items:
                cols = st.columns([1, 5])
                with cols[0]:
                    if item["poster_path"]:
                        st.image(POSTER + item["poster_path"])
                with cols[1]:
                    st.subheader(f"{item['title']} ({item['release_year']})")
                    st.caption(f"Status: {item['status']}")
                    if item["reason"]:
                        st.write(item["reason"])

except Exception as exc:
    st.error(f"Movie Night Planner hit an error: {exc}")
    st.caption(
        "If the movies table is still being loaded by the ingestion pipeline, "
        "this may resolve itself shortly — try refreshing."
    )
