"""Drive the agent through natural-language questions and print its tool calls.

Produces the transcript the capstone submission needs as evidence that the agent
takes real actions (writes), not just answers from model knowledge.

    python scripts/demo_agent.py
"""

from databricks.sdk import WorkspaceClient

from movienight.agent import chat_llm_from_workspace, run_agent
from movienight.db import connection
from movienight.embeddings import client_from_workspace
from movienight.tools import ToolContext

QUESTIONS = [
    "We want something funny and light tonight, nothing over two hours and nothing too violent.",
    "What has the group already rated highly?",
    "Add your top pick to our watchlist and save why you chose it.",
]


def main():
    w = WorkspaceClient()
    llm = chat_llm_from_workspace(w)

    with connection() as conn:
        group = conn.execute(
            "SELECT group_id, name FROM groups ORDER BY group_id LIMIT 1"
        ).fetchone()
        user = conn.execute(
            "SELECT user_id, display_name FROM users ORDER BY user_id LIMIT 1"
        ).fetchone()
        ctx = ToolContext(
            conn=conn,
            embedder=client_from_workspace(w),
            group_id=group["group_id"],
            user_id=user["user_id"],
        )
        print(f"group: {group['name']}   acting as: {user['display_name']}\n")

        before = conn.execute(
            "SELECT count(*) AS n FROM watchlist_items WHERE group_id = %(g)s",
            {"g": ctx.group_id},
        ).fetchone()["n"]

        history = []
        for question in QUESTIONS:
            print("=" * 72)
            print(f"Q: {question}")
            print("=" * 72)
            for step in run_agent(llm, ctx, question, history=history):
                if step.kind == "tool_call":
                    result = step.result or {}
                    status = result.get("status")
                    extra = ""
                    if result.get("count") is not None:
                        extra = f" -> {result['count']} results"
                    if result.get("action"):
                        extra = f" -> {result['action']}"
                    print(f"  TOOL {step.name}({step.arguments}) [{status}]{extra}")
                else:
                    print(f"\n  {step.content}\n")
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": step.content})

        after = conn.execute(
            "SELECT count(*) AS n FROM watchlist_items WHERE group_id = %(g)s",
            {"g": ctx.group_id},
        ).fetchone()["n"]
        recs = conn.execute(
            "SELECT count(*) AS n FROM recommendations WHERE group_id = %(g)s",
            {"g": ctx.group_id},
        ).fetchone()["n"]

        print("=" * 72)
        print(f"watchlist rows: {before} -> {after}    recommendation rows: {recs}")
        wrote = after > before or recs > 0
        print("PASS - the agent wrote to the database" if wrote
              else "FAIL - the agent never wrote anything")
        return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main())
