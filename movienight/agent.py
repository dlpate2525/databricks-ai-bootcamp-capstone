"""Tool-calling loop.

Deliberately knows nothing about movies - only about the protocol. The llm
argument is any callable (messages, tools) -> assistant message, which is what
lets the whole loop be tested offline against a scripted model.

No Agent Bricks and no AI Gateway: registering an MCP service there requires a
per-user OAuth grant that a service call cannot supply. Calling the serving
endpoint directly has no such dependency.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .tools import TOOL_SCHEMAS, dispatch

DEFAULT_ENDPOINT = "databricks-llama-4-maverick"
MAX_TOOL_RESULT_CHARS = 6000

SYSTEM_PROMPT = """\
You help a group of friends decide what to watch tonight.

Answer only from tool results. Never state a plot, runtime, rating, or
release year that did not come from a tool call, and never invent a movie id.

Order of work:
1. ALWAYS call get_group_context FIRST, before any other tool, even though
   search_movies claims it excludes watched or disliked movies
   "automatically". You must still call get_group_context first, every time,
   without exception, so you actually know who is in the group, what they
   have rated, and what they have already seen. Do not skip this step and do
   not assume it already happened.
2. Call search_movies with the mood or theme in `query`, and put hard
   constraints in the structured arguments - runtime limits in
   max_runtime_minutes, "nothing too violent" in exclude_violent, decades in
   min_year. Do not put those constraints only in the query text.
3. Recommend from what search_movies returned. Say why it fits the group,
   citing ratings or constraints - not vibes.

When the user agrees on a film, call add_to_watchlist, then
save_recommendation with the candidates you considered and your reasoning.
Tell the user plainly whenever you have written something.

If a tool returns status "error", relay the message in plain language and ask
the user to clarify. Do not retry with a made-up id and do not fall back on
general knowledge.

Keep answers to a few sentences unless asked for detail.
"""


@dataclass
class AgentStep:
    kind: str                  # "tool_call" | "answer"
    name: str | None = None
    arguments: dict | None = None
    result: dict | None = None
    content: str | None = None


def _truncate(payload):
    text = json.dumps(payload, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        return text[:MAX_TOOL_RESULT_CHARS] + ' ...(truncated)"}'
    return text


def run_agent(llm, ctx, user_message, history=None, max_iterations=6):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})

    steps = []
    for _ in range(max_iterations):
        assistant = llm(messages, TOOL_SCHEMAS)
        tool_calls = assistant.get("tool_calls") or []

        if not tool_calls:
            content = assistant.get("content") or ""
            steps.append(AgentStep(kind="answer", content=content))
            return steps

        messages.append(assistant)

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw = fn.get("arguments") or "{}"
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                arguments = None

            if arguments is None:
                result = {"status": "error",
                          "message": f"Arguments for {name} were not valid JSON."}
            else:
                result = dispatch(name, arguments, ctx)

            steps.append(AgentStep(kind="tool_call", name=name,
                                   arguments=arguments or {}, result=result))
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "name": name, "content": _truncate(result)})

    steps.append(AgentStep(
        kind="answer",
        content=("I could not settle on an answer - I kept needing more "
                 "lookups. Try narrowing the request.")))
    return steps


def chat_llm_from_workspace(workspace_client, endpoint=DEFAULT_ENDPOINT,
                            timeout=120):
    host = workspace_client.config.host.rstrip("/")
    headers = {**workspace_client.config.authenticate(),
               "Content-Type": "application/json"}
    url = f"{host}/serving-endpoints/{endpoint}/invocations"

    def call(messages, tools):
        body = {"messages": messages, "tools": tools,
                "tool_choice": "auto", "max_tokens": 1024}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return {"role": "assistant",
                    "content": f"The model endpoint returned "
                               f"HTTP {e.code}. Try again in a moment."}
        return payload["choices"][0]["message"]

    return call
