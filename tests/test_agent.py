from movienight.agent import SYSTEM_PROMPT, run_agent
from movienight.tools import ToolContext


class ScriptedLLM:
    """Replays canned assistant messages so the loop runs with no network."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        return self.script.pop(0)


def tool_call(name, args, call_id="c1"):
    import json
    return {"role": "assistant", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def answer(text):
    return {"role": "assistant", "content": text}


def make_ctx():
    class Conn:
        def execute(self, *a, **k):
            class C:
                def fetchall(inner): return []
                def fetchone(inner): return None
            return C()
        def commit(self): pass
    return ToolContext(conn=Conn(), embedder=None, group_id=1, user_id=1)


def test_answer_without_tool_calls_returns_single_step(monkeypatch):
    steps = run_agent(ScriptedLLM([answer("Watch Toy Story.")]),
                      make_ctx(), "what should we watch?")
    assert len(steps) == 1
    assert steps[0].kind == "answer"
    assert "Toy Story" in steps[0].content


def test_tool_call_is_executed_then_answer_returned(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch",
                        lambda n, a, c: {"status": "ok", "count": 1})
    llm = ScriptedLLM([tool_call("search_movies", {"query": "funny"}),
                       answer("Here you go.")])
    steps = run_agent(llm, make_ctx(), "something funny")
    assert [s.kind for s in steps] == ["tool_call", "answer"]
    assert steps[0].name == "search_movies"
    assert steps[0].arguments == {"query": "funny"}
    assert steps[0].result["status"] == "ok"


def test_tool_results_are_fed_back_to_the_model(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch",
                        lambda n, a, c: {"status": "ok", "marker": "XYZZY"})
    llm = ScriptedLLM([tool_call("get_group_context", {}), answer("done")])
    run_agent(llm, make_ctx(), "hi")
    second_call_messages = llm.seen[1]
    assert any(m.get("role") == "tool" and "XYZZY" in m.get("content", "")
               for m in second_call_messages)


def test_loop_stops_at_max_iterations(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch", lambda n, a, c: {"status": "ok"})
    llm = ScriptedLLM([tool_call("search_movies", {"query": "x"})] * 10)
    steps = run_agent(llm, make_ctx(), "loop forever", max_iterations=3)
    assert sum(1 for s in steps if s.kind == "tool_call") == 3
    assert steps[-1].kind == "answer"
    assert "could not" in steps[-1].content.lower()


def test_malformed_tool_arguments_do_not_crash(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(agent_mod, "dispatch", lambda n, a, c: {"status": "ok"})
    bad = {"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "search_movies", "arguments": "{not json"}}]}
    steps = run_agent(ScriptedLLM([bad, answer("recovered")]),
                      make_ctx(), "hi")
    assert steps[0].kind == "tool_call"
    assert steps[0].result["status"] == "error"
    assert steps[-1].content == "recovered"


def test_large_tool_results_are_truncated_before_being_fed_back(monkeypatch):
    import movienight.agent as agent_mod
    monkeypatch.setattr(
        agent_mod, "dispatch",
        lambda n, a, c: {"status": "ok", "results": ["x" * 200] * 100})
    llm = ScriptedLLM([tool_call("search_movies", {"query": "x"}),
                       answer("done")])
    run_agent(llm, make_ctx(), "hi")
    second_call_messages = llm.seen[1]
    tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
    assert len(tool_msg["content"]) <= agent_mod.MAX_TOOL_RESULT_CHARS + 32
    assert "truncated" in tool_msg["content"]


def test_system_prompt_forbids_answering_without_tools():
    lowered = SYSTEM_PROMPT.lower()
    assert "tool" in lowered
    assert "never" in lowered or "do not" in lowered
