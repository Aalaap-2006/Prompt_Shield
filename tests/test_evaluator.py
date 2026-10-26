import json
from pathlib import Path

from promptshield.agent import SecureAgent
from promptshield.evaluator import (ARMS, DetectCase, eval_injecagent, load_benign, load_bipia,
                                    load_injecagent, pct, risk_of, text_to_chunks, wilson, _payload_chunk_ids)
from promptshield.guardrail import ChunkResult, ScreenReport, Verdict

D = Path(__file__).resolve().parents[1] / "data"


def test_wilson_and_percentile():
    w = wilson(0, 20); assert w["rate"] == 0 and w["hi"] < 0.2
    assert wilson(0, 0)["rate"] is None
    assert pct([1, 2, 3, 4, 5], 0.5) == 3


def test_bipia_loader_places_payload_in_its_own_chunk():
    atk, clean = load_bipia(D / "samples/bipia", n=5)
    assert len(atk) == len(clean) == 10
    for c in atk:
        chunks = text_to_chunks(c.text, c.id)
        assert len(_payload_chunk_ids(c, chunks)) == 1


def test_benign_set_loads():
    b = load_benign(D / "benign_technical.jsonl")
    assert len(b) >= 30 and {"meta_security", "legal"} <= {x.category for x in b}


def test_injecagent_loader_and_risk_labels():
    cases, cat = load_injecagent(D / "samples/injecagent", n=5)
    assert len(cases) == 10 and all(t in cat for c in cases for t in c.attacker_tools)
    assert risk_of({"bare": "SendEmail"}) == "high" and risk_of({"bare": "SearchProducts"}) == "low"


class FakeLLM:
    """Hijackable 'agent': if the injection text is in context, it calls the attacker's last tool."""
    def __init__(self, case): self.case = case
    def chat(self, messages, tools=None, **kw):
        if any(m["role"] == "tool" for m in messages):
            return {"content": "done"}
        ctx = messages[1]["content"]
        if self.case.attacker_instruction[:30] in ctx and tools:
            name = self.case.attacker_tools[-1]
            return {"content": "", "tool_calls": [{"id": "1", "type": "function",
                    "function": {"name": name, "arguments": json.dumps({})}}]}
        return {"content": "summary"}


class FakeGuardrail:
    mode = "dual"
    def screen(self, task, chunks):
        res, ok, bad = [], [], []
        for c in chunks:
            hit = "ignore" in c.text.lower() or "please" in c.text.lower()
            (bad if hit else ok).append(c)
            res.append(ChunkResult(c.chunk_id, Verdict.BLOCK if hit else Verdict.ALLOW, "gate1", 0.9 if hit else 0.0, "fake"))
        return ScreenReport(res, ok, bad, {"total_ms": 1.0})


def test_e2e_arms_with_fake_llm():
    cases, cat = load_injecagent(D / "samples/injecagent", n=3, settings=("dh",))
    # InjecAgent attacker instructions are polite requests -> the fake guardrail catches "please"
    base = eval_injecagent(cases[:1], cat, ["baseline"], FakeLLM(cases[0]), None, progress=False)
    assert base["baseline"]["asr_executed"]["k"] == 1
    gate = eval_injecagent(cases[:1], cat, ["baseline", "gate_only"], FakeLLM(cases[0]), None, progress=False)
    assert gate["gate_only"]["asr_executed"]["k"] == 0          # high-risk + tainted + arg provenance
    assert gate["gate_only"]["attempt_rate"]["k"] == 1          # LLM was still hijacked, gate stopped it
    assert "latency_overhead_vs_baseline_ms" in gate["gate_only"]
