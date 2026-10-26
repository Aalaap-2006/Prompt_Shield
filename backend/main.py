"""FastAPI server exposing parser, guardrail and secure agent (Person 2).
Run from repo root:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from promptshield.agent import SecureAgent, ToolPolicy, default_tools
from promptshield.guardrail import DualGateGuardrail, LLMClient
from promptshield.parser import parse_pdf

MAX_UPLOAD = 20 * 1024 * 1024
app = FastAPI(title="PromptShield", version="0.1.0")


@lru_cache(maxsize=1)
def get_guardrail() -> DualGateGuardrail:
    return DualGateGuardrail()  # loads MiniLM once, caches judge verdicts


async def _read_pdf(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "PDF too large (20 MB max)")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Not a PDF file")
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse")
async def parse(file: UploadFile = File(...), max_chars: int = Form(1200)):
    chunks = parse_pdf(await _read_pdf(file), source=file.filename, max_chars=max_chars)
    return {"n_chunks": len(chunks), "chunks": [c.to_dict() for c in chunks]}


@app.post("/scan")
async def scan(task: str = Form(...), file: UploadFile = File(...)):
    chunks = parse_pdf(await _read_pdf(file), source=file.filename)
    report = get_guardrail().screen(task, chunks)
    d = report.to_dict()
    d["chunks"] = {c.chunk_id: c.text[:300] for c in chunks}
    return d


@app.post("/agent/run")
async def run_agent(task: str = Form(...), file: UploadFile = File(...),
                    mode: str = Form("protected"),            # "protected" | "baseline"
                    confirm_high_risk: bool = Form(False)):
    if mode not in {"protected", "baseline"}:
        raise HTTPException(400, "mode must be 'protected' or 'baseline'")
    chunks = parse_pdf(await _read_pdf(file), source=file.filename)
    outbox: list = []
    protected = mode == "protected"
    agent = SecureAgent(
        llm=LLMClient(), guardrail=get_guardrail() if protected else None,
        tools=default_tools(outbox),
        policy=ToolPolicy(confirm_callback=(lambda n, a: True) if confirm_high_risk else None),
        guardrail_enabled=protected, gate_enabled=protected)
    result = agent.run(task, chunks).to_dict()
    result["outbox"] = outbox          # shows whether an exfiltration email would have been sent
    result["mode"] = mode
    return result
