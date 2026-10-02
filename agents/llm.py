"""Thin wrapper around the OpenAI Python SDK (Chat Completions): one place for the model, reasoning effort,
tool calling, structured outputs and token/cost accounting."""
from __future__ import annotations

import json
import os
import re
import time

from openai import APIError, OpenAI

import config

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        # reads OPENAI_API_KEY (config.py loads it from .env). ADLENS_LLM_BASE_URL points the same SDK at any
        # OpenAI-compatible server, e.g. a local Ollama model: http://localhost:11434/v1
        _client = OpenAI(base_url=config.LLM_BASE_URL or None, api_key=os.environ.get("OPENAI_API_KEY") or "local")
    return _client


class Usage:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0

    def add(self, resp, dt):
        self.calls += 1
        if resp.usage:
            self.input_tokens += resp.usage.prompt_tokens or 0
            self.output_tokens += resp.usage.completion_tokens or 0
        self.seconds += dt

    def cost_usd(self, model: str | None = None) -> float:
        pin, pout = config.PRICE_PER_MTOK.get(model or config.MODEL, (0.0, 0.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6

    def to_dict(self):
        return dict(model=config.MODEL, llm_calls=self.calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    cost_usd=round(self.cost_usd(), 4), llm_seconds=round(self.seconds, 1))


def to_openai_tools(tools: list) -> list:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]


class Reply:
    """Provider-neutral view of one assistant turn."""

    def __init__(self, msg):
        self.text = msg.content or ""
        self.tool_calls = [(tc.id, tc.function.name, parse_json(tc.function.arguments or "{}")) for tc in (msg.tool_calls or [])]
        self.message = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            self.message["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                          for tc in msg.tool_calls]


def parse_json(text: str) -> dict:
    """Strict parse first; local models sometimes wrap JSON in prose or code fences, so fall back to the outermost {...}."""
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"no JSON object in model output: {text[:200]!r}")
        out = json.loads(m.group(0))
    return out if isinstance(out, dict) else {}


def chat(usage: Usage, system: str, messages: list, tools: list | None = None, schema: dict | None = None, schema_name: str = "output") -> Reply:
    local = bool(config.LLM_BASE_URL)
    kwargs = dict(model=config.MODEL, messages=[{"role": "system", "content": system}] + messages)
    if local:
        kwargs.update(max_tokens=2000, temperature=0)       # Ollama-style servers use max_tokens; deterministic decoding
    else:
        kwargs["max_completion_tokens"] = 16000
        if config.EFFORT and config.MODEL.startswith(("gpt-5", "o")):
            kwargs["reasoning_effort"] = config.EFFORT
    if tools:
        kwargs["tools"] = to_openai_tools(tools)
    if schema:
        # Ollama >= 0.5 also enforces json_schema (grammar-constrained decoding), so the same request works locally
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
    t0 = time.time()
    try:
        resp = client().chat.completions.create(**kwargs)
    except APIError:
        if not local:
            raise
        # small local models occasionally hit a repetition limit: retry once with a little sampling diversity
        resp = client().chat.completions.create(**{**kwargs, "temperature": 0.3})
    usage.add(resp, time.time() - t0)
    choice = resp.choices[0]
    if getattr(choice.message, "refusal", None):
        raise RuntimeError(f"model refused: {choice.message.refusal}")
    return Reply(choice.message)


def tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def structured(usage: Usage, system: str, prompt: str, schema: dict, name: str = "output") -> dict:
    return parse_json(chat(usage, system, [{"role": "user", "content": prompt}], schema=schema, schema_name=name).text)
