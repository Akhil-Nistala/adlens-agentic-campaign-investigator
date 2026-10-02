"""AdLens configuration. Everything sensitive comes from environment variables."""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

# Optional local .env (gitignored): KEY=VALUE lines, see .env.example
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# MySQL (local server). The real datasets live in databases created by earlier projects;
# AdLens writes its own tables (prefixed adlens_) into WORK_DB.
DB = dict(
    host=os.environ.get("ADLENS_DB_HOST", "localhost"),
    port=int(os.environ.get("ADLENS_DB_PORT", "3306")),
    user=os.environ.get("ADLENS_DB_USER", "analytics"),
    password=os.environ.get("ADLENS_DB_PASSWORD", ""),
)
WORK_DB = os.environ.get("ADLENS_WORK_DB", "food_delivery_analytics")

# LLM: OpenAI API (Chat Completions). OPENAI_API_KEY is read from the environment / .env.
MODEL = os.environ.get("ADLENS_MODEL", "gpt-5.4-mini")
LLM_BASE_URL = os.environ.get("ADLENS_LLM_BASE_URL", "")    # empty = api.openai.com
EFFORT = os.environ.get("ADLENS_EFFORT", "low")              # reasoning effort for gpt-5.x: low | medium | high
# USD per 1M tokens (input, output), standard tier, from OpenAI's pricing page (checked 2026-10-02)
PRICE_PER_MTOK = {"gpt-5.4-mini": (0.75, 4.50), "gpt-5.4": (2.50, 15.00), "gpt-5-mini": (0.25, 2.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-5.5": (5.00, 30.00)}
MAX_EVIDENCE_STEPS = int(os.environ.get("ADLENS_MAX_STEPS", "8"))
