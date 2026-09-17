"""Settings and data paths. The single place this repo reads configuration.

Every setting has exactly one source:

  contract.yaml  properties of the DATA (model, dimension, semester, schema
                 version). Committed, identical in course-data-prep, and NOT
                 overridable by env — the whole point is that both sides agree.
  environment    properties of THIS MACHINE (database location, LLM host,
                 credentials, hardware) plus retrieval tuning knobs.

Import from here rather than calling os.getenv again elsewhere; two readers of one
setting can drift apart in their defaults.

Paths are anchored to this file, not the cwd: relative paths break when the server
or IDE starts from another directory (SQLite quietly creates an empty db).
"""
from __future__ import annotations

import json
import os
from warnings import deprecated

import yaml
from dotenv import load_dotenv

# override=True: on the host .env wins. Containers get their values from compose.
load_dotenv(override=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Data contract (contract.yaml — no env overrides) ─────────────────────────

CONTRACT_PATH = os.path.join(PROJECT_ROOT, "contract.yaml")

with open(CONTRACT_PATH, encoding="utf-8") as _f:
    CONTRACT: dict = yaml.safe_load(_f)

SCHEMA_VERSION: int = CONTRACT["schema_version"]
EMBED_MODEL: str = CONTRACT["embed_model"]
EMBED_DIM: int = int(CONTRACT["embed_dim"])
SQLITE_EMBED_MODEL: str = CONTRACT["sqlite_embed_model"]

COURSE_YEAR: str = str(CONTRACT["course_year"])
COURSE_SEMESTER: str = str(CONTRACT["course_semester"])

# ── LLM (env) ────────────────────────────────────────────────────────────────

MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GOOGLE_AI = os.getenv("USE_GOOGLE_AI", "false").lower() == "true"

# Gemini 型號。**不要寫死在程式裡** —— Google 會讓舊型號對「新申請的金鑰」停用,
# 實測 gemini-2.5-flash 回 404「no longer available to new users」,而同一把金鑰
# 仍能列出 40 個可用模型。寫死的話,換一把金鑰就壞。
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")

# Ollama 的 context window。**不設就是預設 4096,而那會讓整個 agent 悄悄失效。**
# 2026-09-03 實測:SYSTEM_PROMPT + 工具 schema + 一次工具回傳就達 5,588 token,
# ollama 的處置是截斷而非報錯(`truncating input prompt limit=4096 keep=4`),
# `keep=4` 只留開頭 4 個 token → SYSTEM_PROMPT 整段被切掉 → 模型不知道自己該做什麼,
# 重複呼叫工具直到撞步數上限。先前誤判為「小模型 tool calling 能力不足」。
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

# ── Data files ───────────────────────────────────────────────────────────────

DATA_DB = os.path.join(PROJECT_ROOT, "data.db")

FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")
COURSES_JSONL = os.path.join(PROJECT_ROOT, "courses.jsonl")
FAISS_META_JSON = os.path.join(FAISS_INDEX_DIR, "meta.json")

USER_RECORD_JSON = os.getenv(
    "USER_RECORD_JSON", os.path.join(PROJECT_ROOT, "user_record.json")
)

# ── Backend selection ────────────────────────────────────────────────────────

USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"

SQLITE_DEPRECATION_MSG = (
    "SQLite + FAISS 路徑已棄用,將於後續版本移除。"
    "PostgreSQL 路徑已具備同等的 BM25 + 向量 RRF 混合檢索,且只需要一份 dump。"
    "請改用 USE_SQLITE=false(docker compose up -d postgres)。"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/academic",
)

# ── Retrieval tuning ─────────────────────────────────────────────────────────

RRF_K = int(os.getenv("RRF_K", "60"))
RRF_CANDIDATES = int(os.getenv("RRF_CANDIDATES", "50"))


# ── Contract check ───────────────────────────────────────────────────────────


class ContractMismatch(RuntimeError):
    """Artifacts disagree with contract.yaml — refuse to start rather than rank wrongly."""


_PREP_HINT = (
    "資料由 course-data-prep repo 產生。請確認兩邊的 contract.yaml 一致,"
    "重跑一次準備流程後重新產生資料成品。"
)


def _compare(actual: dict, expected: dict, source: str) -> None:
    diffs = [
        f"  - {key}: 資料成品是 {actual.get(key)!r},contract.yaml 是 {value!r}"
        for key, value in expected.items()
        if str(actual.get(key)) != str(value)
    ]
    if diffs:
        raise ContractMismatch(
            f"資料成品({source})與 contract.yaml 不一致:\n"
            + "\n".join(diffs)
            + f"\n{_PREP_HINT}"
        )


@deprecated(SQLITE_DEPRECATION_MSG)
def _check_sqlite_contract() -> None:
    if not os.path.exists(FAISS_META_JSON):
        raise ContractMismatch(
            f"找不到 {FAISS_META_JSON}(USE_SQLITE=true 需要 faiss_index/)。\n{_PREP_HINT}"
        )
    with open(FAISS_META_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    _compare(
        meta,
        {
            "schema_version": SCHEMA_VERSION,
            "sqlite_embed_model": SQLITE_EMBED_MODEL,
            "course_year": COURSE_YEAR,
            "course_semester": COURSE_SEMESTER,
        },
        source=FAISS_META_JSON,
    )


def _check_pg_contract() -> None:
    import psycopg2

    try:
        conn = psycopg2.connect(DATABASE_URL)
    except psycopg2.Error as e:
        raise ContractMismatch(
            f"連不上 PostgreSQL({e})。請先 `docker compose up -d postgres`,"
            f"或改用 USE_SQLITE=true。"
        ) from e

    try:
        cur = conn.cursor()
        cur.execute("SELECT to_regclass('public.schema_meta')")
        if cur.fetchone()[0] is None:
            raise ContractMismatch(
                "資料庫沒有 schema_meta 表 —— 這個 DB 還沒被準備過(db/init/ 是空的?)。\n"
                "把 course-data-prep 產出的 .sql.gz 放進 ./db/init/,然後\n"
                "`docker compose down -v && docker compose up -d postgres`。\n"
                "(一定要 -v:還原只在 volume 全新時發生,否則新檔案不會生效也不會報錯。)\n"
                f"{_PREP_HINT}"
            )
        cur.execute("SELECT key, value FROM public.schema_meta")
        meta = dict(cur.fetchall())
        cur.close()
    finally:
        conn.close()

    _compare(
        meta,
        {
            "schema_version": SCHEMA_VERSION,
            "embed_model": EMBED_MODEL,
            "embed_dim": EMBED_DIM,
            "course_year": COURSE_YEAR,
            "course_semester": COURSE_SEMESTER,
        },
        source="PostgreSQL schema_meta",
    )


_checked = False


def check_contract(force: bool = False) -> None:
    """Check artifacts against contract.yaml. Runs once per process.

    Guards the silent failure: if the prep side re-embeds with a different model and
    this side doesn't follow, queries don't error — they just return unrelated courses.

    Memoised because app.py constructs CourseLangGraph per request and the PostgreSQL
    check opens a connection.
    """
    global _checked
    if _checked and not force:
        return
    if USE_SQLITE:
        _check_sqlite_contract()
    else:
        _check_pg_contract()
    _checked = True
