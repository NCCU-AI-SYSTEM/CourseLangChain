"""串流路徑 L2 的證據:五種失敗情境下,瀏覽器實際收到什麼。

    uv run python evals/stream_l2/run.py                    # 只跑目前的程式
    uv run python evals/stream_l2/run.py --baseline 34d48d3 # 同時跑修正前的版本,產出對照報表

量的是 `app.py::generate_streaming` 送出的 SSE 事件 —— 也就是前端 Chat.vue 收到的東西,
不是 `astream` 的回傳值。舊版的例外原文就是在 app.py 這一層送出去的。

需要 PostgreSQL(CourseLangGraph 開機會驗資料契約)與 .env 裡設定好的 LLM。
`--baseline` 會在暫存目錄建一個該 commit 的 git worktree(複製 .env 過去),跑完刪掉。

情境(每個都在獨立子程序裡跑,互不影響):

| 代號 | 觸發方式 |
|---|---|
| exception_real | 真實:LLM 端點改成連不上的位址,模型呼叫拋出連線例外 |
| exception_mock | 模擬:例外訊息裡含資料庫連線字串,看它會不會送到瀏覽器 |
| recursion_real | 真實:recursion_limit 調成 2,需要查課的問題一定超限 |
| timeout_real   | 真實:整體逾時調成 3 秒,真模型回答一題要更久 |
| empty_mock     | 模擬:模型最後一輪回空白。真模型無法穩定重現,只能模擬 |
| json_real      | 真實:帶時間條件的問題,text_to_sql 內部的 LLM 會產生 JSON |

報表寫到 `reports/<時間>-<模型>/`:report.md(對照)與 results.json(每個事件原文)。
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
WORKER_TIMEOUT_SEC = 300

QUESTION = "推薦人工智慧相關的課程"
TIME_QUESTION = "推薦星期三以外、下午的人工智慧相關課程"
SECRET = "postgresql://postgres:secret@db:5432/academic"
TIMEOUT_SEC = 3.0

# 每個情境:標題、真實或模擬、問題、使用者畫面上「必須出現」與「不可出現」的字串
SCENARIOS: dict[str, dict] = {
    "exception_real": {
        "title": "1. 執行中發生例外(真實:LLM 端點連不上)",
        "kind": "真實",
        "question": QUESTION,
        "expect": "系統暫時無法處理",
        "forbid": ["APIConnectionError", "Connection error"],
    },
    "exception_mock": {
        "title": "1b. 執行中發生例外(模擬:例外內容含連線字串)",
        "kind": "模擬",
        "question": QUESTION,
        "expect": "系統暫時無法處理",
        "forbid": ["secret", "postgresql://", "RuntimeError"],
    },
    "recursion_real": {
        "title": "2. 步數超限(真實:recursion_limit=2)",
        "kind": "真實",
        "question": QUESTION,
        "expect": "查詢步驟過多",
        "forbid": ["GraphRecursionError", "Recursion limit"],
    },
    "timeout_real": {
        "title": f"3. 超過逾時(真實:整體逾時 {TIMEOUT_SEC:g} 秒)",
        "kind": "真實",
        "question": QUESTION,
        "expect": "處理時間過長",
        "forbid": [],
        "max_elapsed": TIMEOUT_SEC + 2,
    },
    "empty_mock": {
        "title": "4. 模型最後沒回答(模擬:最終回覆是空白)",
        "kind": "模擬",
        "question": QUESTION,
        "expect": "沒能組出回覆",
        "forbid": [],
    },
    "json_real": {
        "title": "5. 工具內部 LLM 的輸出(真實:帶時間條件的問題)",
        "kind": "真實",
        "question": TIME_QUESTION,
        "expect": "人工智慧",
        "forbid": ['"exclude_times"', '"include_times"', "```json"],
    },
}


# ── worker:在指定的程式樹裡跑一個情境 ─────────────────────────────────────────


class _ScriptGraph:
    """假的 brain_agent:依序送出事件;遇到例外就 raise。"""

    def __init__(self, *steps):
        self.steps = steps

    async def astream_events(self, *_args, **_kwargs):
        for step in self.steps:
            if isinstance(step, BaseException):
                raise step
            yield step


def _model_end(text: str, node: str = "agent") -> dict:
    from langchain_core.messages import AIMessage

    return {
        "event": "on_chat_model_end",
        "data": {"output": AIMessage(content=text)},
        "metadata": {"langgraph_node": node},
    }


async def _collect(app, question: str, session_id: str) -> list[dict]:
    events: list[dict] = []
    start = time.monotonic()
    async for line in app.generate_streaming(question, session_id):
        if line.startswith("data: "):
            payload = json.loads(line[len("data: "):])
            events.append({"t": round(time.monotonic() - start, 1), **payload})
    return events


def _worker(root: Path, name: str) -> dict:
    sys.path.insert(0, str(root))
    os.chdir(root)
    if name == "exception_real":
        import paths

        # 要在 utils.llm 被 import 之前改:它在 import 時就把這個值抄走了
        paths.OPENAI_BASE_URL = "http://127.0.0.1:9/v1"

    import app

    agent = app.get_agent()
    if name.endswith("_mock"):
        from harness import SafeAgentExecutor

        status = {"event": "on_tool_start", "name": "retrieve_tool"}
        fake = (
            _ScriptGraph(status, RuntimeError(f"could not connect: {SECRET}"))
            if name == "exception_mock"
            else _ScriptGraph(status, _model_end("   "))
        )
        agent.brain_agent = fake
        agent.executor = SafeAgentExecutor(fake)
    if name == "recursion_real":
        agent.executor.recursion_limit = 2
    if name == "timeout_real":
        agent.executor.timeout_sec = TIMEOUT_SEC

    start = time.monotonic()
    events = asyncio.run(_collect(app, SCENARIOS[name]["question"], f"stream-l2-{name}-{time.time()}"))
    return {"events": events, "elapsed": round(time.monotonic() - start, 1)}


# ── main:跑所有情境、判定、寫報表 ─────────────────────────────────────────────


def _shown(events: list[dict]) -> str:
    """使用者畫面上看得到的文字:回覆本文 + 錯誤細節(Chat.vue 兩個都會顯示)。"""
    parts = []
    for e in events:
        if "type" in e or e.get("data") == "SPECIAL_END_TOKEN":
            continue
        if e.get("data"):
            parts.append(str(e["data"]))
        if e.get("error"):
            parts.append(f"[錯誤細節] {e['error']}")
    return "\n".join(parts)


def _side_events(events: list[dict]) -> list[str]:
    out = []
    for e in events:
        if e.get("type") == "status":
            out.append(f"進度:{e.get('text')}")
        elif e.get("type") in ("courses", "plans"):
            out.append(f"{e['type']}:{len(e.get(e['type']) or [])} 筆")
    return out


def _reply_at(events: list[dict]) -> float | None:
    """回覆(本文或錯誤)送到瀏覽器的時間。不用串流結束時間:結束記號之前還有
    Langfuse flush,端點連不上時會多卡十幾秒,那段和這裡量的 L2 無關。"""
    for e in events:
        if "type" not in e and e.get("data") != "SPECIAL_END_TOKEN":
            return e["t"]
    return None


def _fmt_sec(t: float | None) -> str:
    return "—(沒有回覆)" if t is None else f"{t}s"


def _judge(name: str, result: dict) -> tuple[bool, list[str]]:
    spec = SCENARIOS[name]
    if "crash" in result:
        return False, [f"執行失敗:{result['crash']}"]
    shown = _shown(result["events"])
    reasons = []
    if not shown.strip():
        reasons.append("畫面上沒有任何回覆")
    elif spec["expect"] not in shown:
        reasons.append(f"沒有出現「{spec['expect']}」")
    reasons += [f"出現了「{f}」" for f in spec["forbid"] if f in shown]
    reply_at = _reply_at(result["events"])
    if "max_elapsed" in spec and reply_at is not None and reply_at > spec["max_elapsed"]:
        reasons.append(f"回覆在 {reply_at}s 才送到,超過 {spec['max_elapsed']:g}s")
    return not reasons, reasons


def _run_tree(root: Path, label: str) -> dict:
    results = {}
    for name in SCENARIOS:
        print(f"  [{label}] {name} …", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--root", str(root)],
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_SEC,
            )
            last = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
            results[name] = (
                json.loads(last[-1][len("RESULT "):])
                if last
                else {"crash": f"exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1:]}"}
            )
        except subprocess.TimeoutExpired:
            results[name] = {"crash": f"超過 {WORKER_TIMEOUT_SEC}s 沒有結束"}
        ok, reasons = _judge(name, results[name])
        results[name]["ok"], results[name]["reasons"] = ok, reasons
        print(f"      {'✓' if ok else '✗'} {'; '.join(reasons) or '符合預期'}", flush=True)
    return results


def _block(text: str) -> str:
    text = text.strip() or "(什麼都沒有)"
    return "```text\n" + (text[:600] + "\n…(截斷)" if len(text) > 600 else text) + "\n```"


def _write_report(out_dir: Path, current: dict, baseline: dict | None, rev: str | None, model: str) -> Path:
    lines = [
        "# 串流路徑 L2 證據",
        "",
        f"- 產生時間:{dt.datetime.now():%Y-%m-%d %H:%M}",
        f"- 模型:`{model}`",
        "- 量測位置:`app.py::generate_streaming` 送出的 SSE 事件(前端 Chat.vue 實際收到的內容)",
        "- 「畫面上的文字」= 回覆本文 + 錯誤細節欄位(Chat.vue 兩個都會顯示)",
    ]
    if baseline is not None:
        lines.append(f"- 舊版:commit `{rev}`(修正前);新版:目前的工作目錄")
    lines += ["", "## 總覽", ""]
    header = "| 情境 | 觸發 | " + ("舊版 | " if baseline else "") + "新版 |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for name, spec in SCENARIOS.items():
        cells = [spec["title"], spec["kind"]]
        for res in ([baseline[name]] if baseline else []) + [current[name]]:
            cells.append("✓ 符合" if res["ok"] else "✗ " + ";".join(res["reasons"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "判定標準:畫面上必須出現該情境的友善訊息、不可出現例外型別或內容;"
        f"逾時情境另外要求回覆在 {TIMEOUT_SEC + 2:g} 秒內送到;JSON 情境要求畫面上沒有 text_to_sql 產生的條件 JSON。",
        "",
        "「串流結束」比「回覆送達」晚很多時,通常是結束前的 Langfuse flush 在等連不上的端點,與 L2 無關。",
        "",
        "「模型最後沒回答」只能模擬:真模型回空白是偶發的(小模型偶爾如此),無法刻意觸發。"
        "模擬的事件格式取自 LangGraph `astream_events(version=\"v2\")` 的實際輸出。",
    ]
    for name, spec in SCENARIOS.items():
        lines += ["", f"## {spec['title']}", "", f"問題:「{spec['question']}」", ""]
        for label, res in ([("舊版", baseline[name])] if baseline else []) + [("新版", current[name])]:
            if "crash" in res:
                lines += [f"**{label}**:執行失敗 —— {res['crash']}", ""]
                continue
            side = _side_events(res["events"])
            lines += [
                f"**{label}**(回覆送達 {_fmt_sec(_reply_at(res['events']))},串流結束 {res['elapsed']}s;"
                f"{'✓' if res['ok'] else '✗ ' + ';'.join(res['reasons'])})",
                "",
            ]
            if side:
                lines += ["側通道事件:" + "、".join(side), ""]
            lines += ["畫面上的文字:", "", _block(_shown(res["events"])), ""]

    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", help="同時跑這個 commit 的版本做對照,例如 34d48d3")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--root", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        result = _worker(Path(args.root), args.worker)
        print("RESULT " + json.dumps(result, ensure_ascii=False))
        return 0

    sys.path.insert(0, str(PROJECT_ROOT))
    from paths import OPENAI_MODEL

    model = OPENAI_MODEL or "unknown"
    print("新版(目前的工作目錄)")
    current = _run_tree(PROJECT_ROOT, "新版")

    baseline = None
    if args.baseline:
        tmp = Path(tempfile.mkdtemp(prefix="stream-l2-baseline-"))
        tree = tmp / "tree"
        subprocess.run(["git", "worktree", "add", "--detach", str(tree), args.baseline],
                       cwd=PROJECT_ROOT, check=True, capture_output=True)
        try:
            shutil.copy(PROJECT_ROOT / ".env", tree / ".env")
            print(f"舊版(commit {args.baseline})")
            baseline = _run_tree(tree, "舊版")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=PROJECT_ROOT, check=False)
            shutil.rmtree(tmp, ignore_errors=True)

    out_dir = HERE / "reports" / f"{dt.datetime.now():%Y%m%d-%H%M}-{model.replace(':', '_').replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"model": model, "baseline_rev": args.baseline, "current": current, "baseline": baseline},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = _write_report(out_dir, current, baseline, args.baseline, model)
    print(f"\n報表:{report.relative_to(PROJECT_ROOT)}")
    return 0 if all(r["ok"] for r in current.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
