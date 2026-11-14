"""一鍵跑完 tests/ 底下所有測試檔。

    uv run python -m tests                 # 全部
    uv run python -m tests scheduler mcp   # 檔名含 scheduler 或 mcp 的
    uv run python -m tests -v              # 每個檔的完整輸出都印出來
    uv run python -m tests --require-db    # 連不上資料庫就算失敗(本機完整驗證用)

測試檔是獨立腳本(不依賴 pytest):各自印出「結果:N passed, M failed[, K skipped]」,
失敗時 exit code 非 0。這裡逐一用子程序跑 —— 每個檔都會登記 L3 規則、改全域狀態,
放在同一個程序裡會互相干擾。

結束碼:任何一個檔失敗、崩潰或沒印出結果行 → 1;否則 0。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PER_FILE_TIMEOUT_SEC = 600

_RESULT = re.compile(r"結果:(\d+) passed, (\d+) failed(?:, (\d+) skipped)?")


def _run(module: str, verbose: bool) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", f"tests.{module}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=PER_FILE_TIMEOUT_SEC,
        )
        output, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        output, code = f"{e.stdout or ''}{e.stderr or ''}\n(超過 {PER_FILE_TIMEOUT_SEC}s 逾時)", -1

    if verbose:
        print(output)
    m = _RESULT.findall(output)
    passed, failed, skipped = (int(x or 0) for x in m[-1]) if m else (0, 0, 0)
    ok = code == 0 and bool(m) and failed == 0
    return {
        "module": module,
        "ok": ok,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "code": code,
        "has_result": bool(m),
        "elapsed": time.time() - start,
        "output": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m tests")
    parser.add_argument("patterns", nargs="*", help="只跑檔名含這些字的測試檔")
    parser.add_argument("-v", "--verbose", action="store_true", help="印出每個檔的完整輸出")
    parser.add_argument("--require-db", action="store_true", help="連不上資料庫就算失敗")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    from tests._helpers import db_available

    has_db = db_available()
    print(f"PostgreSQL:{'連得上' if has_db else '連不上 —— 需要資料庫的測試會標成略過'}")
    if args.require_db and not has_db:
        print("--require-db:沒有資料庫,停止。先 `docker compose up -d postgres`。")
        return 1

    modules = sorted(p.stem for p in TESTS_DIR.glob("test_*.py"))
    if args.patterns:
        modules = [m for m in modules if any(p in m for p in args.patterns)]
    if not modules:
        print("沒有符合的測試檔")
        return 1

    results = []
    for module in modules:
        print(f"  … {module}", end="", flush=True)
        r = _run(module, args.verbose)
        results.append(r)
        mark = "✓" if r["ok"] else "✗"
        skip = f", {r['skipped']} skipped" if r["skipped"] else ""
        note = "" if r["has_result"] else f"  (沒有結果行,exit {r['code']})"
        print(f"\r  {mark} {module}: {r['passed']} passed, {r['failed']} failed{skip}"
              f"  [{r['elapsed']:.1f}s]{note}")

    bad = [r for r in results if not r["ok"]]
    for r in bad:
        if args.verbose:
            continue
        print(f"\n──── {r['module']} 的輸出(最後 40 行)────")
        print("\n".join(r["output"].rstrip().splitlines()[-40:]))

    total = {k: sum(r[k] for r in results) for k in ("passed", "failed", "skipped")}
    print(
        f"\n總計:{len(results)} 個檔,{total['passed']} passed, {total['failed']} failed, "
        f"{total['skipped']} skipped;{len(bad)} 個檔未通過"
    )
    if total["skipped"] and not has_db:
        print("(略過的項目大多需要 PostgreSQL;本機完整驗證請開資料庫後加 --require-db)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
