"""換上 course-data-prep 的新資料包,不靠 db/init/(那只在資料庫第一次啟動時執行)。

    uv run python -m ops.deploy deploy             # 有新版就換,沒有就什麼都不做(適合排程)
    uv run python -m ops.deploy deploy --force     # 同一版也重換一次
    uv run python -m ops.deploy rollback           # 切回上一版
    uv run python -m ops.deploy status

流程:
  1. 讀來源的 manifest.json;版本名稱和線上資料庫 deploy_info 記的一樣就結束
  2. 把 data.sql.gz、contract.yaml 複製或下載到暫存區,比對 manifest 的 sha256
  3. 建空的 academic_next 並匯入;線上服務照常
  4. 檢查 academic_next:schema_meta 和成品附的 contract.yaml 一致、有該學期的課、有 v4.doc
  5. 改名切換(見 ops/pg_swap.py),換上新的 contract.yaml,重啟 app
  6. 打健康檢查網址;失敗就把資料庫和 contract.yaml 換回去、再重啟一次

第 3、4 步失敗時線上完全沒被動到。中斷只在第 5 步:改名約一秒,加上 app 重啟的時間;
這段時間進來的查詢會失敗。資料庫裡只有資料包的內容(課表存在記憶體),整個換掉不會丟使用者資料。

設定都在 paths.py(DEPLOY_*):來源可以是本機資料夾(例如 rsync 過來的 releases/current)
或 http(s) 網址。
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

import paths
from ops import pg_swap as S

FILES = ("data.sql.gz", "contract.yaml")
KEEP_STAGED = 3
# 版本名稱會變成暫存區底下的資料夾名,不能帶 / 或 ..
_RELEASE_NAME = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")
UNRECORDED = "(initdb)"


@dataclass(frozen=True)
class Config:
    source: str
    stage: Path
    database_url: str
    psql: str
    restart_cmd: str
    health_url: str
    contract_path: Path
    health_timeout: int = 180


@dataclass(frozen=True)
class Release:
    name: str
    dir: Path
    contract: dict
    contract_text: str


def _log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def _is_url(source: str) -> bool:
    if source.startswith("http://"):
        # manifest 和 sha256 從同一個來源來,只能擋傳輸損壞、擋不了竄改;資料包又是以
        # superuser 身分交給 psql 執行(\! 可以跑 shell),所以明文 http 一律不收。
        raise SystemExit("[deploy] DEPLOY_SOURCE 只接受本機資料夾或 https 網址")
    return source.startswith("https://")


def _read(source: str, name: str) -> bytes:
    if _is_url(source):
        with urllib.request.urlopen(f"{source.rstrip('/')}/{name}", timeout=60) as r:
            return r.read()
    return (Path(source) / name).read_bytes()


def _copy(source: str, name: str, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    if _is_url(source):
        with urllib.request.urlopen(f"{source.rstrip('/')}/{name}", timeout=60) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    else:
        shutil.copyfile(Path(source) / name, tmp)
    os.replace(tmp, dst)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(source: str, stage: Path) -> Release:
    """manifest 指名的檔案放進 stage/<版本>/;已經在那裡且 sha256 相符就不重抓。"""
    if not source:
        raise SystemExit("[deploy] 沒有設定來源:DEPLOY_SOURCE(releases/current 資料夾或網址)")
    manifest = json.loads(_read(source, "manifest.json"))
    name = str(manifest.get("release", ""))
    if not _RELEASE_NAME.match(name):
        raise SystemExit(f"[deploy] manifest 的版本名稱不合法:{name!r}")
    d = stage / name
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        want = manifest["files"][f]["sha256"]
        dst = d / f
        if not (dst.exists() and _sha256(dst) == want):
            _copy(source, f, dst)
        if _sha256(dst) != want:
            dst.unlink()
            raise SystemExit(f"[deploy] {f} 的 sha256 和 manifest 不符,不部署 {name}")
    (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    text = (d / "contract.yaml").read_text(encoding="utf-8")
    return Release(name, d, yaml.safe_load(text), text)


def _prune(stage: Path, keep: str) -> None:
    dirs = sorted((p for p in stage.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in dirs[KEEP_STAGED:]:
        if p.name != keep:
            shutil.rmtree(p)


def install_contract(new: Path, live: Path) -> None:
    if live.exists():
        shutil.copyfile(live, live.with_suffix(".yaml.prev"))
    tmp = live.with_suffix(".yaml.tmp")
    shutil.copyfile(new, tmp)
    os.replace(tmp, live)


def restore_contract(live: Path) -> None:
    prev = live.with_suffix(".yaml.prev")
    if prev.exists():
        os.replace(prev, live)


def _write_contract(text: str, live: Path) -> None:
    tmp = live.with_suffix(".yaml.rollback")      # not .yaml.tmp: install_contract writes that one
    tmp.write_text(text, encoding="utf-8")
    install_contract(tmp, live)
    tmp.unlink(missing_ok=True)


def sync_contract(cfg: Config) -> bool:
    """contract.yaml 改成線上資料庫 deploy_info 記的那份;有改動就回傳 True。

    每個資料庫都帶著自己的 contract,所以不論是回滾、修復中斷,還是手動改過檔案,
    都以線上資料庫為準,不靠 .prev 檔(連續失敗時 .prev 會被覆蓋)。"""
    text = S.deploy_info(cfg.database_url, S.database_name(cfg.database_url)).get("contract")
    if not text:
        return False
    current = cfg.contract_path.read_text(encoding="utf-8") if cfg.contract_path.exists() else ""
    if current == text:
        return False
    _write_contract(text, cfg.contract_path)
    return True


def _lock(stage: Path):
    stage.mkdir(parents=True, exist_ok=True)
    f = (stage / ".deploy.lock").open("w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise SystemExit("[deploy] 另一個 deploy / rollback 正在執行,這次不動作") from None
    return f


def _recover(cfg: Config) -> None:
    for step in S.recover(cfg.database_url):
        _log(f"修復上次中斷的切換:{step}")


def _stamp_live(cfg: Config) -> None:
    """線上資料庫若是 db/init 首次還原來的(沒有 deploy_info),切換前補記它和目前的
    contract.yaml,之後回滾到它時才知道要換回哪份 contract。"""
    db = S.database_name(cfg.database_url)
    if S.exists(cfg.database_url, db) and not S.deploy_info(cfg.database_url, db):
        text = cfg.contract_path.read_text(encoding="utf-8") if cfg.contract_path.exists() else ""
        S.write_deploy_info(cfg.database_url, UNRECORDED, text)


def restart(cmd: str) -> None:
    if not cmd:
        _log("DEPLOY_RESTART_CMD 是空的,沒有重啟 app;請自行重啟,讓它重新檢查 contract")
        return
    _log(f"重啟:{cmd}")
    subprocess.run(shlex.split(cmd), cwd=paths.PROJECT_ROOT, check=True)


def healthy(url: str, timeout: int) -> bool:
    if not url:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(3)
    return False


def deploy(cfg: Config, force: bool = False) -> str:
    with _lock(cfg.stage):   # 排程重疊時第二個直接結束
        _recover(cfg)
        rel = fetch(cfg.source, cfg.stage)
        live = S.deployed_release(cfg.database_url)
        if live == rel.name and not force:
            if sync_contract(cfg):     # 例如上次在換 contract 前被中斷
                _log("contract.yaml 和線上資料庫不符,已改回並重啟")
                restart(cfg.restart_cmd)
                return "repaired"
            _log(f"線上已是 {rel.name}")
            return "up to date"
        db, nxt, _, _ = S.names(S.database_name(cfg.database_url))
        next_url = S.with_database(cfg.database_url, nxt)
        _log(f"{live or '(未記錄的版本)'} → {rel.name}:匯入 {nxt}")
        admin = S.connect_admin(cfg.database_url)
        try:
            S.create_empty(admin, nxt)
        finally:
            admin.close()
        S.ensure_extensions(next_url)
        t0 = time.time()
        S.restore(cfg.psql, nxt, rel.dir / "data.sql.gz", cwd=paths.PROJECT_ROOT)
        _log(f"匯入完成,{time.time() - t0:.0f} 秒")
        problems = S.verify(next_url, rel.contract)
        if problems:
            admin = S.connect_admin(cfg.database_url)
            try:
                S.drop_database(admin, nxt)
            finally:
                admin.close()
            raise SystemExit(f"[deploy] {rel.name} 沒通過檢查,線上未變動:\n" + "\n".join(f"  - {p}" for p in problems))
        S.write_deploy_info(next_url, rel.name, rel.contract_text)
        _stamp_live(cfg)

        try:
            S.swap_in(cfg.database_url)
        except BaseException:
            _recover(cfg)
            raise
        try:
            install_contract(rel.dir / "contract.yaml", cfg.contract_path)
            _log(f"{db} 已切到 {rel.name}")
            restart(cfg.restart_cmd)
            if not healthy(cfg.health_url, cfg.health_timeout):
                raise RuntimeError(f"健康檢查失敗({cfg.health_url})")
        except BaseException as e:     # 含 Ctrl-C:新版不能在沒驗過的狀態下留在線上
            _log(f"{type(e).__name__}: {e};回滾")
            S.undo_swap(cfg.database_url)
            if not sync_contract(cfg):
                restore_contract(cfg.contract_path)
            try:
                restart(cfg.restart_cmd)
            except Exception as re_err:  # noqa: BLE001 - 回滾已完成,重啟失敗另外回報
                _log(f"回滾後重啟失敗:{re_err};請手動重啟 app")
            raise SystemExit(f"[deploy] {rel.name} 沒能上線,已回滾到 {live or '前一版'}") from e
        S.finalize(cfg.database_url)
        _prune(cfg.stage, rel.name)
        _log(f"完成:{rel.name}")
        return "deployed"


def rollback(cfg: Config) -> str:
    with _lock(cfg.stage):
        _recover(cfg)
        _, _, prev, _ = S.names(S.database_name(cfg.database_url))
        target = S.deploy_info(cfg.database_url, prev)
        if not target.get("contract"):
            # 不知道上一版配哪份 contract,切過去 app 很可能起不來:改名之前就停
            raise SystemExit(f"[deploy] {prev} 不存在或沒有記錄它的 contract,不回滾")
        _stamp_live(cfg)
        S.rollback_live(cfg.database_url)
        sync_contract(cfg)
        restart(cfg.restart_cmd)
        if not healthy(cfg.health_url, cfg.health_timeout):
            _log("回滾後健康檢查失敗,換回原本的版本")
            S.rollback_live(cfg.database_url)
            sync_contract(cfg)
            restart(cfg.restart_cmd)
            raise SystemExit("[deploy] 回滾後健康檢查失敗,已換回原本的版本")
        _log(f"已切回 {target.get('release')}")
        return target["release"]


def status(cfg: Config) -> None:
    # 只讀:中斷留下的狀態在這裡顯示,修復交給下一次 deploy / rollback
    db, _, prev, _ = S.names(S.database_name(cfg.database_url))
    for label, name in (("線上", db), ("上一版", prev)):
        info = S.deploy_info(cfg.database_url, name)
        shown = info.get("release") or ("(未記錄的版本)" if S.exists(cfg.database_url, name) else "(無)")
        print(f"{label:4s} {name:16s} {shown}  {info.get('deployed_at', '')}")


def config_from_env(source: str | None = None) -> Config:
    return Config(source=source or paths.DEPLOY_SOURCE, stage=Path(paths.DEPLOY_STAGE),
                  database_url=paths.DATABASE_URL, psql=paths.DEPLOY_PSQL, restart_cmd=paths.DEPLOY_RESTART_CMD,
                  health_url=paths.DEPLOY_HEALTH_URL, contract_path=Path(paths.CONTRACT_PATH))


def main() -> None:
    p = argparse.ArgumentParser(description="換上 course-data-prep 的新資料包")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deploy")
    d.add_argument("--source", help="覆寫 DEPLOY_SOURCE")
    d.add_argument("--force", action="store_true", help="同一版也重新匯入")
    sub.add_parser("rollback")
    sub.add_parser("status")
    a = p.parse_args()
    cfg = config_from_env(getattr(a, "source", None))
    if a.cmd == "deploy":
        deploy(cfg, force=a.force)
    elif a.cmd == "rollback":
        rollback(cfg)
    else:
        status(cfg)


if __name__ == "__main__":
    sys.exit(main())
