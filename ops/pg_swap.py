"""換版用的資料庫操作:建空的 _next、匯入、檢查、改名切換、復原。

四個名字(以 DATABASE_URL 的資料庫 academic 為例):

    academic       線上版本,app 連的就是它
    academic_next  正在匯入、檢查的新版本;沒通過就刪掉,線上完全沒被碰到
    academic_prev  上一版,手動 rollback 用
    academic_old   切換時暫放的上上版:健康檢查通過才刪,失敗時用來把 _prev 還原

改名要求那個資料庫沒有連線。app 每次查詢都重新連線,所以先關掉它接受新連線、
再結束現有連線,改名後再打開。切換只在幾個 ALTER DATABASE 之間,約一秒。
"""
from __future__ import annotations

import gzip
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2

# contract.yaml 裡不寫進 schema_meta 的鍵(舊 SQLite 路徑專用)
_NOT_IN_SCHEMA_META = {"sqlite_embed_model"}


def database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/")


def with_database(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{name}"))


def names(db: str) -> tuple[str, str, str, str]:
    return db, f"{db}_next", f"{db}_prev", f"{db}_old"


def _ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise SystemExit(f"[deploy] 不接受的資料庫名稱 {name!r}")
    return f'"{name}"'


def connect_admin(url: str):
    conn = psycopg2.connect(with_database(url, "postgres"))
    conn.autocommit = True
    return conn


def exists(url: str, name: str) -> bool:
    admin = connect_admin(url)
    try:
        return _exists(admin, name)
    finally:
        admin.close()


def _exists(admin, name: str) -> bool:
    with admin.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        return cur.fetchone() is not None


def drop_database(admin, name: str) -> None:
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {_ident(name)} WITH (FORCE)")


def create_empty(admin, name: str) -> None:
    """新資料庫從 template1 建;ParadeDB 映像的 template1 已帶 vector 與 pg_search。"""
    drop_database(admin, name)
    with admin.cursor() as cur:
        cur.execute(f"CREATE DATABASE {_ident(name)}")


def ensure_extensions(url: str) -> None:
    conn = psycopg2.connect(url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    finally:
        conn.close()


def restore(psql: str, db: str, gz: Path, cwd: str) -> None:
    """把 data.sql.gz 串流給 psql;任何一行 SQL 出錯就停(ON_ERROR_STOP)。"""
    cmd = shlex.split(psql) + ["-v", "ON_ERROR_STOP=1", "-q", "-d", db]
    with gzip.open(gz, "rb") as src, subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as p:
        try:
            for block in iter(lambda: src.read(1 << 20), b""):
                p.stdin.write(block)
        except BrokenPipeError:
            pass      # psql 已經因錯誤結束,下面讀 stderr
        finally:
            p.stdin.close()
        err = p.stderr.read().decode("utf-8", "replace")
    if p.returncode:
        tail = "\n".join(err.strip().splitlines()[-8:])
        raise SystemExit(f"[deploy] 匯入 {db} 失敗(psql exit {p.returncode}):\n{tail}")


def contract_diffs(meta: dict, contract: dict) -> list[str]:
    return [
        f"{k}: 資料庫是 {meta.get(k)!r},contract.yaml 是 {str(v)!r}"
        for k, v in contract.items()
        if k not in _NOT_IN_SCHEMA_META and str(meta.get(k)) != str(v)
    ]


def _query(url: str, sql: str, params: tuple = ()) -> list[tuple]:
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or None)
            return cur.fetchall()
    finally:
        conn.close()


def verify(url: str, contract: dict) -> list[str]:
    """新資料庫能不能上線:schema_meta 要和它附的 contract.yaml 一致,且有那個學期的課。"""
    if _query(url, "SELECT to_regclass('public.schema_meta')")[0][0] is None:
        return ["沒有 public.schema_meta,這不是 course-data-prep 的成品"]
    problems = contract_diffs(dict(_query(url, "SELECT key, value FROM public.schema_meta")), contract)
    year, sem = str(contract["course_year"]), str(contract["course_semester"])
    n = _query(url, "SELECT count(*) FROM public.course WHERE y::text = %s AND s::text = %s", (year, sem))[0][0]
    if not n:
        problems.append(f"public.course 沒有 {year}-{sem} 的課")
    if "v4_embed_model" in contract:
        if _query(url, "SELECT to_regclass('v4.doc')")[0][0] is None:
            problems.append("contract 有 v4 欄位,但沒有 v4.doc")
        elif not _query(url, "SELECT count(*) FROM v4.doc")[0][0]:
            problems.append("v4.doc 是空的")
    return problems


def write_deploy_info(url: str, release: str, contract_text: str) -> None:
    """記在資料庫裡:改名時跟著走,所以 rollback 知道每個資料庫是哪一版、配哪份 contract。"""
    conn = psycopg2.connect(url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS public.deploy_info (key text PRIMARY KEY, value text NOT NULL)")
            for k, v in (("release", release), ("deployed_at", time.strftime("%Y/%m/%d %H:%M:%S")),
                         ("contract", contract_text)):
                cur.execute("INSERT INTO public.deploy_info VALUES (%s, %s) "
                            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (k, v))
    finally:
        conn.close()


def deploy_info(url: str, name: str) -> dict:
    if not exists(url, name):
        return {}
    target = with_database(url, name)
    if _query(target, "SELECT to_regclass('public.deploy_info')")[0][0] is None:
        return {}
    return dict(_query(target, "SELECT key, value FROM public.deploy_info"))


def deployed_release(url: str) -> str | None:
    return deploy_info(url, database_name(url)).get("release")


def _rename(admin, src: str, dst: str, tries: int = 5) -> None:
    with admin.cursor() as cur:
        for attempt in range(tries):
            cur.execute(f"ALTER DATABASE {_ident(src)} ALLOW_CONNECTIONS false")
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE datname = %s AND pid <> pg_backend_pid()", (src,))
            try:
                cur.execute(f"ALTER DATABASE {_ident(src)} RENAME TO {_ident(dst)}")
                break
            except psycopg2.errors.ObjectInUse:
                if attempt == tries - 1:
                    cur.execute(f"ALTER DATABASE {_ident(src)} ALLOW_CONNECTIONS true")
                    raise
                time.sleep(0.5)
        cur.execute(f"ALTER DATABASE {_ident(dst)} ALLOW_CONNECTIONS true")


def swap_in(url: str) -> None:
    """_next 上線:prev → old、live → prev、next → live。"""
    db, nxt, prev, old = names(database_name(url))
    admin = connect_admin(url)
    try:
        if _exists(admin, prev):
            drop_database(admin, old)
            _rename(admin, prev, old)
        if _exists(admin, db):
            _rename(admin, db, prev)
        _rename(admin, nxt, db)
    finally:
        admin.close()


def undo_swap(url: str) -> None:
    """swap_in 的反向:剛上線的那版刪掉,prev 回到線上,old 回到 prev。"""
    db, nxt, prev, old = names(database_name(url))
    admin = connect_admin(url)
    try:
        drop_database(admin, nxt)
        _rename(admin, db, nxt)
        if _exists(admin, prev):
            _rename(admin, prev, db)
        if _exists(admin, old):
            _rename(admin, old, prev)
        drop_database(admin, nxt)
    finally:
        admin.close()


def finalize(url: str) -> None:
    db, _, _, old = names(database_name(url))
    admin = connect_admin(url)
    try:
        drop_database(admin, old)
    finally:
        admin.close()


def rollback_live(url: str) -> None:
    """線上與 prev 對調。"""
    db, _, prev, old = names(database_name(url))
    admin = connect_admin(url)
    try:
        if not _exists(admin, prev):
            raise SystemExit(f"[deploy] 沒有 {prev},無法回滾")
        drop_database(admin, old)
        _rename(admin, db, old)
        _rename(admin, prev, db)
        _rename(admin, old, prev)
    finally:
        admin.close()
