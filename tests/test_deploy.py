"""ops.deploy:換版部署(匯入新資料庫 → 檢查 → 改名切換 → 換 contract → 重啟 → 失敗回滾)。

    uv run python -m tests.test_deploy

純函數的部分不需要資料庫。整合測試要一個可以亂建、亂刪資料庫的 PostgreSQL(ParadeDB):

    DEPLOY_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5435/deploytest \
    DEPLOY_PSQL="docker exec -i <container> psql -U postgres" \
    uv run python -m tests.test_deploy

它只動 deploytest 和它的 _next、_prev、_old,不碰 academic。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from ops import deploy as D
from ops import pg_swap as S

_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


def raises(fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
    except SystemExit:
        return True
    return False


CONTRACT = {"schema_version": 9, "embed_model": "BAAI/bge-base-zh-v1.5", "embed_dim": 768,
            "sqlite_embed_model": "BAAI/bge-m3", "course_year": "114", "course_semester": "2",
            "v4_embed_model": "intfloat/multilingual-e5-large-instruct", "v4_embed_dim": 1024,
            "v4_bm25_tokens": "char12"}


def contract_for(tag: str) -> dict:
    """每一版的 contract 都不一樣,回滾時才分得出換回的是哪一份。"""
    return CONTRACT | {"test_tag": tag}


def _artifact_sql(release_year: str = "114", meta_override: dict | None = None, tag: str = "a") -> str:
    meta = {k: str(v) for k, v in contract_for(tag).items() if k != "sqlite_embed_model"} | (meta_override or {})
    rows = ",\n".join(f"('{k}', '{v}')" for k, v in meta.items())
    return f"""
CREATE SCHEMA IF NOT EXISTS public;
CREATE TABLE public.schema_meta (key text PRIMARY KEY, value text NOT NULL);
INSERT INTO public.schema_meta VALUES
{rows};
CREATE TABLE public.course (id text, y text, s text, name text);
INSERT INTO public.course VALUES ('1142703834001', '{release_year}', '2', '自然語言處理');
CREATE SCHEMA v4;
CREATE TABLE v4.doc (course_id text PRIMARY KEY, toks text);
INSERT INTO v4.doc VALUES ('1142703834001', '自 然 自然');
"""


def make_release(root: Path, name: str = "114-2_20261102-0928", sql: str | None = None,
                 tag: str = "a", corrupt: bool = False) -> Path:
    import yaml

    d = root / name
    d.mkdir(parents=True)
    with gzip.open(d / "data.sql.gz", "wt", encoding="utf-8") as f:
        f.write(sql if sql is not None else _artifact_sql(tag=tag))
    (d / "contract.yaml").write_text(yaml.safe_dump(contract_for(tag), allow_unicode=True), encoding="utf-8")
    files = {}
    for p in (d / "data.sql.gz", d / "contract.yaml"):
        files[p.name] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    if corrupt:
        files["data.sql.gz"]["sha256"] = "0" * 64
    (d / "manifest.json").write_text(json.dumps({"release": name, "semester": "114-2", "files": files}))
    (root / "current").unlink(missing_ok=True)
    (root / "current").symlink_to(name)
    return d


# ── 純函數 ────────────────────────────────────────────────────────────────────


def test_names() -> None:
    print("[資料庫名稱]")
    url = "postgresql://postgres:postgres@localhost:5432/academic"
    check(S.database_name(url) == "academic", "從 DATABASE_URL 取資料庫名稱")
    check(S.with_database(url, "postgres") == "postgresql://postgres:postgres@localhost:5432/postgres",
          "換成維護用資料庫 postgres")
    check(S.names("academic") == ("academic", "academic_next", "academic_prev", "academic_old"),
          "next / prev / old 命名")


def test_contract_comparison() -> None:
    print("[schema_meta 與 contract.yaml 比對]")
    meta = {k: str(v) for k, v in CONTRACT.items() if k != "sqlite_embed_model"}
    check(S.contract_diffs(meta, CONTRACT) == [], "一致時沒有差異")
    check(any("schema_version" in d for d in S.contract_diffs(meta | {"schema_version": "8"}, CONTRACT)),
          "版本不同會列出來")
    check(any("v4_embed_model" in d for d in S.contract_diffs(
        {k: v for k, v in meta.items() if k != "v4_embed_model"}, CONTRACT)), "contract 有、資料庫沒有的鍵也算差異")


def test_fetch_and_verify() -> None:
    print("[取得成品並驗證 sha256]")
    with tempfile.TemporaryDirectory() as d:
        root, stage = Path(d) / "releases", Path(d) / "stage"
        make_release(root)
        rel = D.fetch(str(root / "current"), stage)
        check(rel.name == "114-2_20261102-0928", "跟著 current 找到版本名稱")
        check((rel.dir / "data.sql.gz").exists() and (rel.dir / "contract.yaml").exists(), "兩個檔案都複製到暫存區")
        check(rel.contract["course_year"] == "114", "讀到成品附的 contract.yaml")
    with tempfile.TemporaryDirectory() as d:
        root, stage = Path(d) / "releases", Path(d) / "stage"
        make_release(root, corrupt=True)
        check(raises(D.fetch, str(root / "current"), stage), "sha256 不符就拒絕")
    with tempfile.TemporaryDirectory() as d:
        root, stage = Path(d) / "releases", Path(d) / "stage"
        make_release(root, name="bad/../x")
        check(raises(D.fetch, str(root / "current"), stage), "版本名稱帶 / 或 .. 就拒絕(會寫到暫存區外)")
    check(raises(D.fetch, "http://example.invalid/releases/current", Path(".")), "明文 http 來源不收")


def test_swap_contract_file() -> None:
    print("[contract.yaml 換版與還原]")
    with tempfile.TemporaryDirectory() as d:
        live = Path(d) / "contract.yaml"
        live.write_text("schema_version: 8\n")
        new = Path(d) / "new.yaml"
        new.write_text("schema_version: 9\n")
        D.install_contract(new, live)
        check(live.read_text() == "schema_version: 9\n", "換上新的 contract.yaml")
        check(live.with_suffix(".yaml.prev").read_text() == "schema_version: 8\n", "舊的留在 .prev")
        D.restore_contract(live)
        check(live.read_text() == "schema_version: 8\n", "回滾時還原")


# ── 整合:對可拋棄的 PostgreSQL ───────────────────────────────────────────────


def _db_test() -> str | None:
    url = os.environ.get("DEPLOY_TEST_DATABASE_URL")
    if not url or not os.environ.get("DEPLOY_PSQL"):
        print("[整合測試] 略過:沒有 DEPLOY_TEST_DATABASE_URL / DEPLOY_PSQL")
        return None
    return url


def _cleanup(url: str) -> None:
    admin = S.connect_admin(url)
    try:
        for name in S.names(S.database_name(url)):
            S.drop_database(admin, name)
    finally:
        admin.close()


def _deployed(url: str) -> str | None:
    return S.deployed_release(url)


def _cfg(d: Path, url: str, **kw) -> D.Config:
    base = dict(source=str(d / "releases" / "current"), stage=d / "stage", database_url=url,
                psql=os.environ["DEPLOY_PSQL"], restart_cmd="", health_url="", contract_path=d / "contract.yaml")
    return D.Config(**(base | kw))


def _tag(path: Path) -> str | None:
    import yaml

    return (yaml.safe_load(path.read_text()) or {}).get("test_tag")


def _admin(url: str, *statements: str) -> None:
    admin = S.connect_admin(url)
    try:
        with admin.cursor() as cur:
            for st in statements:
                cur.execute(st)
    finally:
        admin.close()


def _initdb_live(url: str, d: Path, tag: str) -> None:
    """模擬 db/init 首次還原出來的線上資料庫:有資料、沒有 deploy_info。"""
    import yaml

    db = S.database_name(url)
    _admin(url, f'CREATE DATABASE "{db}"')
    S.ensure_extensions(url)
    gz = d / "initdb.sql.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write(_artifact_sql(tag=tag))
    S.restore(os.environ["DEPLOY_PSQL"], db, gz, cwd=str(d))
    (d / "contract.yaml").write_text(yaml.safe_dump(contract_for(tag)))


def test_deploy_and_rollback() -> None:
    url = _db_test()
    if not url:
        return
    print("[整合:部署、切換、回滾]")
    _cleanup(url)
    db, nxt, prev, old = S.names(S.database_name(url))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "contract.yaml").write_text("schema_version: 8\n")
        cfg = _cfg(d, url)

        make_release(d / "releases", "114-2_20261102-0928", tag="a")
        check(D.deploy(cfg) == "deployed", "第一次部署:沒有現行資料庫也能上線")
        check(_deployed(url) == "114-2_20261102-0928" and _tag(d / "contract.yaml") == "a", "版本與 contract 都是 a")
        check(D.deploy(cfg) == "up to date", "同一版再跑一次什麼都不做")
        (d / "contract.yaml").write_text("schema_version: 8\n")
        check(D.deploy(cfg) == "repaired" and _tag(d / "contract.yaml") == "a",
              "同一版但 contract.yaml 被改過:依線上資料庫改回")

        make_release(d / "releases", "114-2_20261103-0300", tag="b")
        check(D.deploy(cfg) == "deployed", "第二版上線")
        check(_deployed(url) == "114-2_20261103-0300" and _tag(d / "contract.yaml") == "b", "現行是 b")
        check(S.exists(url, prev), "舊版留在 _prev")

        make_release(d / "releases", "114-2_20261104-0300",
                     sql=_artifact_sql(meta_override={"schema_version": "8"}, tag="c"), tag="c")
        check(raises(D.deploy, cfg), "資料包的 schema_meta 和它附的 contract 不符:切換前就停")
        check(_deployed(url) == "114-2_20261103-0300" and not S.exists(url, nxt), "線上沒動,_next 已刪")

        make_release(d / "releases", "114-2_20261105-0300", sql=_artifact_sql(release_year="113", tag="d"), tag="d")
        check(raises(D.deploy, cfg), "資料包沒有 contract 學期的課:切換前就停")

        make_release(d / "releases", "114-2_20261106-0300", tag="e")
        check(raises(D.deploy, _cfg(d, url, health_url="http://127.0.0.1:9/api/terms", health_timeout=3)),
              "重啟後健康檢查失敗就回報錯誤")
        check(_deployed(url) == "114-2_20261103-0300" and _tag(d / "contract.yaml") == "b",
              "自動回滾:資料庫和 contract 都回到 b")
        check(not S.exists(url, old), "暫存的 _old 已清掉")

        make_release(d / "releases", "114-2_20261107-0300", tag="f")
        check(raises(D.deploy, _cfg(d, url, restart_cmd="false")), "重啟指令失敗也算部署失敗")
        check(_deployed(url) == "114-2_20261103-0300" and _tag(d / "contract.yaml") == "b",
              "重啟失敗:同樣回滾到 b")

        check(D.rollback(cfg) == "114-2_20261102-0928" and _tag(d / "contract.yaml") == "a",
              "手動 rollback:切回 a,contract 也是 a")
        check(raises(D.rollback, _cfg(d, url, health_url="http://127.0.0.1:9/api/terms", health_timeout=3)),
              "回滾後健康檢查失敗就回報")
        check(_deployed(url) == "114-2_20261102-0928" and _tag(d / "contract.yaml") == "a", "並換回原本的 a")
    _cleanup(url)


def test_initdb_live_and_recovery() -> None:
    url = _db_test()
    if not url:
        return
    print("[整合:db/init 來的線上資料庫、中斷修復、無法回滾的情況]")
    _cleanup(url)
    db, nxt, prev, old = S.names(S.database_name(url))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _initdb_live(url, d, tag="init")
        cfg = _cfg(d, url)
        check(raises(D.rollback, cfg), "沒有上一版:rollback 在改名前就停")
        check(S.exists(url, db), "線上還在")

        make_release(d / "releases", "114-2_20261102-0928", tag="a")
        check(D.deploy(cfg) == "deployed", "從 db/init 的資料庫換到 a")
        check(S.deploy_info(url, prev).get("release") == D.UNRECORDED, "舊的線上資料庫補記為 (initdb)")
        check(D.rollback(cfg) == D.UNRECORDED and _tag(d / "contract.yaml") == "init",
              "回滾到 db/init 那版,contract 換回它當時的")

        # 模擬切換到一半被殺:線上改成 prev 之後、next 改成線上之前
        D.rollback(cfg)                                   # 回到 a,prev 是 initdb
        _admin(url, f'ALTER DATABASE "{prev}" RENAME TO "{old}"', f'ALTER DATABASE "{db}" RENAME TO "{prev}"',
               f'ALTER DATABASE "{prev}" ALLOW_CONNECTIONS false')
        check(not S.exists(url, db), "(模擬)線上資料庫不見了")
        check(D.deploy(cfg) == "up to date", "下一次 deploy 先修復再判斷")
        check(_deployed(url) == "114-2_20261102-0928" and S.deploy_info(url, prev).get("release") == D.UNRECORDED,
              "修復後線上是 a、上一版是 initdb")
        check(not S.exists(url, old), "沒有殘留 _old")
    _cleanup(url)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
