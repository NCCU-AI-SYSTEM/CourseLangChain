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


def _artifact_sql(release_year: str = "114", meta_override: dict | None = None) -> str:
    meta = {k: str(v) for k, v in CONTRACT.items() if k != "sqlite_embed_model"} | (meta_override or {})
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
                 contract: dict | None = None, corrupt: bool = False) -> Path:
    import yaml

    d = root / name
    d.mkdir(parents=True)
    with gzip.open(d / "data.sql.gz", "wt", encoding="utf-8") as f:
        f.write(sql if sql is not None else _artifact_sql())
    (d / "contract.yaml").write_text(yaml.safe_dump(contract or CONTRACT, allow_unicode=True), encoding="utf-8")
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


def test_deploy_and_rollback() -> None:
    url = _db_test()
    if not url:
        return
    import yaml

    print("[整合:部署、切換、回滾]")
    _cleanup(url)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        live_contract = d / "contract.yaml"
        live_contract.write_text("schema_version: 8\n")
        cfg = D.Config(source=str(d / "releases" / "current"), stage=d / "stage", database_url=url,
                       psql=os.environ["DEPLOY_PSQL"], restart_cmd="", health_url="", contract_path=live_contract)

        make_release(d / "releases", "114-2_20261102-0928")
        check(D.deploy(cfg) == "deployed", "第一次部署:沒有現行資料庫也能上線")
        check(_deployed(url) == "114-2_20261102-0928", "deploy_info 記下版本")
        check(D.deploy(cfg) == "up to date", "同一版再跑一次什麼都不做")

        make_release(d / "releases", "114-2_20261103-0300")
        check(D.deploy(cfg) == "deployed", "第二版上線")
        check(_deployed(url) == "114-2_20261103-0300", "現行是第二版")
        check(S.exists(url, S.names(S.database_name(url))[2]), "舊版留在 _prev")

        make_release(d / "releases", "114-2_20261104-0300", sql=_artifact_sql(meta_override={"schema_version": "8"}))
        check(raises(D.deploy, cfg), "資料包的 schema_meta 和它附的 contract 不符:切換前就停")
        check(_deployed(url) == "114-2_20261103-0300", "現行版本沒被動到")

        make_release(d / "releases", "114-2_20261105-0300", sql=_artifact_sql(release_year="113"))
        check(raises(D.deploy, cfg), "資料包沒有 contract 學期的課:切換前就停")

        bad_health = D.Config(**{**cfg.__dict__, "health_url": "http://127.0.0.1:9/api/terms", "health_timeout": 3})
        make_release(d / "releases", "114-2_20261106-0300")
        check(raises(D.deploy, bad_health), "重啟後健康檢查失敗就回報錯誤")
        check(_deployed(url) == "114-2_20261103-0300", "並自動回滾到前一版")
        check(yaml.safe_load(live_contract.read_text())["schema_version"] == 9, "contract.yaml 也還原成前一版的")
        check(not S.exists(url, S.names(S.database_name(url))[3]), "暫存的 _old 已清掉")

        D.rollback(cfg)
        check(_deployed(url) == "114-2_20261102-0928", "手動 rollback 切回更早一版")
    _cleanup(url)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
