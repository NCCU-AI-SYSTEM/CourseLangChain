"""contract.yaml 與資料成品的一致性檢查。

    uv run python -m tests.test_contract

沿用專案既有的 standalone 風格(自帶 check + 報表,不依賴 pytest)。

為什麼要測這個:兩個 repo 沒有共用程式碼,靠 contract.yaml 對齊。若 course-data-prep
換了 embedding 模型而這邊沒跟上,query 向量與庫裡的向量來自不同模型 —— 檢索**不會報錯**,
只會安靜地回一堆語意無關的課。所以「不一致必須大聲失敗」本身就得有測試釘住。
"""
from __future__ import annotations

import json
import os
import tempfile

import paths

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


def test_contract_loaded() -> None:
    print("[contract.yaml 載入]")
    check(paths.SCHEMA_VERSION >= 1, "schema_version 有值")
    check(bool(paths.EMBED_MODEL), "embed_model 有值")
    check(paths.EMBED_DIM > 0, "embed_dim 為正整數")
    check(paths.COURSE_YEAR.isdigit(), "course_year 是數字字串")
    check(paths.COURSE_SEMESTER.isdigit(), "course_semester 是數字字串")
    check(paths.RRF_K > 0 and paths.RRF_CANDIDATES > 0, "RRF 參數為正")


def _write_meta(tmpdir: str, **overrides) -> str:
    meta = {
        "schema_version": paths.SCHEMA_VERSION,
        "sqlite_embed_model": paths.SQLITE_EMBED_MODEL,
        "course_year": paths.COURSE_YEAR,
        "course_semester": paths.COURSE_SEMESTER,
    }
    meta.update(overrides)
    index_dir = os.path.join(tmpdir, "faiss_index")
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return path


def _check_sqlite_with(meta_path: str):
    """把 FAISS_META_JSON 暫時指到測試用的檔案,跑一次契約檢查。"""
    original = paths.FAISS_META_JSON
    paths.FAISS_META_JSON = meta_path
    try:
        paths._check_sqlite_contract()
        return None
    except paths.ContractMismatch as e:
        return str(e)
    finally:
        paths.FAISS_META_JSON = original


def test_matching_meta_passes() -> None:
    print("[meta 一致時放行]")
    with tempfile.TemporaryDirectory() as tmp:
        err = _check_sqlite_with(_write_meta(tmp))
        check(err is None, "完全一致 → 不 raise")


def test_each_mismatch_is_caught() -> None:
    print("[每個欄位不一致都要被抓到]")
    cases = {
        "schema_version": 999,
        "sqlite_embed_model": "some/other-model",
        "course_year": "999",
        "course_semester": "9",
    }
    for field, bad_value in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            err = _check_sqlite_with(_write_meta(tmp, **{field: bad_value}))
            check(err is not None, f"{field} 不符 → raise ContractMismatch")
            if err:
                check(field in err, f"{field} 的錯誤訊息點名該欄位")
                check(
                    "course-data-prep" in err,
                    f"{field} 的錯誤訊息指向 course-data-prep",
                )


def test_missing_artifact_is_actionable() -> None:
    print("[成品不存在時給可行動的訊息]")
    with tempfile.TemporaryDirectory() as tmp:
        err = _check_sqlite_with(os.path.join(tmp, "does_not_exist.json"))
        check(err is not None, "缺檔 → raise 而非 FileNotFoundError")
        check(err is not None and "course-data-prep" in err, "訊息指向 course-data-prep")


def test_deprecation_is_signalled() -> None:
    print("[SQLite 路徑發出 DeprecationWarning]")
    import warnings

    with tempfile.TemporaryDirectory() as tmp:
        meta = _write_meta(tmp)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _check_sqlite_with(meta)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        check(bool(dep), "走 SQLite 路徑會發 DeprecationWarning(PEP 702)")


if __name__ == "__main__":
    test_contract_loaded()
    test_matching_meta_passes()
    test_each_mismatch_is_caught()
    test_missing_artifact_is_actionable()
    test_deprecation_is_signalled()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed")
    raise SystemExit(1 if _FAIL else 0)
