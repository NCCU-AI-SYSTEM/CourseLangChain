"""查詢集與標註表的讀寫(TSV),以及讀入時的防呆。不碰資料庫。

用 TSV 而不是 CSV:repo 的 .gitignore 忽略所有 *.csv,而標註結果必須進版控。
寫檔帶 UTF-8 BOM,用 Excel 直接開中文才不會亂碼;讀檔用 utf-8-sig,有沒有 BOM 都能讀。
"""
from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERIES_MANUAL = HERE / "queries_manual.tsv"  # 人工撰寫的「主題」「口語」兩層,手動維護
QUERIES = HERE / "queries.tsv"  # build_gold sample 產生:規則層 + 上面那份
JUDGMENTS = HERE / "judgments.tsv"  # 相關性標註:規則層自動產生,人工層要人填
RESULTS_DIR = HERE / "results"

RULE_STRATA = ("teacher", "title")
MANUAL_STRATA = ("topic", "conversational")
STRATA = RULE_STRATA + MANUAL_STRATA
STRATUM_LABEL = {
    "teacher": "教師姓名",
    "title": "精確課名",
    "topic": "主題",
    "conversational": "口語長句",
}

QUERY_FIELDS = ("qid", "stratum", "query", "source")
JUDGMENT_FIELDS = ("qid", "course_id", "grade", "source", "name", "teacher", "snippet")

# 4 碼學年學期 + 9 碼科目代號。科目代號可能含大寫字母(例:1151044M01001),
# 115-1 有 131 門是這樣,只收純數字會把它們誤判成壞資料。
_COURSE_ID = re.compile(r"^\d{4}[0-9A-Z]{9}$")
_SCIENTIFIC = re.compile(r"[eE][+-]?\d|\.")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Query:
    qid: str
    stratum: str
    query: str
    source: str  # "rule" = 規則標註;"manual" = 人工標註


def clean(value: object) -> str:
    """TSV 的一格不能含 tab 或換行,連續空白也一併收成一格。"""
    return _WS.sub(" ", "" if value is None else str(value)).strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            {k: (v or "") for k, v in row.items() if k is not None}
            for row in csv.DictReader(f, delimiter="\t")
        ]


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for row in rows:
            writer.writerow([clean(row.get(k)) for k in fields])


def check_course_id(cid: str, where: str) -> str:
    """course_id 必須是 4 碼學年學期加 9 碼科目代號。

    最常見的壞法是用 Excel 開標註表再存檔:全數字的課號會被轉成 1.15E+12 這種科學記號,
    存回去就再也對不上課程。寧可在這裡大聲失敗,也不要讓壞掉的標註默默進到分數裡。
    """
    cid = cid.strip()
    if not _COURSE_ID.match(cid):
        hint = (
            "看起來被 Excel 轉成科學記號了。請改用 VS Code 編輯,或在 Excel 匯入時把 "
            "course_id 欄設成「文字」。"
            if _SCIENTIFIC.search(cid)
            else "格式應為 4 碼學年學期加 9 碼科目代號(數字或大寫字母)。"
        )
        raise ValueError(f"{where}:course_id {cid!r} 不是合法課號。{hint}")
    return cid


def parse_grade(raw: str, where: str) -> int | None:
    """grade 只接受 0、1、2;留空代表尚未標註,回 None。"""
    raw = raw.strip()
    if raw == "":
        return None
    if raw not in {"0", "1", "2"}:
        raise ValueError(f"{where}:grade 只能是 0、1、2 或留空,收到 {raw!r}")
    return int(raw)


def load_queries(path: Path = QUERIES) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    for line, row in enumerate(read_tsv(path), 2):
        q = Query(
            qid=row.get("qid", "").strip(),
            stratum=row.get("stratum", "").strip(),
            query=row.get("query", "").strip(),
            source=row.get("source", "").strip(),
        )
        where = f"{path.name} 第 {line} 行"
        if q.stratum not in STRATA:
            raise ValueError(f"{where}:未知的題型 {q.stratum!r},可用:{', '.join(STRATA)}")
        if not q.qid or not q.query:
            raise ValueError(f"{where}:qid 與 query 不能空白")
        if q.qid in seen:
            raise ValueError(f"{where}:qid {q.qid} 重複")
        seen.add(q.qid)
        queries.append(q)
    return queries


def load_judgments(path: Path = JUDGMENTS) -> dict[str, dict[str, int | None]]:
    """qid -> {course_id: grade};grade 為 None 代表尚未標註。"""
    out: dict[str, dict[str, int | None]] = {}
    for line, row in enumerate(read_tsv(path), 2):
        where = f"{path.name} 第 {line} 行"
        cid = check_course_id(row.get("course_id", ""), where)
        out.setdefault(row.get("qid", "").strip(), {})[cid] = parse_grade(
            row.get("grade", ""), where
        )
    return out
