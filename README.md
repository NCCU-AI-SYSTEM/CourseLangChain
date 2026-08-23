# CourseLangGraph

[![](https://dcbadge.vercel.app/api/server/n8w5qE4xyA)](https://discord.gg/n8w5qE4xyA)

政大課程查詢 / 排課 agent。這個 repo 只負責**查詢**;課程資料的準備(schema、時間欄位、
embedding)在另一個 repo [`course-data-prep`](#資料從哪裡來),產出一份 PostgreSQL 成品(`.sql.gz`)給這邊用。

所以一般開發者**不需要 GPU、不需要下載 embedding 模型、也不用碰 course-data-prep** ——
拿到成品檔就能跑。

## 需求

- Docker(跑 PostgreSQL)
- [`uv`](https://github.com/astral-sh/uv) 與 `python >= 3.13`(用到 PEP 702 的 `warnings.deprecated`)
- [`ollama`](https://github.com/ollama/ollama) 或 Google AI API key

## 快速開始

已經有課程資料成品(`.sql.gz`)的話:

```sh
cp .env.example .env          # 至少要填 MODEL(用 ollama list 看有哪些)
cp <某處拿到的>data.sql.gz db/init/

docker compose up -d postgres # 首次啟動會自動還原 db/init/ 裡的成品
uv sync
uv run python app.py          # FastAPI,http://localhost:8000/api/ask?question=...
```

CLI 模式:`uv run python main.py`;用 Docker 跑 app:`docker compose --profile app up -d`。

> [!IMPORTANT]
> **完整流程看 [docs/SETUP.md](docs/SETUP.md)** —— 資料準備的每一個步驟、各階段怎麼驗證、
> 以及常見問題(換資料沒生效、容器連不到 Ollama、契約不符開不起來…)。
> 沒有 `.sql.gz` 檔、或要重新產生一個學期的資料,請從那份開始看。

## 檢索怎麼做的

PostgreSQL 上的混合檢索,BM25 與向量各跑一次,再用 RRF 融合:

```
score(course) = 1/(60 + BM25名次) + 1/(60 + 向量名次)
```

兩路各自擅長的東西,是量出來的,不是猜的(115-1 實測):

- **BM25**(ParadeDB `pg_search`,分詞器 `pdb.jieba`):字面比對,強在**罕見的專有名詞** ——
  老師名、教科書作者(「Varian」)這種向量放不進語意空間的字。反過來,中文短複合詞是它的弱點:
  jieba 把「資料庫」切成 資料/庫,查詢又是 OR,於是所有含「資料」的課都進來,「資料庫應用」
  反而擠不進前八名。
- **向量**(`pgvector`,逐欄分塊、無索引):語意相近,正好補上那個弱點 —— 同一個「資料庫」
  查詢,「資料庫應用」的 textbook / schedule / evaluation / objective 四塊全排在最前面。
  但它對專有名詞無能為力:單用向量查老師名,回的是一堆不相干的語言課。
- **RRF** 只看名次不看原始分數 —— BM25 分數與 cosine 距離量綱不同,直接相加沒有意義。
  `k=60` 讓「兩路都排前面」贏過「單路第一名」。

> [!IMPORTANT]
> 查詢用的是 `|||`(OR)而**不是**預設的 `@@@`。ParadeDB 的 `@@@` 對多詞查詢預設是
> **AND**,查「我想學怎麼寫程式」會回 **0 筆** —— 對話式輸入正是這個系統的主要用法,
> 用 AND 等於檢索直接失效。

資料庫用 `paradedb/paradedb` 官方 image,`pg_search` 與 `pgvector` 都已內建預編,
不需要自己編任何擴充。

## 資料從哪裡來

`course-data-prep` 負責準備資料,而且**自己帶一個 PostgreSQL**(port 5433,與這裡的 5432
並存)。原始課程資料、中間狀態都留在那邊,**這個 repo 只收準備好的成品**:

```sh
# 在 course-data-prep 底下
docker compose up -d postgres         # 起自己的資料庫
LOAD_SQLITE=<爬蟲>.db docker compose --profile load run --rm pgloader
uv run python -m prep.migrate         # 套 DDL、寫 schema_meta
uv run python -m prep.populate_time   # time_raw → 結構化時間欄位
uv run python -m prep.embed           # 逐欄分塊 → 向量(這步吃 GPU)
uv run python -m prep.check           # 出貨前檢查,dump 指令見那邊的 README
```

embedding 刻意放在 host 上跑:Docker Desktop 在 macOS 沒有 GPU passthrough,Linux 要
`nvidia-container-toolkit`,在容器裡跑等於強制用 CPU。逐步說明見 [docs/SETUP.md](docs/SETUP.md)。

### contract.yaml

`contract.yaml` (embedding model, dimension, semester, schema version) is a
**copy that ships with the artifact** and describes what is inside that
`.sql.gz`. course-data-prep writes the same values into the database's
`schema_meta` table; this side compares them at startup and refuses to serve on
a mismatch.

Do not edit it here. Changing the model or semester happens in course-data-prep
— edit, rerun the pipeline, build a new artifact — and then `.sql.gz` and
`contract.yaml` are replaced together.

擋的是這個失敗模式:那邊換了 embedding 模型重新灌資料,這邊沒跟上 —— query 向量與庫裡的
向量來自不同模型,**檢索不會報錯,只會安靜地回一堆語意無關的課**。

## 已棄用:`USE_SQLITE=true`

SQLite + FAISS 是 PostgreSQL 之前的舊架構,現在兩邊做的是同一件事。它仍可運作但
**不再是支援對象**,呼叫到會發 `DeprecationWarning`,預計後續版本移除。

新的部署一律用 `USE_SQLITE=false`(預設值)。理由:兩套檢索堆疊會各自漂移 —— 而且已經
漂過一次,PostgreSQL 那邊一度悄悄少了向量融合、退化成純 BM25 都沒人發現。

> `DeprecationWarning` 預設被 Python 藏起來,要看到請跑
> `uv run python -W default::DeprecationWarning app.py`。

## 測試

不依賴 pytest,每支都可以單獨跑。**要用 `-m`**(直接跑檔案路徑會 `ModuleNotFoundError`,
因為專案根目錄不在 `sys.path` 上):

```sh
uv run python -m tests.test_contract
uv run python -m tests.test_scheduler
uv run python -m tests.test_schedule_tool
uv run python -m tests.test_regression_fixes
uv run python -m tests.test_harness_integration
uv run python -m tests.test_user_profile
```

## 前端

**[CourseLangGraph-frontend](https://github.com/NCCUCourseScheduling/CourseLangGraph-frontend)**

## Final Report

[report](https://docs.google.com/document/d/1CkelC_x8B_QnVHEiIZisG1d8BJwXoYg02lqaqgbQlFY/edit?usp=sharing)
