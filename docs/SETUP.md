# 安裝與執行指南

從零到能查詢。整個系統分兩個階段,**分屬兩個 repo**:

| 階段 | 誰要做 | 在哪裡 | 需要 GPU? |
|---|---|---|---|
| **一、資料準備** | 每學期一次,資料維護者 | [`course-data-prep`](../../course-data-prep/README.md) | 建議 |
| **二、執行** | 每個開發者 | 本 repo(以下) | 不用 |

**大部分人只需要階段二。** 拿到別人產好的 `.sql.gz` 就能跑 —— 不用 GPU、不用下載 embedding
模型、也不用碰 `course-data-prep`。

> 沒有 `.sql.gz`、或要重新產生一個學期的資料?
> **完整的資料準備步驟在 [`course-data-prep/README.md`](../../course-data-prep/README.md)**,
> 那裡有逐步說明與每一步的預期輸出。做完之後回到這裡的 2-2。

---

## 事前準備

| 需求 | 說明 |
|---|---|
| Docker | 資料庫一定要用(ParadeDB:PostgreSQL + `pg_search` + `pgvector`) |
| [`uv`](https://github.com/astral-sh/uv) | 在 host 上跑 app 用;純 Docker 跑法可略過 |
| Python ≥ 3.13 | 用到 PEP 702 的 `warnings.deprecated`;`uv` 會自己裝 |
| [Ollama](https://github.com/ollama/ollama) | 本機 LLM。或改用 Google AI(見 `.env`) |
| Langfuse(可選) | 追蹤用,預設 `http://localhost:3000` |

先確認 Ollama 有模型可用:

```sh
ollama list
```

---

# 階段二:執行

## 2-1. 設定

```sh
cp .env.example .env
```

至少要改 `MODEL`(用 `ollama list` 看有哪些)。`USE_SQLITE` 保持 `false`。
設定都寫在 `.env` 裡。

(embedding 模型、維度、學期這些「資料的屬性」不在 `.env`,在 `contract.yaml` ——
要改請看 [course-data-prep](../../course-data-prep/README.md)。)

## 2-2. 放資料

把成品 `.sql.gz` 放進 `db/init/`:

```sh
cp <某處拿到的>course-1142.sql.gz db/init/
```

> [!IMPORTANT]
> **`db/init/` 只放一個資料檔。** 這個目錄會直接掛成容器的 `/docker-entrypoint-initdb.d`,
> entrypoint 會把裡面所有 `.sql` / `.sql.gz` 依**檔名順序全部執行**。放兩份的話兩份都會跑,
> 第二份會因為表已存在而失敗(`ON_ERROR_STOP=1` → 容器起不來)。
>
> 這個 repo **只收準備好的成品**。原始課程 SQL 屬於資料準備階段,請放
> [`course-data-prep/raw/`](../../course-data-prep/README.md),不要放這裡。

## 2-3. 起資料庫

```sh
docker compose up -d postgres
docker compose logs postgres | grep initdb
# /usr/local/bin/docker-entrypoint.sh: running /docker-entrypoint-initdb.d/course-1142.sql.gz
```

還原是 postgres 官方 entrypoint 原生做的(`.sql.gz` 會自動 gunzip 後餵給
`psql -v ON_ERROR_STOP=1`),這邊沒有任何自訂腳本。

> [!IMPORTANT]
> restore **只在資料庫第一次建立時**發生(容器 initdb 機制)。換 dump 要先
> `docker compose down -v` 清掉 volume,否則新檔案不會生效,而且不會有任何錯誤訊息。

檢查資料進去了:

```sh
docker compose exec postgres psql -U postgres -d academic -c \
  "SELECT COUNT(*) courses, COUNT(embedding) vecs FROM course WHERE y='114' AND s='2';"
#  courses | vecs
#     3472 | 3472     ← 兩個數字要一樣,不然檢索會退化成純 BM25
```

## 2-4. 跑 app —— 兩種方式擇一

### (A) 用 uv 在 host 上跑(預設)

```sh
uv sync
uv run python app.py              # FastAPI,http://localhost:8000
```

CLI 版:`uv run python main.py`

### (B) 用 Docker 跑(不想在本機裝 Python)

```sh
docker compose --profile app up -d
```

> [!IMPORTANT]
> Ollama 預設只聽 `127.0.0.1`,容器連不到,查詢會回「系統暫時無法處理您的要求」。
> 要讓它聽所有介面:
> ```sh
> OLLAMA_HOST=0.0.0.0 ollama serve
> ```
> 容器裡沒有 GPU,所以 LLM 一律留在 host 跑,compose 只把位址指過去。
> 容器用的是 `.env` 裡的 `DOCKER_OLLAMA_HOST` / `DOCKER_LANGFUSE_BASE_URL`
> (預設 `host.docker.internal`),**不是** `OLLAMA_HOST` / `LANGFUSE_BASE_URL` ——
> 那兩個是 host 用的 `localhost`,帶進容器會變成連容器自己。

## 2-5. 確認能動

```sh
curl -s --get --data-urlencode "question=給我關於資料庫的課" \
     --data "stream=false" http://localhost:8000/api/ask
```

應該會看到 Markdown 表格,第一列是「資料庫系統」。

跑測試(不依賴 pytest;**要用 `-m`**,直接跑檔案路徑會 `ModuleNotFoundError`):

```sh
uv run python -m tests.test_contract
uv run python -m tests.test_scheduler
uv run python -m tests.test_regression_fixes
uv run python -m tests.test_user_profile
uv run python -m tests.test_schedule_tool
uv run python -m tests.test_harness_integration
```

## 2-6.(可選)開啟 Langfuse 追蹤

`.env` 的 key 留空 = 追蹤直接關掉,而且**只在 log 印一行提示、不會報錯**,很容易
以為有在收其實沒有。要用就把 Langfuse 跑起來,填入專案的 API key:

```sh
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="http://localhost:3000"
```

確認真的有收到:

```sh
curl -s -u "pk-lf-...:sk-lf-..." \
  "http://localhost:3000/api/public/traces?limit=5" | jq '.meta.totalItems'
```

一筆 trace 裡看得到整個 ReAct 迴圈:`[AGENT] agent` → `[GENERATION] ChatOllama`
→ `[TOOL] retrieve_tool` → `[CHAIN] should_continue`。

---

# 疑難排解

> embedding / 產 dump 那邊的問題(模型下載失敗、維度不符、HNSW 記憶體不足)看
> [`course-data-prep/README.md`](../../course-data-prep/README.md) 的疑難排解。

### 開機就被擋:「資料成品與 contract.yaml 不一致」

兩個 repo 的 `contract.yaml` 設定值沒對齊,或 dump 是舊的。訊息會直接指出哪個欄位對不上:

```
- schema_version: 資料成品是 '6',contract.yaml 是 3
```

這是刻意的 —— 模型/維度不一致時檢索**不會報錯**,只會安靜地回一堆語意無關的課,
那是最難 debug 的失敗模式,所以寧可不啟動。

`contract.yaml` is a **copy that ships with the artifact — do not edit it
here.** It describes what is inside that `.sql.gz`. Changing the model or
semester happens in course-data-prep: edit, rerun the pipeline, build a new
artifact, then replace both files together. Editing it here only makes it
disagree with the data.

`contract.yaml` is mounted read-only, so replacing it needs no rebuild —
`docker compose restart app` is enough.

### 「資料庫沒有 schema_meta 表」

這個 DB 還沒被準備過。把 dump 放進 `db/init/` 後
`docker compose down -v && docker compose up -d postgres`。

### 換了 dump 卻沒生效

沒有 `down -v`。restore 只在 volume 全新時跑。

### 查詢回「處理時間過長」

harness 預設 120 秒。本機模型差很多 —— 9B 的 thinking 模型在 Apple Silicon 上跑
「檢索 20 門再排課」約需 165 秒。調大:

```sh
AGENT_TIMEOUT_SEC=600 uv run python app.py
```

### Docker 跑 app,查詢都回「系統暫時無法處理」

多半是容器連不到 Ollama。見 2-4(B) —— 要 `OLLAMA_HOST=0.0.0.0 ollama serve`。

### 檢索結果很不準

先確認向量有灌進去(見 2-3 的檢查)。`vecs` 少於 `courses` 的話會退化成純 BM25,
對話式問句(「我想學怎麼寫程式」)會明顯變差。
