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
cp <某處拿到的>data.sql.gz db/init/
```

> [!IMPORTANT]
> **`db/init/` 只放一個資料檔。** 這個目錄會直接掛成容器的 `/docker-entrypoint-initdb.d`,
> entrypoint 會把裡面所有 `.sql` / `.sql.gz` 依**檔名順序全部執行**。放兩份的話兩份都會跑,
> 第二份會因為表已存在而失敗(`ON_ERROR_STOP=1` → 容器起不來)。
>
> 這個 repo **只收準備好的成品**。原始課程 SQL 屬於資料準備階段,請放
> [`course-data-prep`](../../course-data-prep/README.md),不要放這裡。

## 2-3. 起資料庫

```sh
docker compose up -d postgres
docker compose logs postgres | grep initdb
# /usr/local/bin/docker-entrypoint.sh: running /docker-entrypoint-initdb.d/data.sql.gz
```

還原是 postgres 官方 entrypoint 原生做的(`.sql.gz` 會自動 gunzip 後餵給
`psql -v ON_ERROR_STOP=1`),這邊沒有任何自訂腳本。

> [!IMPORTANT]
> restore **只在資料庫第一次建立時**發生(容器 initdb 機制)。換 dump 要先
> `docker compose --profile app down -v` 清掉 volume,否則新檔案不會生效,
> 而且不會有任何錯誤訊息。(`--profile app` 的理由見 2-4(B)。)

檢查資料進去了:

```sh
docker compose exec postgres psql -U postgres -d academic -c \
  "SELECT (SELECT COUNT(DISTINCT id) FROM course) courses,
          (SELECT COUNT(*) FROM course_chunk) chunks;"
#  courses | chunks
#     2925 |  28634     ← chunks 為 0 的話檢索會退化成純 BM25
```

向量存在 `course_chunk`(一門課約 10 塊),不是 `course` 的欄位。

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

> [!NOTE]
> **第一次查詢會慢好幾分鐘。** query encoder(`BAAI/bge-base-zh-v1.5`,約 400MB)
> 沒有打進 image,是第一次查詢時才下載的。下載期間那個請求就一直掛著,看起來像當掉。
> 快取放在 `hfcache` volume,所以只有第一次要等 —— 重建容器、`--build` 都不會重抓,
> 只有 `down -v` 會。想先暖起來:
> ```sh
> docker compose exec app python -c \
>   "from tools.retrieve import _get_encoder; _get_encoder()"
> ```
> host 上已經有這個模型的話(跑過 course-data-prep 就會有),直接複製更快。
> **`mkdir` 那行不能省** —— `docker cp` 在目標目錄不存在時會把來源*當成*該目錄,
> 結果是模型內容被攤平進 `hub/`,快取看起來有 781MB 卻完全找不到模型:
> ```sh
> docker compose exec -u root app mkdir -p /home/user/.cache/huggingface/hub
> docker cp ~/.cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5 \
>   courselangchain-app-1:/home/user/.cache/huggingface/hub/
> docker compose exec -u root app chown -R user:user /home/user/.cache/huggingface
> # 確認:應該印出 models--BAAI--bge-base-zh-v1.5
> docker compose exec app ls /home/user/.cache/huggingface/hub
> ```

> [!IMPORTANT]
> **`down` 要帶 `--profile app`。** app 服務在 profile 底下,`docker compose down -v`
> 不帶 profile **不會移除 app 容器** —— 它會活下來繼續跑舊 image 的舊程式碼,
> 而資料庫已經換過了。症狀是查詢報 `column does not exist` 之類、但你確定改過那段碼。
> 換資料或改程式後重來:`docker compose --profile app down -v`。

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
curl -s --get --data-urlencode "question=我想學怎麼寫程式，推薦一門課" \
     --data "stream=false" http://localhost:8000/api/ask
```

應該會看到「程式設計概論」。這句話刻意挑過:斷詞後是 我想學/怎麼/寫/程式,一個字都對不上
課名,BM25 單獨跑是找不到的 —— 回得出來就代表向量那路真的有在作用。

(別拿課名去驗。「資料庫」這種中文短複合詞會被 jieba 切開,BM25 反而不準,見
[README](../README.md#檢索怎麼做的);而且課名是逐學期變的,拿某一門課當驗證基準,
換個學期就對不上了。)

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

> embedding / 產 dump 那邊的問題(模型下載失敗、維度不符)看
> [`course-data-prep/README.md`](../../course-data-prep/README.md) 的疑難排解。

### 開機就被擋:「資料成品與 contract.yaml 不一致」

兩個 repo 的 `contract.yaml` 設定值沒對齊,或 dump 是舊的。訊息會直接指出哪個欄位對不上:

```
- schema_version: 資料成品是 '8',contract.yaml 是 7
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

如果 `down -v` 跑了還是沒生效,檢查有沒有帶 `--profile app` —— 不帶的話 app 容器不會被
移除,它會拿舊 image 的舊程式碼去查新資料庫。

### 查詢回「處理時間過長」

harness 預設 600 秒,這個值已經照本機模型抓過:9B 的 thinking 模型跑「檢索 20 門再排課」
約 165 秒,帶時間條件的問句要多跑一次 `text_to_sql_tool`(多一輪完整 LLM 來回),
實測 5 步 397 秒。還是不夠就改 `.env`:

```sh
AGENT_TIMEOUT_SEC=900
```

host 與容器都吃這一份 —— `paths.py` 用 `load_dotenv` 讀它,compose 也拿它做變數展開。
改完 host 直接重跑,容器 `docker compose --profile app up -d app`。

雲端模型(Google AI)快得多,可以往下調。

Ollama 一次只跑一個請求。前一個查詢被 client 端砍掉時,伺服器那邊還會繼續算完,
後面的請求就排在後面等 —— 症狀是連 `curl /api/chat` 都沒反應,但 `/api/tags` 秒回。
不是當機,等它算完就好;要確認的話看 `ollama runner` 的 CPU 是不是還在動。

### Docker 跑 app,查詢都回「系統暫時無法處理」

多半是容器連不到 Ollama。見 2-4(B) —— 要 `OLLAMA_HOST=0.0.0.0 ollama serve`。

### 檢索結果很不準

先確認向量有灌進去(見 2-3 的檢查)。`course_chunk` 是空的話會退化成純 BM25,
對話式問句(「我想學怎麼寫程式」)會明顯變差。
