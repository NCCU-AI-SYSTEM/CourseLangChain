# 課程檢索

> 最後確認：2026-11-15。對照程式：`feat/course-card-api`（PR #7）加上 `fix/stream-hide-tool-output`；
> 對照資料：115-1 學期、`schema_version` 8、`BAAI/bge-base-zh-v1.5`。
> 文中的實測數字都是 2026-11-15 在後端容器裡對執行中的資料庫量的，除非另有註明。
> 文件與程式不一致時，以程式為準，並請更新本文件。

前半段是摘要，給想知道「做了什麼、效果如何」的讀者；後半段是開發者細節，寫到檔案與函式。

---

## 摘要

學生用一句中文描述想上的課（「星期三下午有什麼 AI 相關的課」），系統要從 115-1 學期的
**2,926 門課**裡找出最相關的 10 門（排課時 20 門），交給對話模型整理成回答或排成課表。

做法是**混合檢索**：同一句查詢同時做兩種比對，再依名次融合。

| | 字面比對（BM25） | 語意比對（向量） |
|---|---|---|
| 擅長 | 老師名字、課名這類專有名詞 | 換句話說、口語描述 |
| 弱點 | 中文複合詞會被切開 | 放不進語意空間的專有名詞 |
| 實作 | ParadeDB `pg_search`，jieba 分詞 | `pgvector`，`bge-base-zh-v1.5`，每門課切成多塊 |

兩路用 **RRF（Reciprocal Rank Fusion）** 融合：只看名次不看分數，兩路都排前面的課勝出。
時段、授課語言、學分這類**必須正確的條件**，不交給檢索去「猜」：
由語言模型把句子轉成固定格式的 JSON，再由程式轉成 SQL，在檢索之前就先篩掉不符合的課。

### 成果

離線評估 60 題（2026-10-18，[完整報告](../evaluation/results/report.md)）：

| 方法 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| 只用 BM25 | 0.894 | 0.687 | 0.761 |
| 只用向量 | 0.753 | 0.755 | 0.721 |
| **混合（RRF k=60，正式設定）** | **0.954** | **0.855** | **0.853** |

混合比單用 BM25 的 nDCG@10 高 0.092（95% 信賴區間 +0.025 ~ +0.163），比單用向量高 0.132（+0.067 ~ +0.198），
兩者的區間都不含 0。分題型看更能說明「為什麼要兩路」：

| 題型（各 30 題） | BM25 Recall@10 | 向量 Recall@10 | 混合 Recall@10 |
|---|---|---|---|
| 查老師名字 | 0.956 | **0.506** | 0.942 |
| 查精確課名 | **0.833** | 1.000 | 0.967 |

向量查老師名字只找回一半，BM25 查課名漏掉六分之一；融合之後兩種題型都在 0.94 以上。

速度：一次檢索 0.46 ~ 0.60 秒（容器內、CPU、向量模型已載入）；後端啟動後第一次查詢要多等約 4 秒載入模型。

### 限制

- **評估只涵蓋字面型的查詢。** 「主題」與「口語長句」兩種題型共 40 題、782 筆候選還沒有人工標註，
  目前的數字偏向 BM25 擅長的題目，不能當成整體結論。
- **沒有相關度門檻。** 條件很嚴時（例如星期三下午，340 門課裡課名與 AI 相關的只有 2 門），
  系統仍會回滿 10 門，後面幾門只是弱相關。
- **複合詞會被切開。** 「資料庫」被切成「資料」與「庫」，兩門「資料庫應用」只排到第 10 與第 11 名。

---

## 一次查詢怎麼走

```mermaid
flowchart TD
    U["使用者：星期三下午有什麼 AI 相關的課？"] --> A["對話模型（ReAct agent）"]
    A -->|"有時間或其他條件"| T["text_to_sql_tool<br/>模型把條件寫成 JSON"]
    T -->|"constraints_to_where<br/>程式把 JSON 轉成 SQL 條件"| R
    A -->|"沒有條件"| R["retrieve_tool(keyword, top_k, sql_filter)"]
    R --> DB[("PostgreSQL<br/>BM25 一路 + 向量一路<br/>→ RRF 融合")]
    DB --> R
    R -->|"編號清單（含 13 碼課號）"| A
    A -->|"Markdown 表格"| UI["前端畫面"]
    R -.->|"側通道：候選課程（課號不經模型轉述）"| UI
    UI -.->|"點課程卡片：GET /api/courses/{id}"| API["課程詳細資料"]
```

1. 對話模型依 [系統提示](../agents/brain_agent.py) 判斷意圖。查課（意圖 B）預設 `top_k=10`，排課（意圖 D）建議 `top_k=20`。
2. 句子裡有時段、語言、學分等條件時，先呼叫 `text_to_sql_tool`，把回傳的 SQL 條件當作 `sql_filter` 傳給 `retrieve_tool`。
3. `retrieve_tool` 在一條 SQL 裡跑完 BM25、向量與 RRF，回傳編號清單。
4. 清單是給模型讀的文字。同一份文字由 `main.py` 攔下、解析成結構化資料，經 SSE 側通道送到前端畫成課程卡片。
   **課號因此不經模型轉述**，避免模型抄錯 13 碼。

---

## 資料：檢索的對象

資料由 [CourseDataPrep](https://github.com/NCCU-AI-SYSTEM/CourseDataPrep) 每學期產生一次，
以 `data.sql.gz` 交付，後端啟動時比對 `contract.yaml` 與資料庫裡的 `schema_meta`，不一致就拒絕服務。

| 表 / 欄位 | 內容 | 115-1 規模 |
|---|---|---|
| `public.course` | 一門課屬於幾個類別就有幾列，`id` **不唯一** | 3,651 列、2,926 門 |
| `course.content_text` | BM25 的對象：由課名、英文課名、老師、目標、簡介、進度、評量、書目組成的生成欄位 | — |
| `course.pk` | BM25 索引用的代理鍵（ParadeDB 的 key 必須唯一，`id` 不是） | — |
| `course.sessions` | 上課時段，jsonb 陣列，例如 `[{"weekday": 5, "start_hour": 9, "end_hour": 12}]` | — |
| `course.has_morning` 等 | 是否有早上／中午／下午／晚上的課，給「只講時段、沒講星期」的條件用 | — |
| `public.course_chunk` | 向量那一路的對象，一門課切成多塊，每塊一個 768 維向量 | 28,851 塊，2,926 門全部有向量 |
| `public.schema_meta` | 資料的版本、embedding 模型、學期 | `schema_version` 8 |

各欄位切出的塊數：

| 欄位 | 塊數 |
|---|---|
| 每週進度 `schedule` | 10,969 |
| 指定書目 `textbook` | 5,303 |
| 評量方式 `evaluation` | 4,558 |
| 課程簡介 `syllabus` | 4,321 |
| 課程目標 `objective` | 3,656 |
| 只有課名（沒有任何大綱的課）`title` | 44 |

---

## BM25 這一路

程式：[`tools/retrieve.py`](../tools/retrieve.py) 的 `_RRF_SQL`，`bm25_raw` 與 `bm25` 兩段。

- **索引**建在 `content_text` 上，分詞器是 `pdb.jieba`（`CourseDataPrep/init-scripts/03-schema.sql`）。
- **查詢運算子用 `|||`（OR），不是預設的 `@@@`。** `@@@` 對多詞查詢是 AND，「我想學怎麼寫程式」會回 0 筆。
- **只放有區辨力的欄位。** BM25 會依文件長度正規化，每門課都長得一樣的欄位不只沒用，還會壓低真正有用的欄位：

  | 刻意排除的欄位 | 原因（115-1 實測，見 `03-schema.sql` 註解） |
  |---|---|
  | `teaching_approach` | 3,568 門課措辭完全相同，只差百分比 |
  | `ai_policy` | 1,058 門課共用「This Course Does Not Involve the Use of AI」，含「AI」，會回答相反的查詢 |
  | `detail_*` | 上千字的評分樣板 |

  這些欄位仍可由 `course_detail_tool` 查到，只是不用來找課。
  老師名字反而**要放**：名字很短，不影響長度正規化，而且是向量無法取代的欄位。
- **一門課取最高分。** `course` 表一門課有多列，先撈 `50 × 8 = 400` 列（`_DUP_FACTOR = 8`），依課號取 `MAX(score)` 後排名，取前 50 門。

jieba 對查詢的切法決定了 BM25 會比對什麼（2026-11-15 用資料庫實際的分詞器查）：

| 查詢 | 切詞結果 | 影響 |
|---|---|---|
| 人工智慧 | 人工、智慧、人工智慧 | 搜尋模式會多切出子詞，「智慧財產」的課也會命中「智慧」 |
| 資料庫 | 資料、庫 | 「資料」很常見，含「資料」的課全部湧進來 |
| 想學統計但數學不太好 | 想學、統計、但、數學、不太好 | 口語詞也參與比對，靠向量那一路補語意 |

## 向量這一路

程式：`_RRF_SQL` 的 `vec_raw` 與 `vec` 兩段；查詢向量由 `_encode_query` 產生。

- **模型**：`BAAI/bge-base-zh-v1.5`，768 維。查詢必須用和建資料時**同一個模型**編碼，
  否則向量不可比，檢索不會報錯，只會安靜地回傳無關的課。所以模型名寫在 `contract.yaml`，由 `check_contract()` 在啟動時把關。
- **逐欄切塊**（`CourseDataPrep/src/prep/chunk.py`）。模型只讀 512 個 token，`schedule` 一欄的中位數就有 464 個，
  整門課塞進一個向量時 82% 的課會被截斷。所以五個欄位各自切塊：每塊 320 token、重疊 40，
  優先在 `|`（爬蟲用來分隔大綱段落）與句號處切；每塊開頭都重複「課程名稱、授課老師、欄位名」，
  單獨被找到時仍看得出是哪門課的哪一段。
- **一門課取最接近的一塊**（`MIN(dist)`）。問教科書就由教科書那一塊命中，不會被整門課的平均稀釋。
  因為一門課約 10 塊，向量這一路先撈 `50 × 16 = 800` 塊（`_CHUNK_FACTOR = 16`），合併成課之後取前 50 門。
- **刻意不建 HNSW 索引。** 這個 ParadeDB 版本不開放 `hnsw.ef_search`，HNSW 掃描大約只回 40 列、約 12 門課，
  而 BM25 給 400 列，融合的兩邊嚴重不對等。循序掃描 28,634 個向量只要 43 ~ 108 毫秒，結果精確、沒有上限
  （CourseDataPrep 的量測）。資料量大一個數量級時再重新評估。
- **沒有向量時退回純 BM25**（`_BM25_ONLY_SQL`），並在 log 記一行警告。

## RRF 融合

$$\text{score}(c) = \frac{1}{k + r_{\text{BM25}}(c)} + \frac{1}{k + r_{\text{vec}}(c)}$$

- **只用名次，不用原始分數。** BM25 分數與 cosine 距離量綱不同，直接相加沒有意義。
- **`k = 60`**（`RRF_K`）。k 越大，名次之間的差距越小，「兩路都排前面」越容易勝過「只在一路排第一」。
- **只出現在一路的課不會漏掉。** `FULL OUTER JOIN` 加 `COALESCE(…, 0)`，另一路的那一項以 0 計。
- 每路各取前 50 門（`RRF_CANDIDATES`），融合後依 `rrf DESC` 取 `top_k`。

## 硬條件：時段、語言、學分

程式：[`tools/text_to_sql.py`](../tools/text_to_sql.py) 的 `_pg_json_path`、[`tools/constraints.py`](../tools/constraints.py)。

語言模型**只負責把句子寫成 JSON**，SQL 由程式產生：

```json
{"include_times": [{"weekday": 3, "start_hour": 13, "end_hour": 18}], "lang": "英文", "point_min": 3}
```

| JSON 欄位 | 轉成的 SQL 條件 |
|---|---|
| `include_times`，沒有 `weekday` | `has_afternoon = true` 這類旗標（快） |
| `include_times`，有 `weekday` | `EXISTS (SELECT 1 FROM jsonb_array_elements(sessions) … )`，時段有重疊就算 |
| `exclude_times` | 同上，加 `NOT` |
| `lang` | `lang = '英文'` |
| `kind` | `kind = 2`（1 必修、2 選修、3 通識、4 體育；其他值一律丟掉） |
| `point_min` / `point_max` | `point >= 3` / `point <= …` |
| `unit` | `unit ILIKE '%…%'` |

設計上的幾個決定：

- **`temperature=0`、關掉 thinking、輸出上限 512**（`get_chat_llm(extract=True)`）。這是抽取任務：
  溫度 0.8 時同一句「晚上的課」一次回正確、一次回空陣列。
- **時段邊界寫死在提示詞裡**（早上 8–12、中午 12–13、下午 13–18、晚上 18–22），不讓模型自己推算。
- **`kind=0` 會被丟掉。** 模型常把沒問到的欄位填 0，而資料裡真的有 46 門 `kind=0` 的課，
  不擋的話「晚上的課」會從 294 門悄悄變成 6 門。
- **條件在兩路的 `LIMIT` 之前就套用。** 候選從符合條件的課裡挑，不是先挑 50 門再過濾（那樣常常會被濾到剩 0 門）。
- **解析失敗**時回 `ERROR: …`，`retrieve_tool` 會要模型重跑一次 `text_to_sql_tool`（系統提示允許重試一次）。

排課的「避開星期五」**不走這條路**：它是 `schedule_tool` 的 `avoid_weekdays` 參數，由排課程式直接排除。

## 輸出與前端

`retrieve_tool` 回傳給模型讀的文字，每門課一行，最後附上時間代碼對照：

```
1. 人工智慧方法與工具｜course_id: 1151070394001｜時間: 四D56｜老師: 陳昭伶｜學分: 3.0
```

- `main.py` 的 `astream` 在 `retrieve_tool` 結束時，用 `parse_formatted_docs` 把這段文字解析回結構化資料，
  以 `{"type": "courses", …}` 經 SSE 送給前端。同一輪查兩次時以課號去重。
- 前端點課程卡片時呼叫 `GET /api/courses/{course_id}`（[`tools/course_info.py`](../tools/course_info.py)）：
  格式不對回 400、找不到回 404、SQLite 模式回 501、資料庫連不上回 503。
- 查「單一門課」的詳細內容走 `course_detail_tool`：有課號就用課號查；只有課名時用 `name LIKE` 並鎖定當學期，
  找到多筆就把候選交給使用者選。

## 參數

| 設定 | 預設 | 位置 | 作用 |
|---|---|---|---|
| `RRF_K` | 60 | 環境變數，`paths.py` | RRF 的 k |
| `RRF_CANDIDATES` | 50 | 環境變數，`paths.py` | 每路取幾門候選 |
| `EMBED_DEVICE` | `cpu` | 環境變數 | 查詢向量用的裝置；只算一句話，CPU 就夠 |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/academic` | 環境變數 | 容器內由 `docker-compose.yml` 指定 |
| `USE_SQLITE` | `false` | 環境變數 | `true` 走舊的 SQLite + FAISS 路徑（已棄用，不要再加功能） |
| `_DUP_FACTOR` | 8 | `tools/retrieve.py` | BM25 一路撈幾倍的列數（一門課在 `course` 有多列） |
| `_CHUNK_FACTOR` | 16 | `tools/retrieve.py` | 向量一路撈幾倍的塊數（一門課約 10 塊） |
| `top_k` | 10（排課 20） | 工具參數，由模型決定 | 最後回幾門 |

**改參數前先跑離線評估**，用數字比較，不要憑感覺。

---

## 離線評估

程式與說明：[`evaluation/`](../evaluation/README.md)。評估直接重用 `tools/retrieve.py` 的 SQL 與參數，量到的就是正式系統在跑的那段檢索。

| 題型 | 題數 | 怎麼標註 | 狀態 |
|---|---|---|---|
| 查老師名字 | 30 | 規則：那位老師開的課都算相關 | 已完成 |
| 查精確課名 | 30 | 規則：同名的課都算相關 | 已完成 |
| 主題（「機器學習」） | 20 | 人工，0 / 1 / 2 三級 | 未標註 |
| 口語長句（「想學統計但數學不太好」） | 20 | 人工 | 未標註 |

- 指標：Recall@10、MRR@10、nDCG@10；未標註的課一律算不相關，所以另外報 judged@10（前 10 名中已標註的比例）。
- 系統間差異用**分層成對 bootstrap**（2,000 次）估 95% 信賴區間：同一組題目逐題相減，各題型分開重抽。
- k 值掃描：在規則層上 k=10 的三個指標都最好（nDCG@10 0.891 對 0.853）。但規則層偏向字面比對，
  **人工層標完之前不建議改 k**。

```sh
cd ~/NCCU_AI_Project/CourseLangChain
uv run python -m evaluation.build_gold sample   # 抽規則層題目
uv run python -m evaluation.build_gold pool     # 匯出人工層待標註候選 → 編輯 evaluation/judgments.tsv
uv run python -m evaluation.run_eval            # 產生 evaluation/results/report.md
uv run python -m tests.test_eval_metrics        # 指標本身的測試，不需要資料庫
```

## 實測案例（2026-11-15）

在後端容器裡直接呼叫 `_pg_retrieve(keyword, 5, sql_filter)`，不經過語言模型：

| 查詢 | 前 5 名 | 解讀 |
|---|---|---|
| 人工智慧 | 人工智慧方法與工具 ×2、計算思維與人工智慧 ×2、人工智慧實務專題 | 字面與語意一致，兩路都排前面 |
| 翁堃嵐（老師名） | 第 1、4 名是他開的課，其餘三門無關 | BM25 找到名字；向量那一路對名字沒有幫助，只帶進雜訊 |
| 資料庫 | 資料分析與程式設計入門 ×4、生活中的資訊資源與應用 | 複合詞被切開；兩門「資料庫應用」排第 10、11 名 |
| 想學統計但數學不太好 | 教育統計學、傳播統計、多變量分析 ×2、數值分析 | 抓到「統計」，但「數學不太好」的意圖沒有反映在排序上 |
| 人工智慧 + 星期三下午 | 媒體科技與創新法律、資料科學核心技術專題、宗教文創… | 符合時段的 340 門裡課名與 AI 相關的只有 2 門，其餘用弱相關的課補滿 |
| 人工智慧 + 不要星期五 | 與不加條件時相近，星期五的「計算思維與人工智慧」被排除 | 排除條件正常運作 |

---

## 已知問題與改進方向

依影響排序。

1. **`sql_filter` 原樣接進 SQL，而且資料庫用超級使用者連線。**
   `retrieve_tool` 的 `sql_filter` 參數由模型填寫，直接以字串插入查詢（`_pg_retrieve`）；
   `constraints.py` 的 `unit` 與 `lang` 也是字串插入。後端連線帳號是 `postgres`（`docker-compose.yml`）。
   目前沒有被利用的紀錄，但經由提示詞注入，模型可能在 `sql_filter` 裡放進任意 SQL。
   改法：後端改用唯讀角色連線；`retrieve_tool` 改收 JSON，SQL 只在程式內由 `constraints_to_where` 產生；字串值改用參數綁定。
2. **沒有相關度門檻。** 條件很嚴時會用弱相關的課補滿 `top_k`。
   可能的改法：設 RRF 分數或兩路名次的門檻，不足時回「只找到 N 門明顯相關的課」。改之前要先有人工層的評估數字。
3. **複合詞切詞。** 「資料庫」切成「資料」與「庫」。
   可能的改法：jieba 自訂詞典，或加一個字元 n-gram 欄位（v4 的 BM25 就改用字元 n-gram，`contract.yaml` 的 `char12`）。
4. **含英文字母的課號拿不到課程卡片。** 115-1 有 131 門課的課號不是 13 碼純數字。
   `parse_formatted_docs`（`tools/retrieve.py`）只收純數字，這些課在聊天回答裡看得到，但不會出現在候選卡片；
   `tools/session_schedule.py` 的 9 碼補學期也只處理純數字。`GET /api/courses/{id}` 已經接受字母（`^\d{4}[0-9A-Z]{9}$`）。
5. **同分時順序不固定。** 只在一路出現、名次相同的兩門課 RRF 分數完全相同，SQL 沒有次要排序。
   評估程式已用課號打破同分；正式系統可在 `_RRF_SQL` 最後的 `ORDER BY rrf DESC` 補上 `, id`。
6. **評估缺人工層。** 見上方「離線評估」。標完之前，關於語意型查詢的結論都只是推測。
7. **檢索 SQL 與 `constraints_to_where` 沒有單元測試。** 目前靠離線評估間接涵蓋；時段轉 SQL 的邊界情況（跨時段、`include` 與 `exclude` 同時出現）值得補測試。
8. **v4 資料尚未接上。** CourseDataPrep 從 2026-11 起多產一個 `v4` schema（e5-large-instruct 1024 維向量、字元 n-gram BM25）。
   後端還沒有任何查詢用到它；執行中的資料庫也還沒有 `v4` schema。改用 v4 時，本文件與離線評估都要更新。

### 最近修正

- **工具內部的模型輸出被當成答案顯示**（2026-11-15，`fix/stream-hide-tool-output`）。
  問含時段的問題時，`text_to_sql_tool` 內部那次模型呼叫回的 `{"include_times": …}` 會出現在畫面最前面。
  `main.py` 的 `astream` 改為只收主迴圈（`langgraph_node == "agent"`）的輸出，回歸測試 `tests/test_stream_filter.py`。

---

## 修改檢索時

1. 改之前先跑一次離線評估，留下對照數字。
2. 改 `_RRF_SQL` 時，`evaluation/modes.py` 會自動用到新的 SQL；「只用向量」那句是照抄 `vec_raw` 寫的，要一起改。
3. 改 `retrieve_tool` 的輸出格式時，`parse_formatted_docs` 的正規表示式要一起改（三者相鄰擺放），否則前端卡片會消失。
4. 換 embedding 模型或改切塊方式，是 **CourseDataPrep** 的工作：在那邊改 `contract.yaml`、重跑 `migrate` 與 `embed`、重出資料包，
   再把 `contract.yaml` 原樣複製過來。**不要只在後端改模型名**。
5. 跑測試：`uv run python -m tests.test_stream_filter`、`uv run python -m tests.test_eval_metrics`，再用一題含時段的問題實際問一次。

## 檔案地圖

| 檔案 | 內容 |
|---|---|
| [`tools/retrieve.py`](../tools/retrieve.py) | `retrieve_tool`、`_RRF_SQL`、`_BM25_ONLY_SQL`、查詢編碼、輸出格式與 `parse_formatted_docs` |
| [`tools/text_to_sql.py`](../tools/text_to_sql.py) | `text_to_sql_tool`：把條件寫成 JSON 的提示詞 |
| [`tools/constraints.py`](../tools/constraints.py) | JSON → SQL 條件 |
| [`tools/course_detail.py`](../tools/course_detail.py) | 查單一門課的詳細內容 |
| [`tools/course_info.py`](../tools/course_info.py) | 課程卡片 API 的資料（`GET /api/courses/{id}`） |
| [`main.py`](../main.py) | `astream`：側通道送出候選課程與排課方案 |
| [`agents/brain_agent.py`](../agents/brain_agent.py) | 系統提示：什麼時候查課、`top_k` 用多少 |
| [`paths.py`](../paths.py) | `RRF_K`、`RRF_CANDIDATES`、`DATABASE_URL`、`contract.yaml` 讀取 |
| [`contract.yaml`](../contract.yaml) | 資料契約（從 CourseDataPrep 原樣複製） |
| [`evaluation/`](../evaluation/README.md) | 離線評估 |
| `CourseDataPrep/init-scripts/03-schema.sql` | `content_text` 與 BM25 索引 |
| `CourseDataPrep/init-scripts/05-chunks.sql`、`src/prep/chunk.py` | 向量切塊與不建 HNSW 的理由 |

另見：[CourseDataPrep 文件](https://github.com/NCCU-AI-SYSTEM/CourseDataPrep/blob/main/docs/README.md)（資料怎麼建出來）、
[部署整合文件的檢索總覽](https://github.com/NCCU-AI-SYSTEM/CourseLangChain-all/blob/main/docs/developer/retrieval.md)。
