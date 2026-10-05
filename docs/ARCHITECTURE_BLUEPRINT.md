# Personalized Course Scheduling Agent Blueprint

> [!WARNING]
> **這份文件的「現況」註記停在 2026-06,檢索那一段已經過時。**
> 設計目標與工具切分仍然有效,但以下描述已經不是現在的實作:
>
> | 文中寫的 | 現在實際上是 |
> |---|---|
> | `query_courses_tool` | `retrieve_tool`(`tools/query_courses.py` 只剩相容轉接) |
> | FAISS(bge-m3)+ BM25 ensemble | PostgreSQL 上的 ParadeDB `pg_search`(`pdb.jieba`)+ pgvector,在 SQL 裡做 RRF 融合 |
> | `vectorstore.pkl` / `build.py` | 已刪除;資料成品改由 `course-data-prep` repo 產生 |
> | `data.db`(SQLite) | 已棄用(`USE_SQLITE`),預設走 PostgreSQL |
>
> 目前的架構與安裝流程以 [`SETUP.md`](SETUP.md) 與專案根目錄的 `README.md` 為準。

## 1. Goal

將目前的單一課程查詢 RAG 系統，重構為「大腦 Agent + 多工具 + 排課引擎」的個人化排課平台。

新系統目標：

- 能理解學生背景與排課目標
- 能調用課程資料、使用者資料、課程地圖與畢業規則
- 能自動產出多個一週課表方案
- 能避免衝堂並處理硬性限制
- 能考慮期中、期末、作業量等壓力因素
- 能解釋排課原因並支援互動式調整

---

## 2. Current State Summary

> 更新於 2026-06：藍圖第一波(Brain Agent + 三層 harness + 排課引擎)已落地，本節改為記錄實際現況，不再是原本的「RAG 問答鏈」。

### 2.1 已實作

後端已從原本「固定 5 節點 graph(`route → sql_gen → validate_sql → retrieve → respond`)」升級為 **ReAct Brain Agent + 三層 harness**：

- **Brain Agent**(`agents/brain_agent.py`)：用 `langgraph.prebuilt.create_react_agent` 包住工具，由 LLM 依意圖自行決定 tool 呼叫順序，取代舊的條件分支。
- **三層 harness**(`harness/`)：把正確性/安全性從不可信的 agent 推進確定性程式碼。
  - L1 `input_sanitizer`：prompt injection 偵測 + 長度上限
  - L2 `SafeAgentExecutor`：step 上限(recursion_limit)+ wall-clock timeout + 例外永不外漏
  - L3 `output_validator`：各 tool 自註冊驗證器；排課器註冊「無衝堂」對最終輸出再驗一次
- **已上線工具**(`tools/`)：
  - `query_courses_tool` —— 即藍圖的 `course_catalog_tool`，查課與排課共用的檢索。FAISS(bge-m3)+ BM25 ensemble；帶 `sql_filter` 時改走純 BM25。回傳含 13 位 `course_id` + 學分。
  - `text_to_sql_tool` —— 自然語言時間限制 → SQL WHERE 子句，當作 query 的 `sql_filter`。
  - `course_detail_tool` —— 單一門課詳情(課綱 / 評分 / 教科書 / 上課方式…)。
  - `schedule_tool` + `tools/scheduler.py` —— 純函數排課引擎(DFS+剪枝 → 驗證 → 排序 → Markdown)，零 LLM。對應藍圖的 conflict_validation + schedule_planner(MVP)。

### 2.2 近期修正(2026-06)

本波針對檢索與排課的正確性修了數個 bug：

- **檢索鎖當前學期**：`data.db` 跨學年共 10 萬+ 筆，`query_courses` / `course_detail` 原本沒鎖 `y/s`，會撈到歷年同名課。改由 `paths.COURSE_YEAR / COURSE_SEMESTER` 單一來源鎖定(`build.py` 共用)。
- **中文檢索修復**：BM25 預設 `str.split()` 對中文整句變成單一 token，關鍵字在 `sql_filter` 路徑完全失效(查「管理」回體育課)。導入 jieba 斷詞(`utils/zh_tokenize.py`，無 jieba 時退回字元 bigram)，並重建 `vectorstore.pkl` 讓 ensemble 的 BM25 半邊也乾淨。
- **時間解析強健化**：`utils/time.getSessionArray` 遇未知字元(逗號/空白)會 crash、且「只有星期」會產生幻影節次，連帶可能讓 L3 驗證誤拒正確課表。已重寫解析迴圈。
- **text_to_sql prompt**：移除自相矛盾的「ReAct 思考 + 只輸出 SQL」指示，改為純單句輸出。

### 2.3 仍待補(對齊後續 Phase)

相對於完整藍圖，目前仍缺：

- 個人化：`user_profile` / `preference_memory` / 畢業規則 / 課程地圖(Phase 3)
- 壓力模型：`workload` / 期中期末(Phase 4)
- 互動微調：what-if / 多方案比較 UI(Phase 5)
- 結構化資料：目前仍以單一 `COURSE` 表為主，尚未拆出 meeting / curriculum / load 等表(見 §7)

---

## 3. Target Architecture

```text
Frontend Chat UI
    |
    v
API Gateway / Session Layer
    |
    v
Brain Agent
    |
    +--> Profile Tools
    |      - user_profile_tool
    |      - preference_memory_tool
    |
    +--> Academic Tools
    |      - course_catalog_tool
    |      - curriculum_map_tool
    |      - graduation_rule_tool
    |
    +--> Planning Tools
    |      - conflict_validation_tool
    |      - schedule_planner_tool
    |      - workload_estimation_tool
    |      - schedule_ranker_tool
    |      - alternative_plan_tool
    |
    +--> Explanation Tools
           - plan_explainer_tool
           - what_if_simulation_tool

Underlying Services
    - course service
    - user service
    - rule engine
    - optimization engine
    - memory service

Data Sources
    - course_db
    - user_db
    - curriculum_db
    - exam_load_db
    - feedback_db
```

---

## 4. Core Design Principle

### 4.1 LLM only does what it is good at

LLM 負責：

- 理解意圖
- 補全需求
- 協調工具
- 產生說明
- 比較方案與做語意化回覆

### 4.2 Rules and scheduling are not left to the LLM

規則引擎或排程演算法負責：

- 衝堂檢查
- 學分限制
- 必修與先修條件
- 課表生成
- 候選方案排序

### 4.3 Data must be structured

若資料只有文字檢索，很難做穩定排課。建議建立結構化欄位：

- 課程時間
- 學分
- 必選修
- 先修課
- 開課系所
- 授課教師
- 評量方式
- 期中期末週
- 作業量估計

---

## 5. Agent and Tool Responsibilities

## 5.1 Brain Agent

職責：

- 解析使用者目標
- 判斷要調用哪些 tool
- 統整多方資料
- 觸發排課流程
- 回傳推薦方案與解釋

典型輸入：

- 「我是資管三年級，這學期想修 18 學分，避免週五課」
- 「我想壓力低一點，但還是要補畢業學分」
- 「如果我一定要修資料庫，幫我重排」

典型輸出：

- 推薦課表 A
- 備選課表 B / C
- 每個方案的優缺點
- 建議取捨

---

## 5.2 Essential Tools

### `course_catalog_tool`

用途：

- 查詢本學期開課清單
- 回傳課程時間、教師、學分、課程代碼、限制條件

輸入：

- semester
- department filters
- keyword
- required_only

輸出：

- structured course list

> 現況：已實作為 `tools/query_courses.py`(`query_courses_tool`)。已鎖當前學期(`y/s`)、BM25 導入 jieba 中文斷詞；keyword/sql_filter 雙模式可用。`required_only`、department filter 尚未支援。

### `user_profile_tool`

用途：

- 取得學生個人背景與已知偏好

輸入：

- user_id

輸出：

- department
- grade
- completed_courses
- current_plan
- preferred_credit_range
- preference profile

### `curriculum_map_tool`

用途：

- 取得課程地圖與推薦修課路徑

輸入：

- department
- grade

輸出：

- required courses
- elective groups
- recommended sequence

### `graduation_rule_tool`

用途：

- 檢查畢業條件與缺口

輸入：

- user_profile
- completed_courses
- candidate_courses

輸出：

- missing requirements
- must_take list
- rule violations

### `conflict_validation_tool`

用途：

- 檢查課程時間與制度衝突

輸入：

- selected courses

輸出：

- time conflicts
- duplicated credits
- prerequisite violations
- semester rule violations

> 現況：衝堂、學分上下限、避開時段、重複修課(同名去重)、時間未定排除 已由 `tools/scheduler.py::validate_schedule` 實作,並在 L3 對 agent 最終輸出再驗一次。先修條件(prerequisite)檢查尚未做。

### `schedule_planner_tool`

用途：

- 根據條件產生候選課表

輸入：

- candidate course pool
- hard constraints
- soft constraints

輸出：

- 3 to 10 candidate schedules

> 現況：MVP 已實作為 `tools/schedule_tool.py` + `tools/scheduler.py`(DFS + 剪枝列舉合法組合 → 排序 → Markdown),純函數零 LLM。排序目前用啟發式(上課天數少、學分高);尚未接 workload / preference 等軟限制評分。

### `workload_estimation_tool`

用途：

- 為每個課表估算學期與週期壓力

輸入：

- selected courses
- exam/load metadata

輸出：

- weekly load score
- midterm pressure score
- final pressure score
- assignment intensity
- overall stress score

### `schedule_ranker_tool`

用途：

- 依照使用者目標排序候選課表

輸入：

- candidate schedules
- user preference profile

輸出：

- ranked schedules
- ranking reason

### `plan_explainer_tool`

用途：

- 產生可讀性高的解釋與建議

輸入：

- final ranked schedules

輸出：

- human-readable comparison
- recommendation summary

---

## 5.3 Recommended Additional Tools

### `preference_memory_tool`

記住長期偏好，例如：

- 不要早八
- 週五盡量空堂
- 一天不要超過四門
- 偏好某幾位老師
- 希望集中兩到三天上課

### `alternative_plan_tool`

產出不同策略版本：

- 畢業優先型
- 壓力最小型
- 專業探索型
- 空堂最多型

### `what_if_simulation_tool`

支援互動調整：

- 如果改成週四不上課會怎樣
- 如果一定要修某堂必修會怎樣
- 如果學分降到 15 會怎樣

### `feedback_learning_tool`

蒐集學生後續回饋：

- 實際修課壓力是否如預期
- 推薦課是否喜歡
- 推薦老師是否合適

這會讓之後的 workload 與 preference 模型更準。

---

## 6. End-to-End Workflow

## 6.1 Primary flow

```text
1. User sends request
2. Brain Agent parses intent
3. Brain Agent loads user profile
4. Brain Agent loads curriculum and graduation requirements
5. Brain Agent fetches course catalog for the target semester
6. Rule engine filters out invalid courses
7. Planner engine generates multiple schedules
8. Workload engine scores each schedule
9. Ranker sorts schedules based on goals
10. Explainer generates final recommendation
11. User requests refinement
12. Brain Agent reruns only affected parts
```

## 6.2 Example scenario

使用者：

「我是資管三年級，想修 18 學分，避免週五與早八，這學期壓力不要太高，但要補齊畢業需求。」

系統步驟：

1. 讀取學生系級、已修課、缺少學分與個人偏好
2. 載入資管系課程地圖與畢業門檻
3. 查詢本學期可選課
4. 移除不符先修、衝堂、已修過課程
5. 生成數個合法課表
6. 計算每個課表的期中期末壓力與單週密度
7. 排序並挑出最適方案
8. 以自然語言解釋取捨

---

## 7. Data Model Blueprint

## 7.1 Course table

建議核心欄位：

```text
courses
- course_id
- semester
- name
- department
- teacher
- credits
- required_type
- category
- capacity
- language
- description
- prerequisites
- restrictions
- grading_policy
- workload_score
- midterm_week
- final_week
```

## 7.2 Course meeting table

```text
course_meetings
- meeting_id
- course_id
- weekday
- period_start
- period_end
- location
```

## 7.3 User profile table

```text
user_profiles
- user_id
- student_id
- department
- grade
- program_type
- target_credit_min
- target_credit_max
- scheduling_goal
```

## 7.4 User academic record

```text
user_course_history
- user_id
- course_id
- semester
- grade_result
- passed
```

## 7.5 User preference table

```text
user_preferences
- user_id
- avoid_early_classes
- avoid_friday
- max_courses_per_day
- preferred_teachers
- preferred_days
- compact_schedule
- stress_tolerance
```

## 7.6 Curriculum rules

```text
curriculum_rules
- department
- program_type
- rule_id
- category
- minimum_credits
- required_course_ids
- elective_group_ids
- prerequisite_rule
```

## 7.7 Course load metadata

```text
course_load_profiles
- course_id
- assignment_intensity
- exam_intensity
- project_intensity
- reading_intensity
- overall_difficulty
```

---

## 8. Constraint Design

排課必須區分硬限制與軟限制。

## 8.1 Hard constraints

不可違反：

- 時間衝堂
- 學分超出上限或低於下限
- 先修條件未滿足
- 必修漏修
- 重複修課
- 年級或系所限制
- 開課學期限制

## 8.2 Soft constraints

盡量滿足：

- 不要早八
- 週五少課或沒課
- 每日課程數量平衡
- 壓力分散
- 期中期末不集中
- 偏好教師
- 減少空堂
- 某些天保留做專題或打工

---

## 9. Schedule Scoring Blueprint

每個候選課表可以用 weighted score 排序：

```text
total_score =
  graduation_fit * 0.30 +
  preference_fit * 0.20 +
  stress_score * 0.20 +
  timetable_compactness * 0.10 +
  teacher_preference * 0.10 +
  exploration_value * 0.10
```

可依產品策略調整權重。

### 建議評分維度

- `graduation_fit`
  是否有效補足畢業缺口

- `preference_fit`
  是否符合個人偏好

- `stress_score`
  期中期末與作業壓力是否過高

- `compactness`
  是否減少零碎空堂

- `teacher_preference`
  是否符合教師偏好

- `diversity_or_exploration`
  是否符合探索新領域需求

---

## 10. API Blueprint

建議將現有單一路由擴充為以下 API：

### `POST /api/agent/chat`

用途：

- 與 Brain Agent 對話

request:

```json
{
  "user_id": "u123",
  "message": "我是資管三年級，幫我排下學期課表"
}
```

### `GET /api/users/{user_id}/profile`

用途：

- 查詢學生背景與偏好

### `GET /api/courses`

用途：

- 查詢課程清單與條件篩選

### `POST /api/schedules/generate`

用途：

- 直接要求系統生成候選課表

### `POST /api/schedules/evaluate`

用途：

- 對某組課表做衝堂與壓力評估

### `POST /api/schedules/simulate`

用途：

- 進行 what-if 模擬

### `POST /api/preferences/update`

用途：

- 更新使用者長期偏好

---

## 11. Suggested Backend Module Layout

建議把後端拆成：

```text
CourseLangGraph/
  agents/
    brain_agent.py
    planner_agent.py

  tools/
    course_catalog.py
    user_profile.py
    curriculum_map.py
    graduation_rules.py
    conflict_validation.py
    schedule_planner.py
    workload_estimator.py
    schedule_ranker.py
    plan_explainer.py
    preference_memory.py
    what_if_simulation.py

  services/
    course_service.py
    user_service.py
    curriculum_service.py
    planner_service.py
    scoring_service.py

  repositories/
    course_repository.py
    user_repository.py
    curriculum_repository.py
    schedule_repository.py

  models/
    schemas.py
    course.py
    user.py
    schedule.py

  engines/
    rule_engine.py
    optimizer.py
    workload_engine.py

  api/
    routes_agent.py
    routes_user.py
    routes_schedule.py
```

---

## 12. Frontend Blueprint

前端不應只是一個純聊天框，建議逐步發展成：

### 12.1 Chat workspace

用途：

- 跟 Brain Agent 對話
- 填寫偏好
- 看推薦說明

### 12.2 Schedule comparison panel

用途：

- 並列比較方案 A / B / C
- 顯示學分、衝堂、壓力分數、空堂數

### 12.3 Weekly timetable view

用途：

- 視覺化一週課表
- 顯示期中與期末風險標註

### 12.4 Preference editor

用途：

- 設定早八、週五、學分、壓力容忍度
- 設定教師偏好與上課日偏好

### 12.5 Graduation progress dashboard

用途：

- 顯示哪些畢業需求已完成、哪些還缺

---

## 13. MVP Roadmap

## Phase 1: Agent foundation　（大致完成）

目標：

- 將單一 RAG 改為可調用 tool 的 Brain Agent

交付：

- session-aware API
- tool calling interface　✅(ReAct Brain Agent + 三層 harness)
- basic user profile loading　⬜(尚未,user_profile_tool 未做)
- basic course catalog lookup　✅(`query_courses_tool`)

## Phase 2: Constraint-based scheduling　（部分完成）

目標：

- 能生成合法課表

交付：

- conflict validation　✅(`scheduler.validate_schedule` + L3 再驗)
- prerequisite check　⬜(尚未)
- credit range check　✅
- candidate schedule generation　✅(`schedule_tool` + `scheduler.find_schedules`)

## Phase 3: Personalization

目標：

- 課表開始有個人化差異

交付：

- preference memory
- schedule ranking
- multiple plan styles

## Phase 4: Stress-aware planning

目標：

- 納入考試與作業壓力

交付：

- workload estimator
- midterm/final pressure scoring
- stress-aware recommendation

## Phase 5: Interactive refinement

目標：

- 使用者可反覆調整條件

交付：

- what-if simulation
- one-click regenerate
- plan comparison UI

---

## 14. Priority Recommendations

如果你們現在要開始做，我建議優先順序如下：

1. 先把「課程查詢鏈」改成「Brain Agent + Tool routing」
2. 建立結構化的 course/user/curriculum schema
3. 先完成硬限制排課器，不要一開始就追求很聰明
4. 再加入使用者偏好與排序
5. 最後補壓力模型與互動模擬

原因是：

- 沒有結構化資料，很難做穩定排課
- 沒有硬限制引擎，LLM 很容易排出錯誤課表
- 沒有排序模型，個人化會停留在表面

---

## 15. Key Risks

### 15.1 Incomplete data

若只有課名與時間，無法做好壓力評估與畢業規則檢查。

### 15.2 Over-reliance on LLM

若直接讓模型自由生成課表，容易出現衝堂、違規或幻覺。

### 15.3 Weak feedback loop

若沒有收集學生實際回饋，壓力估算會一直停留在人工假設。

### 15.4 One-shot planning UX

排課通常不是一次完成，需要支援多輪微調與比較。

---

## 16. First Refactor Proposal for This Repo

> 進度(2026-06)：Step 1(brain_agent)、Step 2(course_catalog ← `query_courses_tool`)、Step 5(schedule_planner MVP ← `schedule_tool`)、Step 6(部分,排課輸出已是 Markdown 方案但尚無 `plan_explainer_tool`)已完成;Step 3(結構化 repository 層)、Step 4(`user_profile_tool` + mock user)尚未做。詳見 §2.1。

基於目前 repo，第一波建議不要一次大改到底，可以先做這些：

### Step 1

保留現有 FastAPI 與前端聊天介面，但新增一層 `brain_agent.py`

### Step 2

把目前 retriever-based 課程查詢包成第一個 tool：

- `course_catalog_tool`

### Step 3

新增結構化的 SQLite repository：

- 讀 `data.db`
- 回傳標準化課程資料

### Step 4

新增 `user_profile_tool` 與 mock user data

### Step 5

新增 `schedule_planner_tool` 的 MVP 版本：

- 先只處理衝堂
- 學分上下限
- 使用者避開時段

### Step 6

新增 `plan_explainer_tool`

讓回覆不只是表格，而是：

- 推薦方案
- 原因
- 風險
- 可替代方案

---

## 17. Definition of Done

當以下條件成立，就代表第一版藍圖落地成功：

- 系統能辨識使用者身份與偏好
- 系統能查課並過濾不合法課程
- 系統能自動生成至少 3 個可行課表
- 系統能解釋每個方案的差異
- 系統能根據「不要早八 / 避開週五 / 壓力低」這類需求重新排課
- 系統能支援後續擴充更多 tool

---

## 18. Suggested Next Deliverables

接下來可以直接往下做的文件或實作：

1. tool input/output schema spec
2. backend folder refactor plan
3. sqlite data schema migration
4. schedule planner MVP algorithm design
5. frontend wireframe for schedule comparison

如果要快速推進，下一步最值得做的是：

「先把 Brain Agent 與 Tool interface 的骨架寫出來，再把目前課程查詢能力接成第一個 tool。」
