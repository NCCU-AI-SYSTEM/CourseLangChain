# Personalized Course Scheduling Agent Blueprint

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

目前專案後端核心比較接近「RAG 問答鏈」：

- 從 `vectorstore.pkl` 取回課程相關內容
- 將使用者問題與檢索結果送入固定 prompt
- 回傳表格式課程回答

這種做法適合課程查詢，但不足以支援：

- 多步驟決策
- 個人化約束處理
- 排課最佳化
- 使用者偏好記憶
- 多方案比較

因此建議升級為以 orchestration 為核心的 agent architecture。

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

### `schedule_planner_tool`

用途：

- 根據條件產生候選課表

輸入：

- candidate course pool
- hard constraints
- soft constraints

輸出：

- 3 to 10 candidate schedules

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
CourseLangChain/
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

## Phase 1: Agent foundation

目標：

- 將單一 RAG 改為可調用 tool 的 Brain Agent

交付：

- session-aware API
- tool calling interface
- basic user profile loading
- basic course catalog lookup

## Phase 2: Constraint-based scheduling

目標：

- 能生成合法課表

交付：

- conflict validation
- prerequisite check
- credit range check
- candidate schedule generation

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
