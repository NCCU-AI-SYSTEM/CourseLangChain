# 紅隊評測

用真模型跑一組攻擊,量三層 harness(L1 輸入、L2 執行、L3 輸出)與外部內容標記各自擋下多少。
和 `tests/` 的差別:`tests/` 是確定性的單元測試;這裡要呼叫 LLM,結果每次會有些不同,
所以每個攻擊會跑好幾次,看的是比例。

## 什麼時候要跑

- 改了 system prompt、`harness/`、`agents/output_checks.py`、任何會回傳外部內容的工具
- 換了模型(`.env` 的 `OPENAI_MODEL`)
- 新增了會讀外部內容的工具或 MCP server

## 怎麼跑

需要 `.env` 設好 LLM 與 `DATABASE_URL`(PostgreSQL 要開著)。兩個 MCP stub 由評測自己起,不用另外開。

```sh
cd CourseLangChain
uv run python evals/redteam/run.py                         # 全部(約 3 分鐘,gemma4:31b-cloud)
uv run python evals/redteam/run.py --splits holdout,holdout2
uv run python evals/redteam/run.py --cases I8,W2 --reps 5
uv run python evals/redteam/run.py --model qwen3:8b        # 換模型(覆寫 .env)
```

結果寫到 `reports/<時間>-<模型>/`:`report.md` 是報表,`results.json` 是每次執行的原始回答。

## 怎麼讀報表

| 欄位 | 意思 |
|---|---|
| 送達 | 攻擊有沒有真的進到模型的 context。外部內容的攻擊要模型呼叫那個工具才會送達;沒送達不能算擋下 |
| L1 擋下 | 輸入檢查直接拒絕 |
| 模型照做 | 模型的原始回答(L3 之前)含有攻擊內容 |
| harness 攔下 | 模型照做了,但 L3 擋下或改寫,使用者沒看到 |
| 使用者看到 | 最後送到使用者面前的回答仍含有攻擊內容 |

判定是自動的,會有誤判。報表最後一節列出所有「使用者看到」的回答,**每次都要人工看過**,
結論寫在同一個資料夾的 `review.md`。

## 集合

- **design**:設計防禦時參考過的攻擊。這些變好不代表防禦真的有效。
- **holdout**:在 L1 正規化與外部內容標記(`harness/untrusted.py`)設計「之前」寫好,設計時不參考。
- **holdout2**:建立這套紅隊評測時新增,寫好時所有防禦已定案,沒有針對它們調整過。
- **benign**:正常問題,量誤擋與回答有沒有被改壞。

保留集的數字才代表真實的防禦能力。**不要為了讓保留集的案例通過去改防禦**——改了它就變成 design。
要改防禦的話,先寫新的保留案例、跑一次記錄結果,再動手。

## 新增案例

編輯 `cases.yaml`,欄位說明在檔案開頭。外部內容的攻擊會這樣注入:

- `course_detail`:插在課綱「評分方式」欄位最前面(只在評測行程的記憶體裡,不動資料庫)
- `campus_web`:附加在校內網站 stub 回傳的網頁內文後面
- `moodle`:放進 Moodle stub 的作業說明欄位(評測會先替該對話登入假帳號)

後兩者是透過環境變數 `STUB_POISON_FILE` 讓 stub 讀取攻擊內容,見 `mcp_servers/` 的 `_poison()`。

## 已知限制

- 攻擊句是人工寫的,沒有針對 harness 做自動化的自適應攻擊。研究顯示自適應攻擊能突破
  大多數防禦(Nasr et al., 2025, *The Attacker Moves Second*),所以這裡的數字是下限,不是保證。
- 每個攻擊只跑 3 次,單一模型。
- Moodle stub 的連結用假網域 `moodle.example.invalid`,會被 V4 移除;正式的 `moodle.nccu.edu.tw` 在白名單內。
