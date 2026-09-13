# 校內網站檢索 MCP 工具合約（草案）

給維護校內網站問答服務（NCCU-Crawler）的組員。

CourseLangChain 的 agent 會透過 MCP 連到你們的服務，由模型自己決定什麼時候呼叫。
照這份合約實作，我們這邊不必改程式，只要設定一個網址。

同目錄的 `campus_web_stub.py` 是照這份合約寫的假 server，可以直接拿來當範本。

## 連線方式

- 傳輸方式：MCP **streamable HTTP**，端點路徑 `/mcp`（例如 `http://<主機>:8765/mcp`）。
- SDK 版本不限。已實測我們的 client（`mcp` 1.30）可以連 `mcp` 2.2 寫的 server。
- **要讓 Docker 容器連進來，server 必須綁 `0.0.0.0`。** 綁 `127.0.0.1` 時 SDK 會自動開啟
  DNS rebinding 防護，容器經 `host.docker.internal` 連線會收到 **HTTP 421**。

## 工具規定

| 項目 | 規定 | 原因 |
|---|---|---|
| 名稱 | 暫定 `search_campus_web`；不可與我們的工具同名（`retrieve_tool`、`schedule_tool` 等） | 同名的工具會被略過，只記一行 log |
| 參數 | 只用扁平的 `str` / `int` / `float` / `bool`，不要巢狀物件 | 小模型填巢狀參數的錯誤率很高 |
| 說明（docstring） | 第一句寫「什麼時候該呼叫」，並寫明**不是用來查課程** | 模型靠這段文字挑工具 |
| 回傳 | 純文字：檢索到的段落 + 來源網址（格式見下） | agent 回答時要附來源 |
| 查無資料 | 回白話訊息，例如 `找不到與「…」相關的校內網站資料。` | |
| 失敗 | 回 `ERROR: <原因>` 字串，或讓 SDK 回報工具錯誤 | 兩種都會轉成 `ERROR:` 交給模型，對話不會中斷 |
| 耗時 | 列工具清單 10 秒內、單次呼叫 60 秒內 | 超過就當成服務無法使用 |

建議的回傳格式：

```
[1] <頁面標題>
來源: <網址>
內容: <段落>

[2] ...
```

## 請回傳「段落」，不要回傳「LLM 寫好的答案」

我們的 agent 本身就有 LLM 會整理回答。如果你們的工具也先用 LLM 寫好答案：

- 一個問題要跑兩次 LLM，等待時間變長。
- Gemini 免費額度更容易撞到 429（實測兩分鐘內問第二題就會碰到）。
- 我們拿不到原始段落，沒辦法附來源，也沒辦法檢查模型有沒有亂編。

## 本機測試

```bash
# 1. 起 server（你們的，或這支假的）
uv run python -m mcp_servers.campus_web_stub --port 8765

# 2. 在 CourseLangChain/.env 加這一行，然後重啟後端
CAMPUS_WEB_MCP_URL="http://localhost:8765/mcp"

# 3. 接線測試（不需要 LLM，也不需要資料庫）
uv run python -m tests.test_mcp_tools
```

server 沒開時，後端照常啟動，只是少了校內網站工具，log 會出現一行「連不上校內網站 MCP server」。
