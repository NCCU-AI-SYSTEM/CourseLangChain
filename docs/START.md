# 本機啟動

每個服務開一個終端機，照順序跑。全部起來後用瀏覽器開 **http://localhost:3000**。

| # | 服務 | Port | 位置 |
|---|------|------|------|
| 1 | PostgreSQL | 5433 | 本 repo |
| 2 | Moodle MCP | 3033 | nccu-moodle-mcp repo |
| 3 | 後端(FastAPI) | 8000 | 本 repo |
| 4 | 前端(Vite) | 3000 | [CourseLangGraph-frontend](https://github.com/NCCUCourseScheduling/CourseLangGraph-frontend) repo |

LLM 另外跑,`.env` 的 `OPENAI_BASE_URL` 指向它(預設 `http://localhost:4000/v1`)。
沒開的話聊天會回 `OpenAIConnectionError`。

## 1. 資料庫

Docker 有開就只要跑一次，之後會一直在背景。

在本 repo 根目錄:

```bash
docker compose up -d postgres
```

5432 被別的專案佔用，所以 `.env` 設了 `POSTGRES_PORT=5433`,`DATABASE_URL` 也指向 5433。

## 2. Moodle MCP

在 nccu-moodle-mcp repo 根目錄:

```bash
uv run nccu-moodle-mcp http
```

要在後端之前啟動。後端先起來的話 Moodle 工具不會載入，重開後端即可。
畫面上出現「Moodle 服務暫時無法連線」就是這個沒開。

## 3. 後端

在本 repo 根目錄:

```bash
uv run python app.py
```

API 文件:http://localhost:8000/docs

## 4. 前端

在前端 repo 根目錄:

```bash
npx pnpm@9 run dev
```

- 用 `feat/add-to-schedule-from-chat` 分支(有 Moodle、課表、個人資料等功能)。
- `vite.config.ts` 的 proxy 要指向後端:`"/api": "http://localhost:8000"`。
- `node_modules` 若是在 Linux 容器裡裝的,macOS 上會報 esbuild 平台錯誤，刪掉重裝:
  `rm -rf node_modules && npx pnpm@9 install`

## 常見問題

| 症狀 | 原因 |
|------|------|
| `address already in use`(8000) | 已經有一個後端在跑,`lsof -iTCP:8000 -sTCP:LISTEN` 找出來關掉 |
| Moodle 服務暫時無法連線 | Moodle MCP(3033)沒開 |
| `OpenAIConnectionError` | LLM(4000)沒開 |
| log 一直出現 `localhost:3000 ... otel` 錯誤 | Langfuse 沒開，不影響功能;把 `.env` 的 `LANGFUSE_*_KEY` 清空就會消失 |
