"""校內網站檢索的**假** MCP server —— 給本機開發與測試用,不是正式服務。

正式版由另一組(NCCU-Crawler)實作,工具合約見同目錄 README.md。這支只負責兩件事:
1. 對方服務還沒好之前,CourseLangChain 就能把 MCP 串接、工具路由、錯誤處理跑通;
2. 當作合約的可執行範例 —— 對方照這個簽名與回傳格式實作,後端不必改。

資料是寫死的幾筆,**內容不是真實公告**(每筆都有標示);網址取自 NCCU-Crawler README
裡實際爬過的頁面。

跑:    uv run python -m mcp_servers.campus_web_stub [--host 127.0.0.1] [--port 8765]
後端:  CAMPUS_WEB_MCP_URL=http://localhost:8765/mcp

用 mcp 1.x 的 FastMCP:langchain-mcp-adapters 0.3.2 明確限制 `mcp<2`,
0.3.0 雖沒寫上限但實測在 mcp 2.x 會 ImportError。對方的 server 用 2.x 不受影響(協定相容)。
"""
from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

_STUB_NOTE = "【PoC 假資料,非真實公告】"

_PAGES: list[dict] = [
    {
        "title": "教務處註冊組",
        "url": "https://aca.nccu.edu.tw/zh/關於本處/註冊組",
        "keywords": ("註冊組", "註冊", "學籍", "休學", "復學", "退學", "畢業"),
        "text": "註冊組承辦學籍相關業務,例如休學、復學、退學與畢業資格審查。",
    },
    {
        "title": "教務處最新消息(註冊組)",
        "url": "https://aca.nccu.edu.tw/zh/最新消息/註冊組",
        "keywords": ("公告", "最新消息", "申請", "期限", "截止", "休學"),
        "text": "註冊組發布的各項申請公告與辦理期限。",
    },
    {
        "title": "教務處",
        "url": "https://aca.nccu.edu.tw",
        "keywords": ("教務處", "教務", "加退選", "選課時間", "行事曆"),
        "text": "教務處網站,提供註冊、課務等各組的業務說明與公告。",
    },
]


def search_campus_web(query: str, max_results: int = 3) -> str:
    """查詢政大校內各單位網站(教務處、註冊組等)的公告、規章與行政資訊。
    使用者問學校行政流程、單位資訊、申請辦法、校務日期時呼叫;不是用來查課程。"""
    q = query.strip()
    if not q:
        return "ERROR: query 不可為空"

    scored = [(sum(k in q for k in page["keywords"]), page) for page in _PAGES]
    hits = [page for score, page in sorted(scored, key=lambda x: -x[0]) if score > 0]
    if not hits:
        return f"找不到與「{q}」相關的校內網站資料。"

    return "\n\n".join(
        f"[{i}] {page['title']}\n來源: {page['url']}\n內容: {_STUB_NOTE}{page['text']}"
        for i, page in enumerate(hits[: max(1, max_results)], start=1)
    )


def build_server(host: str, port: int) -> FastMCP:
    """host 必須在建構時就給:FastMCP 依它決定 DNS rebinding 防護。

    綁 127.0.0.1 時只接受 Host 為 localhost 的請求 —— 容器經 host.docker.internal
    連進來會被回 421,所以要讓容器連,server 得用 --host 0.0.0.0 起。
    """
    server = FastMCP(
        name="campus_web_stub",
        instructions="政大校內網站檢索(PoC 假資料)。",
        host=host,
        port=port,
    )
    server.tool()(search_campus_web)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="校內網站檢索的假 MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    build_server(args.host, args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
