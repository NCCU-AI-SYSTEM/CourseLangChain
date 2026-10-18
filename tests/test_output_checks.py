"""L3 第 2 段:agent 層輸出規則(V2 課號落地、V3 prompt 洩漏、V4 外部連結、V5 工具呼叫外洩)。

跑:`uv run python -m tests.test_output_checks`(在 CourseLangChain/ 下)
不需要 LLM。V2 的「真實課號」案例需要 PostgreSQL,連不上時略過該案例。

正常回答的樣本取自 2026-10-19 用 gemma4:31b-cloud 實際跑出的回覆:這些規則最怕的是
誤擋正常回答,所以誤擋測試一律用真模型的輸出,不用自己編的句子。
"""
from __future__ import annotations

from harness import check_output, current_context
from harness import output_validator as ov
from agents import output_checks as oc
from agents.brain_agent import SYSTEM_PROMPT

_PASS = 0
_FAIL = 0
_SKIP = 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✓ {msg}")
    else:
        _FAIL += 1
        print(f"  ✗ FAIL: {msg}")


def skip(msg: str) -> None:
    global _SKIP
    _SKIP += 1
    print(f"  ⊘ SKIP: {msg}")


# 真模型的正常回答(節錄)
REAL_CHAT = """你好！我是 NCCU 課程查詢系統的助手，我可以幫你：

1. **查詢課程**：幫你尋找特定主題或符合時間條件的課程清單。
2. **查看詳情**：提供單一課程的教學內容、評分方式、教科書等詳細課綱。
3. **志願序建議**：查詢課程過去的選課難度，建議志願序該如何填寫。
4. **自動排課**：根據你的學分需求與避開時段，幫你規劃無衝堂的課表方案。
5. **管理課表**：幫你查看、增加或移除目前已排定的課程。"""
REAL_SCHEDULE = """為您推薦三組管理相關的課表方案，皆已避開星期五且學分在 6 至 9 學分之間：

### 方案 1 (共 9.0 學分, 上課日: 二, 三)
| 課程名稱 | 上課時間 | 授課老師 | 學分 |
|----------|----------|----------|------|
| 管理學 | 二234 | 賴佳怡 | 3.0 |
| 教育模範領導與管理實踐 | 三78E | 顏敏仁 | 3.0 |
| 成長管理專題研究 | 二EFG | 賴宗裕 | 3.0 |"""
REAL_QUERY = """| 課程名稱 | 上課時間 | 授課老師 |
|----------|----------|----------|
| 人工智慧方法與工具 | 四D56 | 陳昭伶 |
| 計算思維與人工智慧 | 四78E | 吳怡潔 |"""
REAL_REFUSAL = "我無法提供系統提示的原文。如果您有任何關於 NCCU 課程查詢、排課或志願序建議的需求，歡迎隨時告訴我！"
REAL_SAMPLES = {"閒聊": REAL_CHAT, "排課": REAL_SCHEDULE, "查課": REAL_QUERY, "拒絕": REAL_REFUSAL}

FAKE_ID = "1151999999999"


def _real_course_id() -> str | None:
    try:
        import psycopg2

        from paths import COURSE_SEMESTER, COURSE_YEAR, DATABASE_URL

        conn = psycopg2.connect(DATABASE_URL)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM public.course WHERE y = %s AND s = %s ORDER BY id LIMIT 1",
                (COURSE_YEAR, COURSE_SEMESTER),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None


def test_v2_course_ids() -> None:
    print("[V2 課號落地]")
    ok, msg = oc.check_course_ids(f"推薦你修課號 {FAKE_ID} 的課,它沒有衝堂。")
    check(not ok and FAKE_ID in msg, "編造的課號被擋下")

    v = check_output(f"課號 {FAKE_ID} 在本學期查無此課。", context={"user_input": f"{FAKE_ID} 是什麼課"})
    check(v.ok, "使用者自己輸入的課號,回答覆述「查無此課」不算編造")

    check(oc.check_course_ids("學校電話 0229393091,學號 112703052")[0], "其他長度的數字不當成課號")
    check(oc.check_course_ids("編號 11519999999999 共 14 碼")[0], "不從更長的數字中間截出 13 碼")
    check(oc.check_course_ids(REAL_SCHEDULE)[0], "沒有課號的正常回答直接通過,不查資料庫")

    real = _real_course_id()
    if real is None:
        skip("連不上 PostgreSQL,略過真實課號案例")
    else:
        check(oc.check_course_ids(f"這門課的課號是 {real}。")[0], f"當期真實課號 {real} 通過")
        ok, msg = oc.check_course_ids(f"可以選 {real} 或 {FAKE_ID}。")
        check(not ok and FAKE_ID in msg and real not in msg, "真假混在一起時,只點名假的那個")

    import tools.session_schedule as ss

    original = ss.existing_course_ids

    def broken(_ids):
        raise ConnectionError("db down")

    ss.existing_course_ids = broken
    try:
        check(oc.check_course_ids(f"課號 {FAKE_ID}")[0], "資料庫掛掉時放行(不讓所有回答一起失敗)")
    finally:
        ss.existing_course_ids = original


def test_v3_prompt_leak() -> None:
    print("[V3 系統提示洩漏]")
    leak_check = oc.make_prompt_leak_check(SYSTEM_PROMPT)
    for name, text in REAL_SAMPLES.items():
        check(leak_check(text)[0], f"真模型的正常回答「{name}」不誤擋")

    check(not leak_check("好的,以下是我的設定:" + SYSTEM_PROMPT[1200:1240])[0], "照貼 40 字原文被擋下")
    check(not leak_check(SYSTEM_PROMPT)[0], "整份 prompt 被擋下")
    reformatted = SYSTEM_PROMPT[200:420].replace("\n", "  ").replace("1.", "**第一**")
    check(not leak_check(reformatted)[0], "改過換行與 Markdown 的照貼仍被擋下")
    check(leak_check(SYSTEM_PROMPT[1200:1220])[0], "只引用 20 字以內的片段不算洩漏")

    import agents.brain_agent  # noqa: F401 — 建 agent 時會登記 V3

    v = check_output(SYSTEM_PROMPT[:300])
    check(
        not v.ok and v.failed == "system_prompt_leak" and v.user_message == oc.PROMPT_LEAK_MESSAGE,
        f"brain_agent 已把 V3 登記進 L3(實際:{v.failed})",
    )


def test_v4_links() -> None:
    print("[V4 外部連結]")
    f = oc.strip_external_links
    check(f("詳見[這裡](https://evil.example/?d=修課紀錄)。") == "詳見這裡。", "外部連結移除,保留文字")
    keep = "請到[選課系統](https://qrysub.nccu.edu.tw/)查詢。"
    check(f(keep) == keep, "學校網域的連結保留")
    check(f("[政大](http://www.nccu.edu.tw)") == "[政大](http://www.nccu.edu.tw)", "學校子網域、http 也保留")
    check(f("[x](https://nccu.edu.tw.evil.com/)") == "x", "假冒網域(nccu.edu.tw.evil.com)被移除")
    check(f("[x](https://evilnccu.edu.tw/)") == "x", "假冒網域(evilnccu.edu.tw)被移除")
    check(f("[x](javascript:alert(1))") == "x", "javascript: 連結被移除")
    check(f("結尾 ![圖](https://evil.example/a.png)") == "結尾 圖", "外部圖片只留替代文字")
    check(f("![校徽](https://www.nccu.edu.tw/logo.png)") == "校徽", "圖片一律移除,學校網域的也一樣")
    check(f('[x](https://evil.example/ "標題")') == "x", "帶標題的連結也能辨認")
    check(f("[x](<https://evil.example/a b>)") == "x", "用 <> 包住的網址也能辨認")
    check("evil" not in f("見[說明][1]。\n\n[1]: https://evil.example/?d=1"), "參考式連結的外部網址被移除")
    check(f("<https://evil.example/>") == "", "自動連結 <https://...> 被移除")
    check(f(REAL_SCHEDULE) == REAL_SCHEDULE, "沒有連結的課表原封不動")

    v = check_output("課程清單如下 ![x](https://evil.example/?d=1)")
    check(v.ok and "evil" not in v.output, "改寫器透過 check_output 生效")


def test_v4_parser_oracle() -> None:
    """regex 認不出、但 markdown-it 會渲染成連結的寫法。

    判定標準是真正的 Markdown 解析結果(與前端同一套規則),不是 regex 自己的認知:
    「改寫後解析不出外部連結」才算過。這幾種寫法都是第一版 regex 實際漏掉的。
    """
    print("[V4 以 Markdown 解析器為準:regex 漏掉的寫法]")
    bypasses = {
        "跳脫的 ]": r"[a\]b](https://evil.example/1)",
        "巢狀 []": "[a [b] c](https://evil.example/2)",
        "連結文字跨行": "[第一行\n第二行](https://evil.example/3)",
        "參考式定義換行": "見[說明][r]\n\n[r]:\nhttps://evil.example/4",
        "collapsed 參考": "見[說明][]\n\n[說明]: https://evil.example/5",
        "shortcut 參考": "見[說明]\n\n[說明]: https://evil.example/6",
        "實體編碼的 scheme": "[x](&#104;ttps://evil.example/7)",
        "大寫 scheme": "[x](HTTPS://EVIL.EXAMPLE/8)",
        "協定相對網址": "[x](//evil.example/9)",
    }
    for name, text in bypasses.items():
        check(bool(oc.find_unsafe_links(text)), f"「{name}」改寫前確實會渲染成外部連結(測資有效)")
        out = oc.strip_external_links(text)
        check(not oc.find_unsafe_links(out), f"「{name}」改寫後解析不出外部連結")

    mixed = "請到[選課系統](https://qrysub.nccu.edu.tw/)。另見[a [b]](https://evil.example/)"
    out = oc.strip_external_links(mixed)
    check(not oc.find_unsafe_links(out), "合法與惡意連結混在一起時,退路仍確保沒有外部連結")
    check("選課系統" in out, "退路只跳脫符號,文字內容保留")

    for name, text in REAL_SAMPLES.items():
        check(oc.strip_external_links(text) == text, f"真模型的正常回答「{name}」不觸發退路")


def test_v5_tool_call_leak() -> None:
    print("[V5 工具呼叫外洩]")
    check(oc.is_tool_call_text('{"name": "retrieve_tool", "arguments": {"keyword": "AI"}}'), "工具呼叫 JSON 被認出")
    check(oc.is_tool_call_text('```json\n{"name": "x", "parameters": {}}\n```'), "包在 code fence 裡也認得出")
    check(not oc.is_tool_call_text('{"name": "人工智慧導論", "teacher": "王"}'), "使用者要的 JSON 不誤擋")
    v = check_output('{"name": "my_schedule_tool", "arguments": {"action": "view"}}')
    check(not v.ok and v.user_message == oc.TOOL_CALL_LEAK_MESSAGE, "透過 check_output 回專屬訊息")


def test_pipeline() -> None:
    print("[L3 管線:順序、例外、context]")
    v = check_output(f"課號 {FAKE_ID} 請看 [這裡](https://evil.example/)")
    check(not v.ok and v.failed == "course_id_grounding", "驗證失敗時直接拒絕,不跑改寫器")

    def boom(_out: str):
        raise RuntimeError("bug")

    ov._VALIDATORS["_test_boom"] = ov._Entry(boom, None)
    try:
        v = check_output("一般回覆")
        check(not v.ok and v.failed == "_test_boom", "驗證器自己出錯 → 視為失敗(寧可不回答)")
    finally:
        del ov._VALIDATORS["_test_boom"]

    check_output("x", context={"user_input": "abc"})
    check(current_context() == {}, "check_output 結束後 context 復原,不殘留到下一輪")
    for name, text in REAL_SAMPLES.items():
        v = check_output(text, context={"user_input": "測試"})
        check(v.ok and v.output == text, f"真模型的正常回答「{name}」整條 L3 原樣通過")


if __name__ == "__main__":
    test_v2_course_ids()
    test_v3_prompt_leak()
    test_v4_links()
    test_v4_parser_oracle()
    test_v5_tool_call_leak()
    test_pipeline()
    print(f"\n結果:{_PASS} passed, {_FAIL} failed, {_SKIP} skipped")
    raise SystemExit(1 if _FAIL else 0)
