"""排課函式的 scratch / playground —— 「打個輸入看輸出」用。

用法(VSCode):
  1. 確認右下角 Python 直譯器選到本專案的 .venv
  2. 游標放進任一個 `# %%` 區塊,按 Shift+Enter → 結果出現在 Interactive Window
  3. 改輸入再按一次即可,不用重跑整個檔

也可整檔當一般 script 跑:`python -m tests.scratch_scheduler`
"""

# %% setup —— 第一次先跑這格(切到專案根、讓 import 找得到 tools/utils)
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../CourseLangChain/tests
_ROOT = os.path.dirname(_HERE)                       # .../CourseLangChain
os.chdir(_ROOT)                                      # 讓 data.db 等相對路徑可用
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.scheduler import (  # noqa: E402
    find_schedules,
    has_conflict,
    make_course,
    parse_course_slots,
    validate_schedule,
)

print("setup done, cwd =", os.getcwd())


# %% 測 parse_course_slots —— 改這個字串再 Shift+Enter
parse_course_slots("三234五56")


# %% 測 has_conflict —— 兩門課是否衝堂
a = make_course("1", "課A", 3, "三234")
b = make_course("2", "課B", 3, "三34")  # 改時間字串看結果變化
has_conflict(a, b)


# %% 測 validate_schedule —— 回傳違規清單(空 list = 合法)
validate_schedule([a, b], min_credits=0, max_credits=99, avoid_weekdays=None)


# %% 測 find_schedules —— 用真實 data.db 排課
import sqlite3  # noqa: E402

conn = sqlite3.connect("data.db")
rows = conn.execute(
    "SELECT id,name,point,time FROM COURSE WHERE time NOT IN ('','未定或彈性') LIMIT 15"
).fetchall()
conn.close()

pool = [make_course(*r) for r in rows]
schedules = find_schedules(pool, min_credits=6, max_credits=12, max_results=3)
for s in schedules:
    print(round(sum(c.credits for c in s), 1), "學分:", [c.name for c in s])
