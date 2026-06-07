"""排課引擎 — 純函數核心(段 1)。

**這裡不能出現任何 LLM 呼叫。** 全部是可決定性的 pure function:
solver(DFS+剪枝)、衝堂 validator。段 2 才把這些包成 `schedule_tool`。

對外提供:
- `parse_course_slots(time_str)`  — 把編碼時間字串解析成佔用的 (星期, 節次) 集合
- `has_conflict(a, b)`            — 兩門課是否時間衝堂
- `validate_schedule(...)`        — 回傳違規清單(空 = 合法),排課流程中**不可跳過**
- `find_schedules(...)`           — DFS 列舉所有合法課表組合

衝堂模型刻意用「slot 集合交集」而非時鐘時間:同一 (星期, 節次) 被兩門課佔用即衝堂,
不需要換算成幾點幾分,既正確又不會被 `getSessionArray` 的時鐘換算邏輯影響。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from utils.time import getSessionArray, weekdayCode


@dataclass(frozen=True)
class CourseSlot:
    """排課計算用的最小課程表示(只放 solver 需要的欄位)。"""

    course_id: str
    name: str
    credits: float
    time_str: str
    teacher: str = ""
    # 佔用的 (星期字, 節次字) 集合,例如 {("三","2"),("三","3"),("三","4")}
    slots: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    @property
    def weekdays(self) -> set[str]:
        """這門課用到的星期(中文字)集合。"""
        return {wd for wd, _ in self.slots}


def parse_course_slots(time_str: str) -> frozenset[tuple[str, str]]:
    """把編碼時間字串解析成佔用的 (星期, 節次) slot 集合。

    複用 `utils.time.getSessionArray`,再從每段的 `origin_str`(如 "三234")
    反解出星期字 + 各節次字。時間未定 / 空字串 → 空集合(視為不佔任何 slot)。
    """
    sessions = getSessionArray(time_str or "")
    slots: set[tuple[str, str]] = set()
    for s in sessions:
        origin = s.get("origin_str", "")
        if not origin:
            continue
        weekday = origin[0]
        for period_char in origin[1:]:
            slots.add((weekday, period_char))
    return frozenset(slots)


def make_course(
    course_id: str,
    name: str,
    credits: float,
    time_str: str,
    teacher: str = "",
) -> CourseSlot:
    """工廠:把原始欄位轉成 `CourseSlot`(順便解析 slots)。"""
    return CourseSlot(
        course_id=str(course_id),
        name=name,
        credits=float(credits or 0),
        time_str=time_str or "",
        teacher=teacher or "",
        slots=parse_course_slots(time_str),
    )


def has_conflict(a: CourseSlot, b: CourseSlot) -> bool:
    """兩門課是否時間衝堂(佔用同一 (星期, 節次))。"""
    return bool(a.slots & b.slots)


def _normalize_weekdays(avoid_weekdays: list[str] | set[str] | None) -> set[str]:
    """把避開星期正規化成中文字集合,容忍 "五" / "fri"(忽略) / 數字字串等輸入。"""
    if not avoid_weekdays:
        return set()
    out: set[str] = set()
    for w in avoid_weekdays:
        w = str(w).strip()
        if not w:
            continue
        if w in weekdayCode:
            out.add(w)
        elif w.isdigit():  # "1"->一 ... "7"->日
            idx = int(w) - 1
            if 0 <= idx < len(weekdayCode):
                out.add(weekdayCode[idx])
    return out


def validate_schedule(
    courses: list[CourseSlot],
    min_credits: float,
    max_credits: float,
    avoid_weekdays: list[str] | set[str] | None = None,
) -> list[str]:
    """檢查一組課表是否違反硬限制,回傳違規描述清單(空清單 = 合法)。

    這是排課流程中**不可跳過**的最終把關,即使 solver 已剪枝過也要再驗一次。
    """
    violations: list[str] = []
    avoid = _normalize_weekdays(avoid_weekdays)

    # 1. 兩兩衝堂
    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            overlap = courses[i].slots & courses[j].slots
            if overlap:
                slot_str = ", ".join(f"{wd}{p}" for wd, p in sorted(overlap))
                violations.append(
                    f"衝堂:「{courses[i].name}」與「{courses[j].name}」在 {slot_str} 重疊"
                )

    # 2. 學分上下限
    total = sum(c.credits for c in courses)
    if total < min_credits:
        violations.append(f"學分不足:{total} < 下限 {min_credits}")
    if total > max_credits:
        violations.append(f"學分超標:{total} > 上限 {max_credits}")

    # 3. 觸碰到要避開的星期
    if avoid:
        for c in courses:
            hit = c.weekdays & avoid
            if hit:
                violations.append(f"「{c.name}」落在要避開的星期 {''.join(sorted(hit))}")

    # 4. 重複修課(同名課出現多次)
    seen: set[str] = set()
    for c in courses:
        if c.name in seen:
            violations.append(f"重複修課:「{c.name}」在同一課表出現多次")
        seen.add(c.name)

    # 5. 無有效上課時段(時間未定/彈性)無法排進週課表
    for c in courses:
        if not c.slots:
            violations.append(f"「{c.name}」上課時間未定,無法排入週課表")

    return violations


def find_schedules(
    courses: list[CourseSlot],
    min_credits: float,
    max_credits: float,
    avoid_weekdays: list[str] | set[str] | None = None,
    max_results: int = 5,
) -> list[list[CourseSlot]]:
    """DFS + 剪枝,列舉所有「無衝堂、學分在 [min,max]、不碰避開星期」的課表組合。

    剪枝策略:
    - 觸碰避開星期的課,進 DFS 前先整批濾掉(硬限制,沒有討論空間)
    - 累積學分一旦超過上限 → 該分支停止
    - 候選課若與已選課衝堂 → 跳過(不納入)
    回傳最多 `max_results` 組合,每組合內按 course_id 排序以穩定輸出。
    """
    avoid = _normalize_weekdays(avoid_weekdays)
    # 排除「時間未定」(無 slot 無法排入週課表)與「落在避開星期」的課
    pool = [c for c in courses if c.slots and not (c.weekdays & avoid)]

    results: list[list[CourseSlot]] = []

    def dfs(start: int, chosen: list[CourseSlot], credits: float, names: set[str]) -> None:
        if len(results) >= max_results:
            return
        if min_credits <= credits <= max_credits and chosen:
            results.append(sorted(chosen, key=lambda c: c.course_id))
            # 不 return:允許在合法基礎上繼續加課產生更多方案,但仍受上限剪枝
        for k in range(start, len(pool)):
            cand = pool[k]
            if credits + cand.credits > max_credits:
                continue  # 超上限,剪掉
            if cand.name in names:
                continue  # 同名課(同一門的不同班)一張課表只取一門,避免重複修課
            if any(has_conflict(cand, c) for c in chosen):
                continue  # 與已選衝堂,剪掉
            chosen.append(cand)
            dfs(k + 1, chosen, credits + cand.credits, names | {cand.name})
            chosen.pop()
            if len(results) >= max_results:
                return

    dfs(0, [], 0.0, set())
    return results


def _distinct_weekdays(schedule: list[CourseSlot]) -> int:
    """這張課表總共用到幾個不同的星期(越少 = 越集中)。"""
    days: set[str] = set()
    for c in schedule:
        days |= c.weekdays
    return len(days)


def rank_schedules(schedules: list[list[CourseSlot]]) -> list[list[CourseSlot]]:
    """對候選課表排序(MVP 啟發式,純函數,可決定性)。

    排序準則(由優先到次要):
    1. 上課天數越少越好(課越集中,空檔少)
    2. 學分越高越好(在合法範圍內盡量修滿)
    3. course_id 序列(打破平手,確保輸出穩定)
    後續段 3+ 接 preference / workload 後,這裡會被更完整的 ranker 取代。
    """
    return sorted(
        schedules,
        key=lambda s: (
            _distinct_weekdays(s),
            -sum(c.credits for c in s),
            tuple(c.course_id for c in s),
        ),
    )


def format_schedules_markdown(schedules: list[list[CourseSlot]]) -> str:
    """把候選課表格式化成 Markdown(純函數,給 tool 直接回傳)。"""
    if not schedules:
        return "找不到符合條件的課表組合。"

    blocks: list[str] = []
    for idx, sched in enumerate(schedules, 1):
        total = round(sum(c.credits for c in sched), 1)
        days = "".join(sorted(_distinct_weekdays_chars(sched)))
        lines = [
            f"### 方案 {idx}(共 {total} 學分,上課日:{days or '未定'})",
            "| 課程名稱 | 上課時間 | 授課老師 | 學分 |",
            "|----------|----------|----------|------|",
        ]
        for c in sorted(sched, key=lambda x: x.course_id):
            lines.append(
                f"| {c.name} | {c.time_str or '未定'} | {c.teacher or 'N/A'} | {round(c.credits, 1)} |"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _distinct_weekdays_chars(schedule: list[CourseSlot]) -> set[str]:
    days: set[str] = set()
    for c in schedule:
        days |= c.weekdays
    return days
