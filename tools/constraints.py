from __future__ import annotations

import json as json_mod
from dataclasses import dataclass
from typing import Any


@dataclass
class TimeSlot:
    weekday: int | None = None
    start_hour: int = 0
    end_hour: int = 24


@dataclass
class TimeFilter:
    include_times: list[TimeSlot] | None = None
    exclude_times: list[TimeSlot] | None = None
    lang: str | None = None
    kind: int | None = None
    point_min: float | None = None
    point_max: float | None = None
    unit: str | None = None


def _hour_to_flag(start: int, end: int) -> str | None:
    if start >= 8 and end <= 12:
        return "has_morning"
    if start >= 12 and end <= 13:
        return "has_noon"
    if start >= 13 and end <= 18:
        return "has_afternoon"
    if start >= 18:
        return "has_evening"
    return None


def _slot_to_exists(slot: TimeSlot, negate: bool) -> str:
    prefix = "NOT " if negate else ""
    cond = (
        f"(elem->>'start_hour')::int < {slot.end_hour}"
        f" AND (elem->>'end_hour')::int > {slot.start_hour}"
    )
    if slot.weekday is not None:
        cond = f"(elem->>'weekday')::int = {slot.weekday} AND " + cond
    return (
        f"{prefix}EXISTS ("
        f"SELECT 1 FROM jsonb_array_elements(sessions) AS elem(item) WHERE {cond}"
        f")"
    )


def _make_include_sql(slot: TimeSlot) -> str | None:
    if slot.weekday is None:
        flag = _hour_to_flag(slot.start_hour, slot.end_hour)
        if flag:
            return f"{flag} = true"
    return _slot_to_exists(slot, negate=False)


def _make_exclude_sql(slot: TimeSlot) -> str | None:
    if slot.weekday is None:
        flag = _hour_to_flag(slot.start_hour, slot.end_hour)
        if flag:
            return f"{flag} = false"
    return _slot_to_exists(slot, negate=True)


def constraints_to_where(tf: TimeFilter | None) -> str:
    if tf is None:
        return ""
    clauses: list[str] = []

    if tf.include_times:
        parts = [_make_include_sql(s) for s in tf.include_times if _make_include_sql(s)]
        if parts:
            clauses.append(f"({' OR '.join(parts)})")

    if tf.exclude_times:
        for s in tf.exclude_times:
            sql = _make_exclude_sql(s)
            if sql:
                clauses.append(sql)

    if tf.lang:
        clauses.append(f"lang = {tf.lang!r}")
    if tf.kind is not None:
        clauses.append(f"kind = {tf.kind}")
    if tf.point_min is not None:
        clauses.append(f"point >= {tf.point_min}")
    if tf.point_max is not None:
        clauses.append(f"point <= {tf.point_max}")
    if tf.unit:
        clauses.append(f"unit ILIKE '%{tf.unit}%'")

    return " AND ".join(clauses)


def parse_timefilter_from_json(json_str: str) -> TimeFilter:
    data = json_mod.loads(json_str)

    def _parse_slot(s: dict) -> TimeSlot:
        return TimeSlot(
            weekday=s.get("weekday"),
            start_hour=s["start_hour"],
            end_hour=s["end_hour"],
        )

    kwargs: dict[str, Any] = {}
    if "include_times" in data:
        kwargs["include_times"] = [_parse_slot(s) for s in data["include_times"]]
    if "exclude_times" in data:
        kwargs["exclude_times"] = [_parse_slot(s) for s in data["exclude_times"]]
    for key in ("lang", "kind", "point_min", "point_max", "unit"):
        if key in data:
            kwargs[key] = data[key]

    # 模型常把沒問到的可選欄位填成 0 / "" / null 一併回傳。空字串靠
    # constraints_to_where 的 truthy 檢查就擋掉了,但 kind 用的是 `is not None`,
    # 所以 kind=0 會變成真的 `AND kind = 0` —— 而 0 在資料裡存在(115-1 有 46 門),
    # 於是「晚上的課」從 294 門悄悄變成 6 門,沒有任何錯誤訊息。
    if kwargs.get("kind") not in (1, 2, 3, 4):
        kwargs.pop("kind", None)

    return TimeFilter(**kwargs)
