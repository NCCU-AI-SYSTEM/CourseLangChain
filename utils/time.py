timeChar = [
    "A",
    "B",
    "1",
    "2",
    "3",
    "4",
    "C",
    "D",
    "5",
    "6",
    "7",
    "8",
    "E",
    "F",
    "G",
    "H",
  ]
weekdayCode = ["一", "二", "三", "四", "五", "六", "日"]

def getSessionArray(time_str: str):
  if time_str == "未定或彈性" or time_str == "":
    return []
  current_weekday = 1
  current_start_time = 0
  current_end_time = 0
  last_time = 0
  recording = False
  res = []
  key = 0

  def get_start_time(char: str):
    index = timeChar.index(char)
    time = 6
    if index: time += index
    return time

  def get_end_time(char: str):
    index = timeChar.index(char)
    time = 7
    if index: time += index
    return time

  def push_into_res():
    tmp = {
      "week_code": -1,
      "start_time": -1,
      "end_time": -1,
      "origin_str": ""
    }
    tmp["week_code"] = current_weekday + 1
    tmp["start_time"] = get_start_time(timeChar[current_start_time])
    tmp["end_time"] = get_end_time(timeChar[current_end_time])
    origin_str = weekdayCode[current_weekday]
    for i in range(current_start_time, current_end_time + 1):
      origin_str += timeChar[i]
    tmp["origin_str"] = origin_str
    res.append(tmp)

  for char in time_str:
    if char in weekdayCode:
      if recording:
        push_into_res()
        recording = False
      current_weekday = weekdayCode.index(char)
    elif char in timeChar:
      idx = timeChar.index(char)
      if not recording:
        current_start_time = idx
        recording = True
      elif idx != last_time + 1:
        push_into_res()
        current_start_time = idx
      current_end_time = idx
      last_time = idx
    # 其餘字元(逗號 / 空白 / 全形等)略過,不讓整串解析 crash

  if recording:  # 僅在確有記錄到節次時才收尾,避免「只有星期」產生幻影節次
    push_into_res()
  return res