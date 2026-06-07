"""集中管理資料檔的絕對路徑,**不依賴程序的工作目錄(cwd)**。

問題背景:`data.db` / `vectorstore.pkl` 用相對路徑時,若 server / IDE 從非
`CourseLangChain/` 的目錄啟動,SQLite 會自動建一個空 db(→ `no such table: COURSE`)、
pickle 則 FileNotFoundError。這裡以「本檔所在目錄」為專案根錨定,徹底避開 cwd 問題。
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DB = os.path.join(PROJECT_ROOT, "data.db")
VECTORSTORE_PKL = os.path.join(PROJECT_ROOT, "vectorstore.pkl")
