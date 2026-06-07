import sys
import sqlite3, pickle, json
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents.base import Document
from utils.time import getSessionArray, weekdayCode
from utils.retriever import EnsembleRetriever
from paths import COURSE_SEMESTER, COURSE_YEAR, DATA_DB, VECTORSTORE_PKL


class ClassDocument(Document):
  def __init__(self, content: dict) -> None:
    super().__init__(page_content=str(content), metadata=content)
    
    timeStr = ""
    sessionArray = getSessionArray(content["time"])
    if len(sessionArray) > 0:
      for i, session in enumerate(sessionArray):
        if i > 0:
          timeStr += "、"
        timeStr += f'星期{weekdayCode[session["week_code"] - 1]} {session["start_time"]}:00-{session["end_time"]}:00'
    else:
      timeStr = "未定"
    
    #contentStr = "課程名稱: {}\n課程內容: {}\n上課時間: {}\n老師名稱: {}".format(content["name"], content["objective"], timeStr, content["teacher"])
    #contentStr = "課程名稱是{}, 課程內容有{}, 上課時間是{}, 這堂課的老師是{}".format(content["name"], content["objective"], timeStr, content["teacher"])
    contentStr = "課程名稱是{}, 上課時間是{}, 這堂課的老師是{}".format(content["name"], timeStr, content["teacher"])
    
    self.page_content = contentStr
    self.metadata = content

def dict_factory(cursor, row):
  d = {}
  for idx, col in enumerate(cursor.description):
      d[col[0]] = row[idx]
  return d

def document_factory(cursor, row):
  return ClassDocument(dict_factory(cursor, row))

def build(y: str, s: str, dataFile=DATA_DB, vectorStorePkl=VECTORSTORE_PKL, embeddingModel="BAAI/bge-m3"):
  con = sqlite3.connect(dataFile)
  con.row_factory = document_factory
  cursor = con.cursor()
  
  req = cursor.execute("SELECT * FROM COURSE WHERE y = ? AND s = ?", (y, s))
  res = req.fetchall()
  
  # embedding 一律綁 CPU,讓產出的 pickle 不依賴 CUDA(部署目標是 CPU 容器)。
  # 即使在有 GPU 的機器上建,也在 CPU 上 embed(慢一點,但一次性),確保容器載得動。
  embeddings = HuggingFaceEmbeddings(
    model_name=embeddingModel,
    model_kwargs={"device": "cpu"},
  )

  # initialize the faiss retriever
  vectorStore = FAISS.from_documents(res, embedding=embeddings)
  faiss_retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

  # initialize the bm25 retriever
  bm25_retriever = BM25Retriever.from_documents(res)
  bm25_retriever.k = 5

  # initialize the ensemble retriever (BM25 + FAISS 各半)
  ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

  with open(vectorStorePkl, "wb") as f:
    pickle.dump(ensemble_retriever, f)
      

if __name__ == "__main__":
  # 未給參數時用 paths.py 的 COURSE_YEAR/COURSE_SEMESTER(與查詢端同一來源,避免脫鉤)
  if len(sys.argv) == 1:
    y, s = COURSE_YEAR, COURSE_SEMESTER
  elif len(sys.argv) == 3:
    y, s = sys.argv[1], sys.argv[2]
  else:
    print("Usage: python build.py [<year> <semester>]")
    print("Example: python build.py 114 2  (省略則用 paths.py 的 COURSE_YEAR/SEMESTER)")
    sys.exit(1)
  print(f"Building index for year={y}, semester={s} ...")
  build(y, s)
