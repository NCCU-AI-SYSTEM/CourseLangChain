import sys
import sqlite3, pickle, json
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents.base import Document
from utils.time import getSessionArray, weekdayCode
from utils.retriever import EnsembleRetriever

import torch


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

def build(y: str, s: str, dataFile="data.db", vectorStorePkl="vectorstore.pkl", embeddingModel="BAAI/bge-m3"):
  con = sqlite3.connect(dataFile)
  con.row_factory = document_factory
  cursor = con.cursor()
  
  req = cursor.execute("SELECT * FROM COURSE WHERE y = ? AND s = ?", (y, s))
  res = req.fetchall()
  
  if(torch.cuda.is_available()):
  
    embeddings = HuggingFaceEmbeddings(model_name=embeddingModel)

    # initialize the faiss retriever
    vectorStore = FAISS.from_documents(res, embedding=embeddings)
    faiss_retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
      
    # initialize the bm25 retriever
    bm25_retriever = BM25Retriever.from_documents(res)
    bm25_retriever.k = 5

    # initialize the ensemble retriever
    # ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0, 1])
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    
    with open(vectorStorePkl, "wb") as f:
      pickle.dump(ensemble_retriever, f)
  else:
    bm25_retriever = BM25Retriever.from_documents(res)
    bm25_retriever.k = 5
    
    with open(vectorStorePkl, "wb") as f:
      pickle.dump(bm25_retriever, f)
      

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python build.py <year> <semester>")
    print("Example: python build.py 113 1")
    sys.exit(1)
  build(sys.argv[1], sys.argv[2])
