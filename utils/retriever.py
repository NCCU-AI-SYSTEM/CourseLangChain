from langchain.retrievers import EnsembleRetriever as OriginEnsembleRetriever

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents.base import Document
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Any, Dict, List


# ============================================================================
# 原有的 EnsembleRetriever 類 - 用於序列化和載入
# ============================================================================
class EnsembleRetriever(OriginEnsembleRetriever):
    """自訂的集合檢索器，用於多個檢索方法的混合。"""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """取得相關文檔。"""
        res = super()._get_relevant_documents(query, run_manager=run_manager)
        print([r.metadata["name"] for r in res])
        return res


# ============================================================================
# LangChain Tool 定義
# ============================================================================
class RetrievalInput(BaseModel):
    """檢索工具的輸入結構。"""

    query: str = Field(
        description="要檢索的查詢字符串。應該是清晰、具體的課程或校務相關問題，例如：'計算機科學系有哪些必修課程？' 或 '大一微積分的課程時間是何時？'"
    )


@tool(args_schema=RetrievalInput)
def retrieval_tool(query: str) -> str:
    """
    RAG 課程檢索工具 - 從向量資料庫中搜尋相關課程和校務資訊。

    **使用時機：**
    - 當需要查詢特定課程的信息（名稱、時間、教師、教室）
    - 當需要查詢某個學院或系所的課程列表
    - 當需要搜尋校務相關的學術資訊
    - 當用戶提出關於課程內容、開課時間、教學團隊等問題時

    **預期回傳結果：**
    - 返回一個格式化的字符串，包含檢索到的相關課程和校務文檔
    - 每個檢索結果會列出文檔的名稱和相關內容
    - 如果找不到相關結果，會返回適當的提示信息

    **示例：**
    輸入：query = "計算機科學系 大一 課程"
    輸出：格式化的課程信息列表，包含相關的課程名稱、時間、教師等詳細資訊

    Args:
        query: 用戶的查詢字符串

    Returns:
        格式化的檢索結果字符串，包含所有相關課程和校務資訊的摘要
    """
    # 注意：此工具需要在 Agent 初始化時與檢索器實例綁定
    # 實現細節應在使用端處理，例如在 agent_tools 中注入檢索器實例
    pass


# ============================================================================
# 工廠函數 - 用於生成綁定了檢索器的 Tool
# ============================================================================
def create_retrieval_tool(
    retriever: EnsembleRetriever,
) -> tool:
    """
    根據指定的檢索器實例創建一個可執行的檢索 Tool。

    **為什麼需要工廠函數：**
    LangChain Tool 在定義時需要是純函數。但在實際使用中，Tool 需要訪問
    一個特定的檢索器實例。因此我們使用工廠函數來創建一個閉包，將檢索器
    實例綁定到 Tool 函數中。

    Args:
        retriever: 已初始化的 EnsembleRetriever 實例

    Returns:
        一個經過 @tool 裝飾、可直接傳給 Agent 的 Tool 對象

    Example:
        ```python
        with open("vectorstore.pkl", "rb") as f:
            ensemble_retriever = pickle.load(f)
        
        retrieval_tool = create_retrieval_tool(ensemble_retriever)
        tools = [retrieval_tool]
        agent = create_react_agent(llm, tools)
        ```
    """

    @tool(args_schema=RetrievalInput)
    def retrieval_tool_impl(query: str) -> str:
        """
        RAG 課程檢索工具 - 從向量資料庫中搜尋相關課程和校務資訊。

        **使用時機：**
        - 當需要查詢特定課程的信息（名稱、時間、教師、教室）
        - 當需要查詢某個學院或系所的課程列表
        - 當需要搜尋校務相關的學術資訊
        - 當用戶提出關於課程內容、開課時間、教學團隊等問題時

        **預期回傳結果：**
        - 返回一個格式化的字符串，包含檢索到的相關課程和校務文檔
        - 每個檢索結果會列出文檔的名稱和相關內容
        - 如果找不到相關結果，會返回適當的提示信息

        Args:
            query: 用戶的查詢字符串

        Returns:
            格式化的檢索結果字符串，包含所有相關課程和校務資訊的摘要
        """
        try:
            # 使用提供的檢索器實例進行檢索
            documents: List[Document] = retriever.invoke(query)

            # 記錄檢索結果
            result_names: List[str] = [
                doc.metadata.get("name", "Unknown") for doc in documents
            ]
            print(f"[RAG Retrieval] Query: {query}")
            print(f"[RAG Retrieval] Found {len(documents)} documents: {result_names}")

            # 格式化輸出結果
            if not documents:
                return f"查詢 '{query}' 未找到相關課程或校務資訊。請嘗試使用不同的關鍵詞。"

            formatted_results: List[str] = []
            for idx, doc in enumerate(documents, 1):
                doc_name: str = doc.metadata.get("name", "Unknown Document")
                doc_content: str = doc.page_content

                formatted_results.append(
                    f"[結果 {idx}] {doc_name}\n{doc_content[:500]}..."
                    if len(doc_content) > 500
                    else f"[結果 {idx}] {doc_name}\n{doc_content}"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            error_msg: str = f"檢索過程中發生錯誤：{str(e)}"
            print(f"[RAG Retrieval Error] {error_msg}")
            return error_msg

    return retrieval_tool_impl

# ============================================================================
# 以下為修正後的測試區塊
# ============================================================================
if __name__ == "__main__":
    # 1. 因為你的 Tool 需要一個 retriever 才能運作，我們在測試時先做一個「假的」
    class MockRetriever:
        def invoke(self, query):
            # 模擬回傳一個 Document 物件
            return [Document(page_content="這是模擬的課程資訊內容", metadata={"name": "測試課程"})]

    # 2. 使用工廠函數產生真正的 Tool 實例
    mock_retriever = MockRetriever()
    my_working_tool = create_retrieval_tool(mock_retriever)

    print("="*30)
    print("🚀 Tool 化成功性檢查")
    print("="*30)

    try:
        print(f"【名稱】: {my_working_tool.name}")
        print(f"【描述】: {my_working_tool.description}")
        print(f"【參數格式】: {my_working_tool.args}")
        print("\n✅ 基本資訊檢查通過！")
    except Exception as e:
        print(f"\n❌ 基本資訊讀取失敗：{e}")

    print("\n" + "="*30)
    print("🛠️ 執行測試 (Invoke)")
    print("="*30)
    
    test_query = "測試查詢"
    try:
        # 執行我們產出的 tool
        response = my_working_tool.invoke({"query": test_query})
        
        print(f"【輸入測試資料】: {test_query}")
        print(f"【工具回傳結果】: \n{response}")
        print("\n✅ 工具執行成功！這代表你的 Vivobook 環境與 Tool 封裝完全沒問題！")
    except Exception as e:
        print(f"\n❌ 執行失敗，發生錯誤：{e}")