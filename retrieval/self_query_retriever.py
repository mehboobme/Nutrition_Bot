from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.schema import AttributeInfo

def get_self_query_retriever(
    llm: BaseLanguageModel,
    vectorstore: VectorStore,
    description: str,
    metadata: list[AttributeInfo]
) -> SelfQueryRetriever:
    return SelfQueryRetriever.from_llm(llm, vectorstore, description, metadata)
