from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from core.config import llm, embedding_model  # Ensure llm and embedding_model are defined here
from scripts.build_hypothetical_q_store import get_text_vectorstore, get_table_vectorstore

# ----------------------------------------
# Metadata Fields for SelfQueryRetriever
# ----------------------------------------
metadata_field_info = [
    AttributeInfo(
        name="Category",
        description="Category of the nutritional disorder (e.g., Undernutrition, Vitamin Deficiency, Obesity, etc.)",
        type="string"
    ),
    AttributeInfo(
        name="DisorderType",
        description="Specific type of nutritional disorder (e.g., Protein-Energy Malnutrition, Scurvy, Rickets, etc.)",
        type="string"
    ),
    AttributeInfo(
        name="Page",
        description="Page within the nutritional medical reference document that the information belongs to",
        type="integer"
    )
]

# ----------------------------------------
# Document Description
# ----------------------------------------
document_content_description = (
    "This document contains medical reference information related to nutritional disorders, "
    "including definitions, causes, symptoms, treatments, and categorizations such as undernutrition, "
    "vitamin and mineral deficiencies, and obesity. Each section corresponds to a specific disorder type, "
    "and includes page-level annotations."
)

# ----------------------------------------
# MultiRetriever Class
# ----------------------------------------
class MultiRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        all_results = []
        for retriever in self.retrievers:
            results = retriever.invoke(query)
            all_results.extend(results)
        return all_results

# ----------------------------------------
# Load Vectorstores
# ----------------------------------------
text_vectorstore = get_text_vectorstore()
table_vectorstore = get_table_vectorstore()

# ----------------------------------------
# Create SelfQueryRetrievers
# ----------------------------------------
text_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=text_vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)

table_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=table_vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)

# ----------------------------------------
# Combine Into MultiRetriever
# ----------------------------------------
multi_retriever = MultiRetriever([text_retriever, table_retriever])
