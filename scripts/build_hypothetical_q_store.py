#Creating Hypothetical Questions and Storing in Chroma Vectorstore
#Generating Hypothetical Questions for Text Semantic Chunks
# Define prompt for generating hypothetical questions
from langchain_core.documents import Document
from langchain.schema import Document
from langchain.vectorstores import Chroma
import json
from parsers.llama_parser import tables
from core.config import llm, embedding_model
from scripts.semantic_chunks import vectorstore, semantic_chunks
hypothetical_questions_prompt_chunk = """Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:
{doc}
Generate only a list of questions. Do not mention anything before or after the list.
If the content cannot answer any questions - return empty list.
Ensure that the questions are specific to nutritional disorders, including their causes, symptoms, treatments, prevention, and classification."""

# Generate hypothetical questions for each semantic chunk and store them in a structured format.

# List to store documents with hypothetical questions
hyp_docs = []

# Generate hypothetical questions for each semantic chunk
for i, document in enumerate(semantic_chunks):
    try:
        # Invoke the LLM to generate questions based on the chunk content
        response = llm.invoke(hypothetical_questions_prompt_chunk.format(doc=document.page_content))
        questions = response.content  # Extract the generated questions
    except Exception as e:
        print(e)  # Print error message if any issue occurs
        questions = "[LLM Generation Failed]"  # Assign a default value if generation fails

    # Create metadata for the generated questions
    questions_metadata = {
        'original_content': document.page_content,  # Store the original chunk content
        'source': document.metadata['source'],  # Source document of the chunk
        'page': document.metadata['page'],  # Page number where the chunk appears
        'type': 'hypothetical_question'  # Indicate the content type
    }

    # Create and store the document containing generated questions
    hyp_docs.append(
        Document(
            id=str(i),  # Assign a unique ID to each generated document
            page_content=questions,  # Store the generated questions
            metadata=questions_metadata  # Attach metadata
        )
    )

# Function to print a sample document with hypothetical questions
def print_sample(docs, index=0):
    print("ID:\n", docs[index].id, "\n")
    print("Metadata:")
    print(json.dumps(docs[index].metadata, indent=4), "\n")
    print("Hypothetical Questions:\n", docs[index].page_content)

# Print a sample document
print(f"Total hyp_docs generated: {len(hyp_docs)}")
if len(hyp_docs) > 4:
    print_sample(hyp_docs, 4)
else:
    print("Less than 5 hyp_docs generated.")

#Generating Hypothetical Questions for Tables
# Define prompt for generating hypothetical questions for tables
hypothetical_questions_prompt_table = """Generate a list of exactly 3 hypothetical questions that the below nutritional disorder table could be used to answer:
{table}
Ensure that the questions are specific to nutritional disorders, dietary deficiencies, metabolic disorders, vitamin and mineral imbalances, obesity, and related health conditions.
Generate only a list of questions. Do not mention anything before or after the list.
If the content is a poor source for questions, return an empty list."""
# Generate hypothetical questions for tables in documents and store them as structured metadata.

# List to store documents with hypothetical questions for tables
hypotheticalq_tables = []  # Initialize an empty list to store generated questions

# Generate hypothetical questions for each table in the documents
for source in tables:  # Iterate over all processed documents
    for page_number in tables[source]:  # Iterate over pages in the document
        table_in_page = tables[source][page_number]  # Extract the table from the document

        try:
            # Generate questions using the LLM based on the table content
            response = llm.invoke(hypothetical_questions_prompt_table.format(table=table_in_page))
            questions = response.content
        except Exception as e:
            print(e)  # Print error if LLM invocation fails
            questions = "[LLM Question Generation Failed for Table]"  # Assign a placeholder value in case of an error

        # Metadata for each table
        questions_metadata = {
            'original_content': str(table_in_page),  # Store the content of the original table
            'source': source,  # Store the source document name or identifier
            'page': page_number,  # Store the page number where the table was found
            'type': 'table'  # Indicate that the content type is a table
        }

        # Create a Document object for each set of generated questions
        hypotheticalq_tables.append(
            Document(
                id="table_" + source.replace(" ", "_").replace(".pdf", "") + "_" + str(page_number),  # Generate a unique ID for each table
                page_content=questions,  # Store the generated questions
                metadata=questions_metadata  # Attach metadata for reference
            )
        )

# Function to print a sample document with hypothetical questions for tables
def print_sample(docs, index=0):
    if 0 <= index < len(docs):  # Check if index is within bounds
        print("ID:\n", docs[index].id, "\n")
        print("Metadata:")
        print(json.dumps(docs[index].metadata, indent=4), "\n")
        print("Hypothetical Questions:\n", docs[index].page_content)
    else:
        print(f"Index {index} is out of range for the list with length {len(docs)}.")

# Print a sample document, ensure index is within bounds
print_sample(hypotheticalq_tables, index=min(5, len(hypotheticalq_tables) - 1))

#Storing Hypothetical Chunks of documents and tables as batches in Chroma Vectorstore
# Assign unique IDs to hypothetical questions and store them in the Chroma vector database in batches.
import uuid
# Add IDs to the hypothetical questions for text semantic chunks
documents = [Document(id=f"hyptext_{i}_{uuid.uuid4().hex[:8]}", page_content=doc.page_content, metadata=doc.metadata)  # Create Document objects
             for i, doc in enumerate(hyp_docs)]  # Iterate over the semantic chunks and assign IDs

# Store the document chunks in Chroma vectorstore in batches

# First: initialize persistent Chroma store once
def get_text_vectorstore():
    return Chroma(
        collection_name="text_hypothetical_questions",
        embedding_function=embedding_model,
        persist_directory="./hyp_question_db/text"
    )

# Then: add documents in batches
batch_size = 100
text_store = get_text_vectorstore()
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    try:
        text_store.add_documents(batch)
        print(f"Stored text batch {i//batch_size + 1}")
    except Exception as e:
        print(f"Failed text batch {i//batch_size + 1}: {e}")
text_store.persist()

# Assign unique IDs to hypothetical questions for tables and store them in the Chroma vector database in batches.

# Add IDs to the hypothetical questions for tables
table_documents = [Document(id=f"tablehyp_{i}_{uuid.uuid4().hex[:6]}", page_content=table_chunk.page_content, metadata=table_chunk.metadata)  # Create Document objects
                   for i, table_chunk in enumerate(hypotheticalq_tables)]  # Iterate over table chunks and assign IDs

# Store the table chunks in Chroma vectorstore in batches
# Initialize once
def get_table_vectorstore():
    return Chroma(
        collection_name="table_hypothetical_questions",
        embedding_function=embedding_model,
        persist_directory="./hyp_question_db/table"
    )

# Then add batches
batch_size = 100
table_store = get_table_vectorstore()
for i in range(0, len(table_documents), batch_size):
    batch = table_documents[i: i + batch_size]
    try:
        table_store.add_documents(batch)
        print(f"Stored table batch {i//batch_size + 1}")
    except Exception as e:
        print(f"Failed table batch {i//batch_size + 1}: {e}")
# Persist once after all documents are added
table_store.persist()