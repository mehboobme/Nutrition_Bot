import os
import nest_asyncio
from llama_parse import LlamaParse

nest_asyncio.apply()

# Step 1: Function to parse all PDFs
def parse_pdf_folder(folder_path: str, llamaparse_api_key: str):
    parser = LlamaParse(
        result_type="markdown",
        skip_diagonal_text=True,
        fast_mode=False,
        num_workers=9,
        check_interval=10,
        api_key=llamaparse_api_key
    )

    json_objs = []

    for pdf in os.listdir(folder_path):
        if pdf.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, pdf)
            json_objs.extend(parser.get_json_result(pdf_path))
    
    return json_objs

# Step 2: Function to extract tables
def extract_tables(json_objs):
    page_texts, tables = {}, {}

    for obj in json_objs:
        json_list = obj['pages']
        name = obj["file_path"].split("/")[-1]
        page_texts[name] = {}
        tables[name] = {}

        for json_item in json_list:
            for component in json_item['items']:
                if component['type'] == 'table':
                    tables[name][json_item['page']] = component['rows']

    return tables

# Step 3: Parse and export tables for import elsewhere
llamaparse_api_key = os.getenv("LLAMA_API_KEY")
folder_path = "C:\\NLP\\GL\\Advance_RAG_Project\\data\\unzipped_docs\\Nutritional Medical Reference"
  # Update this with actual folder path

json_objs = parse_pdf_folder(folder_path, llamaparse_api_key)
tables = extract_tables(json_objs)  # <-- This is now accessible via import
