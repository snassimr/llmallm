%reload_ext autoreload
%autoreload 2

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from dotenv import load_dotenv

# Load .env file
load_dotenv()

import os
import sys
import time
import pandas as pd

SYS_DATA_DIR  = "data"
SYS_MODEL_DIR = "model"
SYS_M_LLM_ID = "recursive"
SYS_M_CHUNK_SIZE = 1024
SYS_M_OVERLAP_SIZE = 64

########################## Load document data

from llmsherpa.readers import LayoutPDFReader

file = 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf'
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_url = f"{SYS_DATA_DIR}/{file}"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)
doc = pdf_reader.read_pdf(pdf_url)

from llama_index.readers.schema.base import Document
from llama_index import VectorStoreIndex

index = VectorStoreIndex([])
for chunk in doc.chunks():
    index.insert(Document(text=chunk.to_context_text(), extra_info={}))
query_engine = index.as_query_engine()

response = query_engine.query("How to prepare pizza ?")
response = query_engine.query("What is the title of Figure 2 ?")

from IPython.display import display, HTML

for section in doc.sections():
    print(section.title)

section_title = "Llama 1"
selected_section = None
for section in doc.sections():
    if section.title == section_title:
        selected_section = section
        break

from llama_index.llms import OpenAI
context = selected_section.to_html(include_children=True, recurse=True)
print(context)

selected_section = "Llama 1"
from llama_index.llms import OpenAI
context = selected_section.to_html(include_children=True, recurse=True)
question = "list all the tasks discussed and one line about each task"
resp = OpenAI().complete(f"read this text and answer question: {question}:\n{context}")
print(resp.text)


for table in doc.tables():
    HTML(table.to_html())

doc.tables()



########################## Build query engine
# Define LLM service
from llama_index import ServiceContext

import openai
from llama_index.llms import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-3.5-turbo-16k", max_tokens = 512, temperature = 0.0)

# from llmallm.modeling.load_llm import load_zephyr_7b_beta
# from llmallm.modeling.load_llm import free_gpu_memory
# llm = load_zephyr_7b_beta()

from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex

embed_model = OpenAIEmbedding(model='text-embedding-ada-002',
                              embed_batch_size=10)

node_parser = SimpleNodeParser.from_defaults(chunk_size = SYS_M_CHUNK_SIZE,
                                             chunk_overlap = SYS_M_OVERLAP_SIZE,
                                             include_metadata = True)

service_context = ServiceContext.from_defaults(llm=llm,
                                               embed_model=embed_model,
                                               node_parser=node_parser)

from llama_index import VectorStoreIndex

file = 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf'
vector_index = VectorStoreIndex.from_documents(documents[file],
                                               service_context=service_context)

query_engine = vector_index.as_query_engine(response_mode="compact")

response = query_engine.query("How do OpenAI and Meta differ on AI tools?")
from llama_index.response.notebook_utils import display_response
display_response(response)

# response = query_engine.query("Summarize text in 200 words ?")
# from llama_index.response.notebook_utils import display_response
# display_response(response)

questions_set = {
    'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf':
    [
        ### Out of context question (overlap with common domain)
        "How to prepare pizza ?",
        ### Keyword around question
        "What is the purpose of Figure 2 ?",
        "List citations where Ethan Perez is one of authors",
        ### Global summarization question
        # "Summarize Llama 2 - Open Foundation and Fine-Tuned Chat Models in 500 words",
        ### Typical research paper question
        # "What is the purpose of Red Teaming?",
        # "Prepare table of content ?",
        # "Does paper contain LLama2 comparision to other algorithms ?",
        ### Multi-document questions
        "If RAG and LLM Fine tuning can be combined some way ?"
    ]
}

from llmallm_modeling_utils import prepare_output
qa_dataset_df = prepare_output(files, questions_set, 
                                                 query_engine)

from llmallm_modeling_utils import display_output
display_output(qa_dataset_df, ['file', 'question', 'answer'])

from llmallm_modeling_utils import prepare_extracts
vector_query_engines = {}
vector_query_engines[file] = query_engine
extracts_df = prepare_extracts(files, questions_set, vector_query_engines)

display_output(extracts_df, ['file', 'question', 'text'])