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

# from llmallm.data_prep import get_document_data
# files, documents = get_document_data()

from pathlib import Path
from llama_index import download_loader
PDFReader = download_loader("PDFReader")
loader = PDFReader()
file = 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf'
pdf_url = f"{SYS_DATA_DIR}/{file}"
docs0 = loader.load_data(file=Path(f"{SYS_DATA_DIR}/{file}"))

from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser

bullet_splitter = SentenceSplitter(paragraph_separator=r"\n●|\n-|\n", chunk_size=250)
slides_parser = SimpleNodeParser.from_defaults(
    text_splitter=bullet_splitter,
    include_prev_next_rel=True,
    include_metadata=True
    )
slides_nodes = slides_parser.get_nodes_from_documents(docs0)

from llama_index.node_parser import SentenceWindowNodeParser
from typing import List
import re
def custom_sentence_splitter(text: str) -> List[str]:
    return re.split(r'\n●|\n-|\n', text)
bullet_node_parser = SentenceWindowNodeParser.from_defaults(
    sentence_splitter=custom_sentence_splitter,
    window_size=3,
    include_prev_next_rel=True,
    include_metadata=True
    )

from llama_index.schema import IndexNode

sub_node_parsers =[bullet_node_parser]
all_nodes = []
for base_node in slides_nodes:
    for parser in sub_node_parsers:
        sub_nodes = parser.get_nodes_from_documents([base_node])
        sub_inodes = [
            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
        ]
        all_nodes.extend(sub_inodes)
    # also add original node to node
    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
    all_nodes.append(original_node)
all_nodes_dict = {n.node_id: n for n in all_nodes}


########################## Build query engine
# Define LLM service
from llama_index import ServiceContext

import openai
from llama_index.llms import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-3.5-turbo-16k", max_tokens = 512, temperature = 0.0)
# llm = OpenAI(model="gpt-4", max_tokens = 512, temperature = 0.0)


# from llmallm.modeling.load_llm import load_zephyr_7b_beta
# from llmallm.modeling.load_llm import free_gpu_memory
# llm = load_zephyr_7b_beta()

from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding(model='text-embedding-ada-002',
                              embed_batch_size=10)

service_context = ServiceContext.from_defaults(llm=llm,
                                               embed_model=embed_model)

from llama_index import VectorStoreIndex

vector_index_chunk = VectorStoreIndex(
    all_nodes, service_context=service_context
)

vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=2)

from llama_index.retrievers import RecursiveRetriever
retriever_chunk = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever_chunk},
    node_dict=all_nodes_dict,
    verbose=True,
)

from llama_index.query_engine import RetrieverQueryEngine
query_engine_chunk = RetrieverQueryEngine.from_args(
    retriever_chunk,
    service_context=service_context,
    verbose=True,
    response_mode="compact"
)


response = query_engine_chunk.query("What is the purpose of Figure 6?")
from llama_index.response.notebook_utils import display_response
display_response(response)

response = query_engine_chunk.query("How to prepare pizza ?")
from llama_index.response.notebook_utils import display_response
display_response(response)

response = query_engine_chunk.query("List citations where Ethan Perez is one of authors")
from llama_index.response.notebook_utils import display_response
display_response(response)

response = query_engine_chunk.query("If RAG and LLM Fine tuning can be combined some way ?")
from llama_index.response.notebook_utils import display_response
display_response(response)


questions_set = {
    'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf':
    [
        ### Out of context question (overlap with common domain)
        "How to prepare pizza ?",
        ### Keyword around question
        "What is the purpose of Figure 2?",
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

