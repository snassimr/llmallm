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

SYS_MODEL_DIR = "model"
SYS_DG_LLM_ID = "zephyr-7b-beta" # zephyr-7b-beta

########################## Load external data

from llmallm.data_load import load_external_data_2
files, all_docs = load_external_data_2()

from langchain.embeddings import HuggingFaceBgeEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    Response,
    StorageContext,
    load_index_from_storage,
    SummaryIndex
)

from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.prompts import Prompt
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.schema import IndexNode
from llama_index.agent import OpenAIAgent

from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

model_name = "BAAI/bge-base-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages in research papers"
)

import openai
from llama_index.llms import OpenAI

node_parser = SimpleNodeParser.from_defaults(chunk_size=4096, chunk_overlap=512)

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-16k", max_tokens=512, temperature=0.1),
                                               embed_model=model_norm,
                                               node_parser=node_parser)

agents = {}

for doc in all_docs:
  # build vector index
  vector_index = VectorStoreIndex.from_documents(all_docs[doc], service_context=service_context)
  # build summary index
  summary_index = SummaryIndex.from_documents(all_docs[doc], service_context=service_context)

  # define query engine
  vector_query_engine = vector_index.as_query_engine()
  summary_query_engine = summary_index.as_query_engine()

  # define tools
  query_engine_tools = [
    QueryEngineTool(
        query_engine = vector_query_engine,
        metadata = ToolMetadata(
            name="vector_tool",
            description=f"Useful for retrieving specific context from {doc} "
        )
    ),
    QueryEngineTool(
        query_engine = summary_query_engine,
        metadata = ToolMetadata(
            name="summary_tool",
            description=f"Useful for summarization questions related to {doc} "
        )
    ),
  ]

  # build agent
  function_llm = OpenAI(model = 'gpt-3.5-turbo-16k')
  agent = OpenAIAgent.from_tools(
      query_engine_tools,
      llm=function_llm,
      verbose=True
  )
  agents[doc] = agent


  # define top-level nodes
nodes = []
for doc in all_docs:
  doc_summary = (
      f"This content contains content about {doc}. "
      f"Use this index if you need to lookup specific facts about {doc}.\n"
  )
  node = IndexNode(text=doc_summary, index_id=doc)
  nodes.append(node)

# define top-level retriever
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)
     

# note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=agents,
    verbose=True,
)
     

response_synthesizer = get_response_synthesizer(
    # service_context=service_context,
    response_mode="compact",
)
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)

# question = "If RAG and LLM Fine tuning can be combined some way ?"
# question = "How to prepare pizza ?"
question = "What papers in References are related to Ethan Perez ?"
# question = " Summarize Llama 2 - Open Foundation and Fine-Tuned Chat Models in 500 words"

response = query_engine.query(question)

vector_tool_response = vector_query_engine.query(question)

vector_tool_sources  = vector_tool_response.source_nodes
# vector_tool_files = [i.metadata['file_name'] for i in vector_tool_sources]
# vector_tool_pages = [i.metadata['page_label'] for i in vector_tool_sources]
vector_tool_texts = [i.node.get_content() for i in vector_tool_sources]
vector_tool_scores = [i.score for i in vector_tool_sources]

vector_tool_data = dict()
for i in range(len(vector_tool_sources)):
    extract_data = {
                #   'file_name': vector_tool_files[i-1], 
                #   'page_label': vector_tool_pages[i-1],
                  'text': vector_tool_texts[i-1],
                  'score': vector_tool_scores[i-1],
                 }
    
    vector_tool_data[i] = extract_data

###########################################################################################

doc_indexes = {}

for doc in all_docs:
  # build vector index
  vector_index = VectorStoreIndex.from_documents(all_docs[doc], service_context=service_context)
  doc_indexes[doc] = vector_index

## testing index
index = doc_indexes['Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']
query_engine = index.as_query_engine()
question = "Summarize information is related to Ethan Perez in References section"
reponse = query_engine.query(question)
reponse.response



##############################################################################################

all_docs