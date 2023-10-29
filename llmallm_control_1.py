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

########################## Load external data

from llmallm.data_load import load_external_data_2
files, documents = load_external_data_2()

########################## Build document_agents
# Define LLM service
import openai
from llama_index.llms import OpenAI
# from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext

openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-3.5-turbo-16k", max_tokens = 512, temperature = 0.0)

from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex
from llama_index import  ListIndex
from llama_index import SummaryIndex
from llama_index.tools import QueryEngineTool, ToolMetadata

embed_model = OpenAIEmbedding(model='text-embedding-ada-002',
                              embed_batch_size=10)

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20,
                                             include_metadata = True)

service_context = ServiceContext.from_defaults(llm=llm,
                                               embed_model=embed_model,
                                               node_parser=node_parser)

document_agents = {}

for file in files:

    # Build vector index
    vector_index = VectorStoreIndex.from_documents(documents[file], service_context=service_context)
    
    # Build list index
    list_index = ListIndex.from_documents(documents[file], service_context=service_context)
    
    # Define query engines
    from llama_index.prompts.default_prompts import (
        DEFAULT_TEXT_QA_PROMPT_TMPL, 
        DEFAULT_SIMPLE_INPUT_TMPL, 
        DEFAULT_REFINE_PROMPT_TMPL
    )

    from llama_index import Prompt
    QA_TEMPLATE = Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL)

    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=3
    )
    
    list_query_engine = list_index.as_query_engine(
        similarity_top_k=3
    )

    # Define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=f"Useful for retrieving specific context related to {file}",
            ), 
        ),
        QueryEngineTool(
            query_engine=list_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=f"Useful for summarization questions related to {file}",
            ), 
        ),
    ]

    # Build agent
    from llama_index.agent import ContextRetrieverOpenAIAgent

    llm = OpenAI(model = 'gpt-3.5-turbo-16k')

    # QA_PROMPT_TMPL = (
    # """
    # "Context information is below.\n"
    # "---------------------\n"
    # "{context_str}\n"
    # "---------------------\n"
    # "Given the context information and not prior knowledge, "
    # "answer the question. If the answer is not in the context, inform "
    # "the user that you can't answer the question - DO NOT MAKE UP AN ANSWER.\n"
    # "Question: {query_str}"
    # """)

    # from llama_index.prompts import PromptTemplate
    # QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

    # agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
    # tools = query_engine_tools,
    # retriever = vector_index.as_retriever(similarity_top_k=2),
    # llm = llm, qa_prompt=QA_PROMPT,
    # verbose=True, 
    # )
    
    # Build agent
    from llama_index.agent import OpenAIAgent

    function_llm = OpenAI(model = 'gpt-3.5-turbo-16k')
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True, 
    )

    document_agents[file] = agent

########################## Define index nodes to link to the document agents
from llama_index.schema import IndexNode
from llama_index.retrievers import RecursiveRetriever
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

nodes = []
for file in files:
    doc_summary = (
        f"This content contains details about {file}. "
        f"Use this index if you need to lookup specific facts about {file}.\n"
        "Do not use this index if you want to query multiple documents."
    )
    node = IndexNode(text=doc_summary, index_id=file,)
    nodes.append(node)

# Define top-level retriever
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=document_agents,
    verbose=True,
)

response_synthesizer = get_response_synthesizer(response_mode="compact")

# define query engine
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer, verbose = True
)

# ### out of context question (overlap with common domain)
# question = "How to prepare pizza ?"
### global summarization question
# question = " Summarize Llama 2 - Open Foundation and Fine-Tuned Chat Models in 500 words"
### keyword question
# question = "Find citations of Ethan Perez in References"
question = "If RAG and LLM Fine tuning can be combined some way ?"
# ### generated_questions
# question = "What is the purpose of Red Teaming?"
### content question
# question = "Does paper contain LLama2 comparision to other algorithms ?"

response = query_engine.query(question)
import json
response_text_short   = response.response
response_text_long = response.source_nodes[0].text.split("Response:", 1)[-1].strip()
print(response_text_short)
print(response_text_long)


from llmallm.utils import display_agent_history
display_agent_history(agent)

vector_tool_response = vector_query_engine.query(question)

from llmallm.utils import display_extracts
extracts_df = display_extracts(vector_tool_response)


filename = 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf'
document = documents[filename]


list_tool_response = list_query_engine.query(question)

list_tool_sources  = list_tool_response.source_nodes
list_tool_files = [i.metadata['file_name'] for i in list_tool_sources]
list_tool_pages = [i.metadata['page_label'] for i in list_tool_sources]
list_tool_texts = [i.node.get_content() for i in list_tool_sources]
list_tool_scores = [i.score for i in list_tool_sources]



# 79386bd3-a8d7-43f2-b4dd-79961673f70d

import asyncio
import nest_asyncio
nest_asyncio.apply()



from llama_index.response.notebook_utils import display_source_node

for n in response.source_nodes:
    display_source_node(n, source_length=500)

