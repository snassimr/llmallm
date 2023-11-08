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

from llmallm.data_prep import get_document_data
files, documents = get_document_data()

########################## Build document_agents
# Define LLM service
import openai
from llama_index.llms import OpenAI
# from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext

openai.api_key = os.getenv("OPENAI_API_KEY")
# llm = OpenAI(model="gpt-3.5-turbo-16k", max_tokens = 512, temperature = 0.0)
llm = OpenAI(model="gpt-4", max_tokens = 512, temperature = 0.0)

from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex
from llama_index import  ListIndex
from llama_index import SummaryIndex
from llama_index.prompts.prompt_type import PromptType
from llama_index.tools import QueryEngineTool, ToolMetadata

embed_model = OpenAIEmbedding(model='text-embedding-ada-002',
                              embed_batch_size=10)

node_parser = SimpleNodeParser.from_defaults(chunk_size = SYS_M_CHUNK_SIZE,
                                             chunk_overlap = SYS_M_OVERLAP_SIZE,
                                             include_metadata = True)

service_context = ServiceContext.from_defaults(llm=llm,
                                               embed_model=embed_model,
                                               node_parser=node_parser)

vector_query_engines = {}
document_agents = {}

for file in files:

    # Build vector index
    vector_index = VectorStoreIndex.from_documents(documents[file], service_context=service_context)

    # Build list index
    summary_index = ListIndex.from_documents(documents[file], service_context=service_context)
    
    # Define query engines
    from llama_index.prompts.default_prompts import (
        DEFAULT_TEXT_QA_PROMPT_TMPL, 
        DEFAULT_SIMPLE_INPUT_TMPL, 
        DEFAULT_REFINE_PROMPT_TMPL
    )

    from llama_index import Prompt
    SYS_QA_TEMPLATE = Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)

    vector_query_engine = vector_index.as_query_engine(
        # text_qa_template = SYS_QA_TEMPLATE,
        similarity_top_k=3
    )
    
    summary_query_engine = summary_index.as_query_engine(
        # text_qa_template = SYS_QA_TEMPLATE,
        similarity_top_k=3
    )

    from llama_index import SimpleKeywordTableIndex
    nodes = service_context.node_parser.get_nodes_from_documents(documents[file])

    keyword_index = SimpleKeywordTableIndex(nodes)

    keyword_query_engine = keyword_index.as_query_engine(service_context=service_context)

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
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=f"Useful for summarization questions related to {file}",
            ), 
        ),
        QueryEngineTool(
            query_engine=keyword_query_engine,
            metadata=ToolMetadata(
                name="keyword_tool",
                description= f"Useful for retrieving information about named entities from {file}",
            ), 
        ),
    ]

    # # Build agent
    # from llama_index.agent import ContextRetrieverOpenAIAgent

    # llm = OpenAI(model = 'gpt-3.5-turbo-16k')

    # agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
    # tools = query_engine_tools,
    # retriever = vector_index.as_retriever(similarity_top_k=2),
    # llm = llm,
    # verbose=True, 
    # )
    
    # # Build agent
    # from llama_index.agent import OpenAIAgent

    # # llm = OpenAI(model = 'gpt-3.5-turbo-16k')
    # llm = OpenAI(model = 'gpt-4')

    # agent = OpenAIAgent.from_tools(
    #     query_engine_tools,
    #     llm=llm,
    #     verbose=True, 
    #     system_prompt="You must ALWAYS use at least one of the tools provided.",
    # )

    from llama_index.agent import ReActAgent

    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
        # system_prompt=f"""\
        #                 You are a specialized agent designed to answer questions about {file}.
        #                 Answer using on context is provided by one of tools.
        #                 Do NOT rely on prior knowledge.\
        #                 """,
    )

    vector_query_engines[file]   = vector_query_engine
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

response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    # text_qa_template=SYS_QA_TEMPLATE,
    )

# define query engine
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer, verbose = True,
    # text_qa_template=SYS_QA_TEMPLATE,
    
)

response = query_engine.query("List citations in the document")
print(str(response))

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
        "How Llama 2 compared to other open source models in Table 6",
        "How Llama 2 compared to other open source models in Table 3"
        ### Multi-document questions
        # "If RAG and LLM Fine tuning can be combined some way ?"
    ]
}

from llmallm_modeling_utils import prepare_output
qa_dataset_df, agent_history_df = prepare_output(files, questions_set, 
                                                 document_agents, 
                                                 query_engine)

from llmallm_modeling_utils import display_output
display_output(qa_dataset_df, ['file', 'question', 'answer'])

from llmallm_modeling_utils import display_output
display_output(agent_history_df, ['file', 'question', 'content'])

from llmallm_modeling_utils import prepare_extracts
extracts_df = prepare_extracts(files, questions_set, vector_query_engines)
display_output(extracts_df, ['file', 'question', 'text'])


file = 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf'
nodes = service_context.node_parser.get_nodes_from_documents(documents[file])

keyphrase = "Figure 2:"
selected_nodes    = []
selected_extracts = []

for i in nodes:
    node_content = i.get_content()
    if (keyphrase in node_content):
        selected_nodes.append(i)
        selected_extracts.append(node_content)

from llmallm_modeling_utils import display_strings
display_strings(selected_extracts)


# vector_index._get_node_with_embedding(selected_nodes)







#################################################################################
from dotenv import load_dotenv
# Load .env file
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentType

def get_random_response(q):
    import random
    response = random.choice(["How are you?", "I love you", "I'm very hungry"])
    return response

get_response = get_random_response

tools = [
    Tool(
        name="LlamaQueryEngine",
        func=lambda q: get_response(q),
        description="Useful for answering any question.",
        return_direct=True,
    ),
]

# set Logging to DEBUG for more detailed outputs


memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}

llm = ChatOpenAI(temperature=0)
agent_executor = initialize_agent(
    tools, llm, agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    memory=memory, 
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs=agent_kwargs, 
)

while True:
    text_input = input("User : ")
    print("User : " + text_input)
    if (text_input.lower() == 'stop'):
        break
    response= agent_executor.run(input=text_input)

    print(f'Agent: {response}')

