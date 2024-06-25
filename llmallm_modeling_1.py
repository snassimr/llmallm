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

SYS_DATA_DIR  = "app/data"
SYS_MODEL_DIR = "app/modeling"
SYS_EVAL_DIR = "app/evaluation"
SYS_M_LLM_ID = "recursive"
SYS_M_CHUNK_SIZE = 256
SYS_M_OVERLAP_SIZE = 32

llmallm_path = "/home/matatov.n/projects/llmallm"
# if os.path.exists(llmallm_path) and llmallm_path not in sys.path:
#     sys.path.append(llmallm_path)
os.chdir(llmallm_path)

########################## Load document data

from llmallm.data_prep import get_document_data
files, documents = get_document_data()

########################## Create Document Agents
### Define LLM model

## Open source llm
# from llama_index.llms import HuggingFaceLLM

## Open AI llm
import openai
from llama_index.llms import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
# llm = OpenAI(model="gpt-3.5-turbo-16k", max_tokens = 512, temperature = 0.0)
llm = OpenAI(model="gpt-4", max_tokens = 512, temperature = 0.0)

from llama_index import ServiceContext
from llama_index.prompts.prompt_type import PromptType
from llama_index import Prompt
from llama_index.tools import QueryEngineTool, ToolMetadata

## Define embedding model
# Open AI embeddings
from llama_index.embeddings import OpenAIEmbedding
embed_model = OpenAIEmbedding(model='text-embedding-ada-002',
                              embed_batch_size=10)

# ## Open source embeddings
# from llama_index.embeddings import HuggingFaceEmbedding
# embed_model = HuggingFaceEmbedding(model_name="modeling/llm_models/bge-large-en-v1.5", 
#                                    embed_batch_size = 8, device = "cuda")

from llama_index.node_parser import SimpleNodeParser
node_parser = SimpleNodeParser.from_defaults(chunk_size = SYS_M_CHUNK_SIZE,
                                             chunk_overlap = SYS_M_OVERLAP_SIZE,
                                             include_metadata = True)

service_context = ServiceContext.from_defaults(llm=llm,
                                               embed_model=embed_model,
                                               node_parser=node_parser)

vector_query_engines = {}
document_agents = {}

for file in files:

    nodes = service_context.node_parser.get_nodes_from_documents(documents[file])

    from llama_index import StorageContext
    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    from llama_index import VectorStoreIndex
    from llama_index import  ListIndex
    from llama_index import SummaryIndex
    from llama_index.indices.keyword_table import SimpleKeywordTableIndex

    # Build vector index
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

    # Build list index
    summary_index = ListIndex(nodes, storage_context=storage_context)

    # Build keyword index
    keyword_index = SimpleKeywordTableIndex(nodes, max_keywords_per_chunk = 10, storage_context=storage_context)
    
    # Define query engines
    from llama_index.prompts.default_prompts import (
        DEFAULT_TEXT_QA_PROMPT_TMPL, 
        DEFAULT_SIMPLE_INPUT_TMPL, 
        DEFAULT_REFINE_PROMPT_TMPL
    )

    SYS_QA_TEMPLATE = Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)

    vector_query_engine = vector_index.as_query_engine(
        text_qa_template = SYS_QA_TEMPLATE,
        similarity_top_k=3
    )

    from llmallm.modeling.engines import create_figures_query_engine

    figures_engine = create_figures_query_engine(documents[file], SYS_MODEL_DIR)

    summary_query_engine = summary_index.as_query_engine(
        # text_qa_template = SYS_QA_TEMPLATE,
        similarity_top_k=3
    )

    keyword_query_engine = keyword_index.as_query_engine(
        service_context=service_context
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
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="figures_tool",
                description=f"Useful for retrieving specific context related to figures",
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
                description= f"Useful for retrieving information about named entities, figures and tables from {file}",
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

    # # llm = OpenAI(model = 'gpt-3.5-turbo-16k', temperature = 0.0)
    llm = OpenAI(model = 'gpt-4', temperature = 0.1)
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
        f"Use this index if you need to answer questions about {file}.\n"
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

from llama_index.indices.postprocessor import SimilarityPostprocessor
response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    )

# define query engine
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer, verbose = True,
    # text_qa_template=SYS_QA_TEMPLATE,
    
)

import yaml
from llmallm_modeling_utils import display_output
with open(os.path.join(SYS_EVAL_DIR, "q_custom_data.yaml"), 'r') as q_data_file:
    q_dataset = yaml.safe_load(q_data_file)
q_dataset_df = pd.DataFrame(q_dataset)
display_output(q_dataset_df, [file])

from llmallm_modeling_utils import prepare_output
qa_dataset_df, agent_history_df = prepare_output(files, q_dataset, 
                                                 document_agents, 
                                                 query_engine)

from llmallm_modeling_utils import display_output
display_output(qa_dataset_df, ['file', 'question', 'answer'])

from llmallm_modeling_utils import display_output
display_output(agent_history_df, ['file', 'question', 'content'])

from llmallm_modeling_utils import prepare_extracts
from llmallm_modeling_utils import display_output
extracts_df = prepare_extracts(files, q_dataset, vector_query_engines)
display_output(extracts_df, ['file', 'question', 'text'])


file = 'Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf'
nodes = service_context.node_parser.get_nodes_from_documents(documents[file])
keyphrase = "Figure2"
selected_nodes    = []
selected_extracts = []

for i in nodes:
    node_content = i.get_content()
    if (keyphrase in node_content):
        selected_nodes.append(i)
        selected_extracts.append(node_content)

from llmallm_modeling_utils import display_strings
display_strings(selected_extracts, keyphrase)


vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)
vector_retriever.retrieve("What the purpose of Figure_2 ?")

# vector_index._get_node_with_embedding(selected_nodes)


