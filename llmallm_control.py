from dotenv import load_dotenv

# Load .env file
load_dotenv()

import os
import sys
import time

########################## Load external data

from llama_index import SimpleDirectoryReader

files_folder = "documents"
files = os.listdir(files_folder)
files = [f for f in files if f.endswith(".pdf")]
document_titles = [os.path.splitext(f)[0] for f in files]

start = time.time()
if (not('documents' in locals())):
    documents = {}

for file in files:
    if(not(file in documents)):
        documents[file] = SimpleDirectoryReader(
            input_files=[f"{files_folder}/{file}"]).load_data()

print(f"Documents loaded : {len(documents)}")
print(f"Memory : {sys.getsizeof(documents)}")
print(f"Time : {time.time() - start}")


########################## Build document_agents
# Define LLM service
import openai
from llama_index.llms import OpenAI
# from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext

openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex
from llama_index import SummaryIndex
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20,
                                             include_metadata = True)

service_context = ServiceContext.from_defaults(llm=llm,
                                               node_parser=node_parser)

document_agents = {}

for file in files:

    # Build vector index
    vector_index = VectorStoreIndex.from_documents(documents[file], service_context=service_context)
    
    # Build list index
    list_index = SummaryIndex.from_documents(documents[file], service_context=service_context)
    
    # Define query engines
    from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer
    from llama_index.indices.postprocessor import SimilarityPostprocessor

    vector_query_engine = vector_index.as_query_engine(
        node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5),
                             SimilarityPostprocessor(similarity_cutoff=0.7)])
    list_query_engine = list_index.as_query_engine(
        node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5),
                             SimilarityPostprocessor(similarity_cutoff=0.7)])

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
    function_llm = OpenAI(model="gpt-3.5-turbo-0613")
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
    node = IndexNode(text=doc_summary, index_id=file)
    nodes.append(node)

# define retriever
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# define recursive retriever
# note: can pass `document_agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=document_agents,
    verbose=True,
)

response_synthesizer = get_response_synthesizer(response_mode="compact",)

# define query engine
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context, 
    
)

question = "If RAG and LLM Fine tuning can be combined some way ?"
question = "How to prepare pizza ?"
response = query_engine.query(question)

import asyncio
import nest_asyncio
nest_asyncio.apply()

from llama_index.evaluation import (
    DatasetGenerator, 
    RelevancyEvaluator, 
    ResponseEvaluator, 
    FaithfulnessEvaluator, 
    QueryResponseEvaluator)

# relevancy_evaluator = RelevancyEvaluator(service_context=service_context)
# relevancy_eval_result = relevancy_evaluator.evaluate_response(question = question, 
#                                                               response = response)

faithfulness_evaluator = FaithfulnessEvaluator(service_context=service_context)
faithfulness_eval_result = faithfulness_evaluator.evaluate_response(response=response)

response_evaluator = ResponseEvaluator(service_context=service_context)
response_eval_result = response_evaluator.evaluate_response(response=response)


print(response)

nodes_with_score = query_engine.retrieve(question)

for node_with_score in nodes_with_score :
    node = node_with_score.node
    print(f"Document ID: {node.node_id}")
    print(f"Passage: {node.text}")
    print(f"Relevance Score: {node_with_score.score}")

nodes

from llmallm.llmallm_utils import display_eval_df
a = display_eval_df(question, response, eval_result)

from llama_index.response.notebook_utils import display_source_node

for n in response.source_nodes:
    display_source_node(n, source_length=1500)

response_text_nodes = [node_with_score.id_ for node_with_score in response.source_nodes]

text_nodes = dict()

for file in documents:
    file_text_nodes = node_parser.get_nodes_from_documents(documents = documents[file])
    text_nodes[file] = file_text_nodes

target_id = response.source_nodes[0].node.id_
for k,v in text_nodes.items():
    file = k
    response_in_text_nodes = [obj for obj in v if obj.id_ == target_id]
    if len(response_in_text_nodes) > 0:
        print(file)

# documents['Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf'][0]


# response = query_engine.query("Could you give step by step guidance ?")
# print(response)

# response = query_engine.query("What is Harden Runner in DevOps self-service-centric pipeline security and guardrails?")
# print(response)

for i in text_nodes['Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']:
    substring = "On the other hand, LLM fine-tuning involves training a language model on specific data"
    if (substring in i.text):
        print(i)

question = "How to prepare pizza ?"