
from llama_index.evaluation import DatasetGenerator

# %reload_ext autoreload
# %autoreload 2

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

from dotenv import load_dotenv

# Load .env file
load_dotenv()

import os
import sys
import time

SYS_MODEL_DIR = "model"
SYS_DG_LLM_ID = "zephyr-7b-beta" # openai , zephyr-7b-beta
SYS_DG_N      = 20

########################## Load document data

from llmallm.data_prep import get_document_data
files, documents = get_document_data()

from langchain.embeddings import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext

from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import Prompt

# import openai
# from llama_index.llms import OpenAI

# openai.api_key = os.getenv("OPENAI_API_KEY")
# llm = OpenAI(model = "gpt-3.5-turbo-16k",
#              max_tokens=512, 
#              temperature=0.1)

from llmallm.modeling.load_llm import load_zephyr_7b_beta
from llmallm.modeling.load_llm import free_gpu_memory

llm = load_zephyr_7b_beta()

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)

service_context = ServiceContext.from_defaults(llm=llm, node_parser=node_parser)

def generate_q_data():

    q_data = {}
    for file in files:
        data_generator = DatasetGenerator.from_documents(
                            documents[file],
                            text_question_template=Prompt(
                            "A sample from the documents is below.\n"
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n"
                            "Using the documentation sample, carefully follow the instructions below:\n"
                            "{query_str}"
                            ),
                            question_gen_query=(
                                "You are a search pipeline evaluator. Using the papers provided, "
                                "you must create a list of summary questions and question/answer questions. "
                                "Limit the queries to the information supplied in the context.\n"
                                "Question: "
                            ),
                            service_context=service_context)


        q_dataset  = data_generator.generate_questions_from_nodes(num = SYS_DG_N)
        q_dataset  = [f"{q.strip()}" for q in q_dataset]
        q_data[file] = q_dataset

        import pickle
        filepath = os.path.join(SYS_MODEL_DIR, 'q_data' + '_' + SYS_DG_LLM_ID + '.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(q_data, f)

generate_q_data()

import asyncio
import nest_asyncio
nest_asyncio.apply()

# async def generate_qa_data():
    
#     qa_data = {}

#     for file in files:
#         nodes = node_parser.get_nodes_from_documents(documents[file])
#         dataset_generator = DatasetGenerator(
#                                                 nodes,
#                                                 service_context=service_context,
#                                                 show_progress=True,
#                                                 num_questions_per_chunk=10,
#                                             )

#         qa_dataset = await dataset_generator.agenerate_dataset_from_nodes(num = SYS_DG_N)

#         # qa_dataset = dataset_generator.agenerate_dataset_from_nodes()
#         qa_data[file] = qa_dataset.qr_pairs
    
#     import pickle
#     filepath = os.path.join(SYS_MODEL_DIR, 'qa' + '_' + SYS_DG_LLM_ID + '.pkl')
#     with open(filepath, 'wb') as f:
#         pickle.dump(qa_data, f)


# asyncio.run(generate_qa_data())

def generate_qa_data():
    
    qa_data = {}

    for file in files:
        nodes = node_parser.get_nodes_from_documents(documents[file])
        dataset_generator = DatasetGenerator(
                                                nodes[:1],
                                                service_context=service_context,
                                                show_progress=True,
                                                num_questions_per_chunk=2,
                                            )

        qa_dataset = dataset_generator.generate_dataset_from_nodes(num = 1)
        # qa_dataset = dataset_generator.agenerate_dataset_from_nodes()
        qa_data[file] = qa_dataset.qr_pairs
    
    import pickle
    filepath = os.path.join(SYS_MODEL_DIR, 'qa_data' + '_' + SYS_DG_LLM_ID + '.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(qa_data, f)


generate_qa_data()
    
        

# free_gpu_memory('llm')

