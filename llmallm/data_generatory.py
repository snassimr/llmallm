
from llama_index.evaluation import DatasetGenerator

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
SYS_DG_LLM_ID = "openai" # camel-5b-hf

########################## Load external data

from llmallm.data_load import load_external_data_2
files, documents = load_external_data_2()

from langchain.embeddings import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext

from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import Prompt

import openai
from llama_index.llms import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model = "gpt-3.5-turbo-16k",
             max_tokens=512, 
             temperature=0.1)

# from llama_index.prompts import PromptTemplate
# from llama_index.llms import HuggingFaceLLM
# import torch

# query_wrapper_prompt = PromptTemplate(
#     "Below is an instruction that describes a task. "
#     "Write a response that appropriately completes the request.\n\n"
#     "### Instruction:\n{query_str}\n\n### Response:"
# )

# llm = HuggingFaceLLM(
#     context_window=2048,
#     max_new_tokens=512,
#     generate_kwargs={"temperature": 0.25, "do_sample": False},
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="Writer/camel-5b-hf",
#     model_name="Writer/camel-5b-hf",
#     device_map="auto",
#     tokenizer_kwargs={"max_length": 2048},
#     # uncomment this if using CUDA to reduce memory usage
#     model_kwargs={"torch_dtype": torch.float16}
# )

node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)

service_context = ServiceContext.from_defaults(llm=llm,
                                               node_parser=node_parser)
question_data = {}
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


    questions  = data_generator.generate_questions_from_nodes(num=50)
    questions  = [f"{question.strip()}\n" for question in questions]
    print(f"Generated {len(questions)} questions for {file}.")

    question_data[file] = questions

import pickle
filepath = os.path.join(SYS_MODEL_DIR, 'questions' + '_' + SYS_DG_LLM_ID + '.pkl')
with open(filepath, 'wb') as f:
    pickle.dump(question_data, f)