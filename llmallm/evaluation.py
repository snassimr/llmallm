
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
import pandas as pd

SYS_MODEL_DIR = "model"
SYS_DG_LLM_ID = "openai"
SYS_M_LLM_ID = "recursive"
SYS_EVAL_LLM_ID = "openai" # camel-5b-hf

########################## Load question
from langchain.embeddings import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext

import openai
from llama_index.llms import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model = "gpt-3.5-turbo-16k",
             temperature=0.0)

from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
import torch

import pickle
filepath = os.path.join(SYS_MODEL_DIR, 'questions' + '_' + SYS_DG_LLM_ID + '.pkl')
with open(filepath, 'rb') as f:
    question_data = pickle.load(f)

service_context = ServiceContext.from_defaults(llm=llm)

import time
import asyncio
import nest_asyncio
nest_asyncio.apply()

def generate_responses(query_engine, questions):
    async def run_query(query_engine, q):
        try:
            return await query_engine.aquery(q)
        except:
            return None

    responses = []
    for batch_size in range(0, len(questions), 5):
        batch_qs = questions[batch_size:batch_size+5]

        tasks = [run_query(query_engine, q) for q in batch_qs]
        batch_responses = asyncio.run(asyncio.gather(*tasks))
        print(f"finished batch {(batch_size // 5) + 1} out of {len(questions) // 5}")
        responses.append(batch_responses)

start = time.time()
responses = dict()
for f in files:
    questions = question_data[file]
    file_responses = []
    for q in questions :
        r = query_engine.query(q)
        file_responses.append(r)
    responses[f] = file_responses

print(f"Time : {time.time() - start}")

start = time.time()
responses = dict()
for file in files:
    questions = question_data[file]
    file_responses = generate_responses(query_engine, questions)
    responses[f] = file_responses

print(f"Time : {time.time() - start}")




def evaluate_query_engine(evaluator, query_engine, questions):
    async def run_query(query_engine, q):
        try:
            return await query_engine.aquery(q)
        except:
            return Response(response="Error, query failed.")

    total_correct = 0
    all_results = []
    for batch_size in range(0, len(questions), 5):
        batch_qs = questions[batch_size:batch_size+5]

        tasks = [run_query(query_engine, q) for q in batch_qs]
        responses = asyncio.run(asyncio.gather(*tasks))
        print(f"finished batch {(batch_size // 5) + 1} out of {len(questions) // 5}")

        # eval for hallucination
        if isinstance(evaluator, ResponseEvaluator):
          for response in responses:
              eval_result = 1 if "YES" in evaluator.evaluate(response) else 0
              total_correct += eval_result
              all_results.append(eval_result)
        # eval for answer quality
        elif isinstance(evaluator, QueryResponseEvaluator):
          for question, response in zip(batch_qs, responses):
              eval_result = 1 if "YES" in evaluator.evaluate(question, response) else 0
              total_correct += eval_result
              all_results.append(eval_result)

        # helps avoid rate limits
        time.sleep(1)

    return total_correct, all_results


for file in files:
    questions = question_data[file]
    evaluator = ResponseEvaluator(service_context=service_context)
    total_correct, all_results = evaluate_query_engine(evaluator, query_engine, questions)
    print(f"ResponseEvaluator : {total_correct} correct out of {len(question_dataset)}.")


questions = question_data['Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']

from llama_index.evaluation import ResponseEvaluator
evaluator = ResponseEvaluator(service_context = service_context)
response = query_engine.query(questions[0])
eval_result = evaluator.evaluate(response)
type(evaluator)

from llama_index.evaluation import QueryResponseEvaluator
evaluator = QueryResponseEvaluator(service_context=service_context)
response = query_engine.query(questions[0])
eval_result = evaluator.evaluate_source_nodes(response)

# from llama_index.evaluation import (
#     DatasetGenerator, 
#     RelevancyEvaluator, 
#     ResponseEvaluator, 
#     FaithfulnessEvaluator, 
#     QueryResponseEvaluator)

# # relevancy_evaluator = RelevancyEvaluator(service_context=service_context)
# # relevancy_eval_result = relevancy_evaluator.evaluate_response(question = question, 
# #                                                               response = response)

# # faithfulness_evaluator = FaithfulnessEvaluator(service_context=service_context)
# # faithfulness_eval_result = faithfulness_evaluator.evaluate_response(response=response)



    

