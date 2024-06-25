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
import pandas as pd

SYS_DATA_DIR  = "data"
SYS_MODEL_DIR = "modeling"
SYS_M_LLM_ID = "recursive"
SYS_M_CHUNK_SIZE = 256
SYS_M_OVERLAP_SIZE = 64

llmallm_path = "/home/matatov.n/projects/llmallm"
# if os.path.exists(llmallm_path) and llmallm_path not in sys.path:
#     sys.path.append(llmallm_path)
os.chdir(llmallm_path)

########################## Load document data

from llmallm.data_prep import get_document_data
files, documents = get_document_data()

file = 'Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf'

from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser import SentenceWindowNodeParser
from typing import List
import re

s_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
# slides_parser = SimpleNodeParser.from_defaults(
#     text_splitter=s_splitter,
#     include_prev_next_rel=True,
#     include_metadata=True
#     )
nodes = s_splitter.get_nodes_from_documents(documents[file])

figure_pattern = 'Figure \d{1,3}'
nodes_figures = [i for i in nodes if re.search(figure_pattern, i.to_dict()['text'])]
nodes_figures[5].to_dict()['text']

# from llmallm_modeling_utils import display_strings
# display_strings([nodes_figures[14].to_dict()['text']], 'Figure')

def get_figures_list(text):
    llm_figures_list_template = """
                    The text below contains mention for some figures.
                    Output figures and their numbers as Python list.
                    
                    EXAMPLES
                    -------- 
                    text : "Figure 2 and no other figures" , Answer : ["Figure 2"]
                    text : "Figure 2 and Figure 10 are present" , Answer : ["Figure 2", "Figure 10"]
                    test " "Figures are part of paper" , Answer : []
                    test " "Table 2 is present and Figure 2 also" , Answer : ["Figure 2"]
                    test " "Table 2 is present" , Answer : []
                    
                    TEXT
                    --------
                    {text}

                    ANSWER
                    --------
                    """
    llm_figures_list_prompt = llm_figures_list_template.replace("{text}", text)
    figures_list = llm.complete(llm_figures_list_prompt).text

    return figures_list

import json
from llama_index.llms import OpenAI
llm = OpenAI(model = 'gpt-4', temperature = 0.0)
# llm.additional_kwargs = {"top_k": 1}

for n in nodes_figures:
    figures_list = get_figures_list(n.to_dict()['text'])
    figures_str = " and ".join(json.loads(figures_list))
    print(figures_list)
    n.metadata = {'figures_content' : f"Contains information about {figures_str}"}

from llama_index.prompts.prompt_type import PromptType
from llama_index import Prompt

# Build vector index
from llama_index import StorageContext
from llama_index import VectorStoreIndex
from llama_index import load_index_from_storage

vector_index_save_dir=os.path.join(SYS_MODEL_DIR, "indexes/figures_index")

if not os.path.exists(vector_index_save_dir):
    vector_index = VectorStoreIndex(nodes_figures)
    vector_index.storage_context.persist(persist_dir=vector_index_save_dir)
else:
    vector_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=vector_index_save_dir),
    )

# Define query engines
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL, 
    DEFAULT_SIMPLE_INPUT_TMPL, 
    DEFAULT_REFINE_PROMPT_TMPL
)

from llama_index import Prompt

SYS_QA_TEMPLATE = Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)

vector_query_engine = vector_index.as_query_engine(
    text_qa_template = SYS_QA_TEMPLATE,
    similarity_top_k=10
)

vector_query_engine.query("What the purpose of Figure 3 ?").response

####################################################### create_figures_query_engine

from llama_index import Document

def create_figures_query_engine(document : Document, modeling_dir : str):
    
    import os
    import re
    from llama_index.text_splitter import SentenceSplitter

    s_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
    nodes = s_splitter.get_nodes_from_documents(document)

    figure_pattern = 'Figure \d{1,3}'
    nodes_figures = [i for i in nodes if re.search(figure_pattern, i.to_dict()['text'])]


    def get_figures_list(text):
        llm_figures_list_template = """
                        The text below contains mention for some figures.
                        Output figures and their numbers as Python list.
                        
                        EXAMPLES
                        -------- 
                        text : "Figure 2 and no other figures" , Answer : ["Figure 2"]
                        text : "Figure 2 and Figure 10 are present" , Answer : ["Figure 2", "Figure 10"]
                        test " "Figures are part of paper" , Answer : []
                        test " "Table 2 is present and Figure 2 also" , Answer : ["Figure 2"]
                        test " "Table 2 is present" , Answer : []
                        
                        TEXT
                        --------
                        {text}

                        ANSWER
                        --------
                        """
        llm_figures_list_prompt = llm_figures_list_template.replace("{text}", text)
        figures_list = llm.complete(llm_figures_list_prompt).text

        return figures_list
    
    import json
    from llama_index.llms import OpenAI
    llm = OpenAI(model = 'gpt-4', temperature = 0.0)
    # llm.additional_kwargs = {"top_k": 1}

    for n in nodes_figures:
        figures_list = get_figures_list(n.to_dict()['text'])
        figures_str = " and ".join(json.loads(figures_list))
        print(figures_list)
        n.metadata = {'figures_content' : f"Contains information about {figures_str}"}

    from llama_index.prompts.prompt_type import PromptType
    from llama_index import Prompt

    # Build vector index
    from llama_index import StorageContext
    from llama_index import VectorStoreIndex
    from llama_index import load_index_from_storage

    vector_index_save_dir=os.path.join(SYS_MODEL_DIR, "indexes/figures_index")

    if not os.path.exists(vector_index_save_dir):
        vector_index = VectorStoreIndex(nodes_figures)
        vector_index.storage_context.persist(persist_dir=vector_index_save_dir)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vector_index_save_dir),
        )

    # Define query engines
    from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
    from llama_index import Prompt

    SYS_QA_TEMPLATE = Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)

    vector_query_engine = vector_index.as_query_engine(
        text_qa_template = SYS_QA_TEMPLATE,
        similarity_top_k=10
    )

    return vector_query_engine

figures_engine = create_figures_query_engine(documents[file], SYS_MODEL_DIR)

figures_engine.query("What the purpose of Figure 9 ?").response