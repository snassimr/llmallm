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
SYS_M_CHUNK_SIZE = 128
SYS_M_OVERLAP_SIZE = 16

llmallm_path = "/home/matatov.n/projects/llmallm"
# if os.path.exists(llmallm_path) and llmallm_path not in sys.path:
#     sys.path.append(llmallm_path)
os.chdir(llmallm_path)

########################## Load document data

from llama_index import download_loader
from llama_index import SimpleDirectoryReader

UnstructuredReader = download_loader('UnstructuredReader',)

files = os.listdir(SYS_DATA_DIR)
files = [f for f in files if f.endswith(".pdf")]
files = [f for f in files if f == 'Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf']
document_titles = [os.path.splitext(f)[0] for f in files]

start = time.time()
documents = {}

for file in files:
    if(not(file in documents)):
        dir_reader = SimpleDirectoryReader(input_files=[f"{SYS_DATA_DIR}/{file}"], 
                                            file_extractor={".pdf": UnstructuredReader(),
                                            })
        documents[file] = dir_reader.load_data()

print(f"Documents loaded : {len(documents)}")
print(f"Memory : {sys.getsizeof(documents)}")
print(f"Time : {time.time() - start}")

from llama_index.node_parser import (
    UnstructuredElementNodeParser,
)

node_parser = UnstructuredElementNodeParser()

raw_nodes_2021 = node_parser.get_nodes_from_documents(documents['Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf'])


base_nodes_2021, node_mappings_2021 = node_parser.get_base_nodes_and_mappings(
    raw_nodes_2021
)

from pprint import pprint
res = next(iter(node_mappings_2021.items()))
pprint("The first key-value pair of dictionary is : " + str(res))

from unstructured.partition.auto import partition

elements = partition(f"{SYS_DATA_DIR}/{file}")

print("\n\n".join([str(el) for el in elements[150:210]]))

print("\n\n".join([str(el.to_dict()) for el in elements[150:210]]))

len(elements)

figures_list = [str(el) for el in elements if str(el).startswith("Figure")]
tables_list = [str(el) for el in elements if str(el).startswith("Table")]

import re
import numpy as np

pattern = r'\b\d+(\.\d+)* [A-Z][a-zA-Z]*\b'
content_list1 = [str(el) for el in elements if re.match(pattern, str(el))]
chars_to_remove = ['.', ':']
# content_list2 = [str(el) for el in elements if el.to_dict()['type'] == 'Title']

element_types = np.unique([el.to_dict()['type'] for el in elements])

narrative_text_list = [str(el) for el in elements if el.to_dict()['type'] == 'NarrativeText']

lll = [el.to_dict()['type'] for el in elements if str(el) == '1 Introduction']
