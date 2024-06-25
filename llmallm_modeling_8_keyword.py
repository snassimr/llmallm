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

index = BaseKeywordTableIndex(nodes, index_struct, service_context, keyword_extract_template, max_keywords_per_chunk, use_async, show_progress)
retriever = index.as_retriever(retriever_mode=KeywordTableRetrieverMode.DEFAULT)
# retriever = index.as_retriever(retriever_mode=KeywordTableRetrieverMode.RAKE)

