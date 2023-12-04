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
SYS_M_OVERLAP_SIZE = 20

########################## Load document data

from llmallm.data_prep import prepare_document_data
prepare_document_data(load_mode = 2 , transform_mode = 2)


from llmallm.llmallm.modeling.load_llm_models import load_zephyr_7b_beta
llm = load_zephyr_7b_beta()
