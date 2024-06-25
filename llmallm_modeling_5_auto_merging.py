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
SYS_MODEL_DIR = "modeling"
SYS_M_LLM_ID = "auto_merging"

########################## Load document data

from llmallm.data_prep import get_document_data
files, documents = get_document_data()

file = 'Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf'

from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents[file]]))

########################## Build engine
import os

from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes
from llama_index import StorageContext, load_index_from_storage
from llama_index.retrievers import AutoMergingRetriever
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.query_engine import RetrieverQueryEngine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="auto_merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [256, 128, 64]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

from llama_index.llms import OpenAI

index = build_automerging_index(
    [document],
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    save_dir="./auto_merging_index",
)

query_engine = get_automerging_query_engine(index, similarity_top_k=6)