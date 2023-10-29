import os
import sys
import time

def load_external_data_1():

    from llama_index import SimpleDirectoryReader

    files_folder = "documents"
    files = os.listdir(files_folder)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']
    document_titles = [os.path.splitext(f)[0] for f in files]

    start = time.time()
    documents = {}

    for file in files:
        if(not(file in documents)):
            documents[file] = SimpleDirectoryReader(
                input_files=[f"{files_folder}/{file}"]).load_data()
            
    print(f"Documents loaded : {len(documents)}")
    print(f"Memory : {sys.getsizeof(documents)}")
    print(f"Time : {time.time() - start}")

    return files, documents
            
def load_external_data_2():

    from llama_index import download_loader
    from llama_index import SimpleDirectoryReader

    UnstructuredReader = download_loader('UnstructuredReader',)

    files_folder = "documents"
    files = os.listdir(files_folder)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']
    document_titles = [os.path.splitext(f)[0] for f in files]

    start = time.time()
    documents = {}

    for file in files:
        if(not(file in documents)):
            dir_reader = SimpleDirectoryReader(input_files=[f"{files_folder}/{file}"], file_extractor={
                        ".pdf": UnstructuredReader(),
                        })
            documents[file] = dir_reader.load_data()

    print(f"Documents loaded : {len(documents)}")
    print(f"Memory : {sys.getsizeof(documents)}")
    print(f"Time : {time.time() - start}")

    return files, documents

