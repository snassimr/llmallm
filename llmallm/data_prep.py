import os
import sys
import time

SYS_DATA_DIR = "data"

def create_document_data_1():

    from llama_index import SimpleDirectoryReader

    files = os.listdir(SYS_DATA_DIR)
    files = os.listdir(files_folder)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']
    document_titles = [os.path.splitext(f)[0] for f in files]

    start = time.time()
    documents = {}

    for file in files:
        if(not(file in documents)):
            documents[file] = SimpleDirectoryReader(
                input_files=[f"{SYS_DATA_DIR}/{file}"]).load_data()
            
    print(f"Documents loaded : {len(documents)}")
    print(f"Memory : {sys.getsizeof(documents)}")
    print(f"Time : {time.time() - start}")

    import pickle
    filepath = os.path.join(SYS_DATA_DIR, 'data.pkl')
    stored_document_data = {'files' : files, 'documents' : documents}
    with open(filepath, 'wb') as f:
        pickle.dump(stored_document_data, f)
            
def create_document_data_2():

    from llama_index import download_loader
    from llama_index import SimpleDirectoryReader

    UnstructuredReader = download_loader('UnstructuredReader',)

    files = os.listdir(SYS_DATA_DIR)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']
    document_titles = [os.path.splitext(f)[0] for f in files]

    start = time.time()
    documents = {}

    for file in files:
        if(not(file in documents)):
            dir_reader = SimpleDirectoryReader(input_files=[f"{SYS_DATA_DIR}/{file}"], file_extractor={
                        ".pdf": UnstructuredReader(),
                        })
            documents[file] = dir_reader.load_data()

    print(f"Documents loaded : {len(documents)}")
    print(f"Memory : {sys.getsizeof(documents)}")
    print(f"Time : {time.time() - start}")

    import pickle
    filepath = os.path.join(SYS_DATA_DIR, 'data.pkl')
    stored_document_data = {'files' : files, 'documents' : documents}
    with open(filepath, 'wb') as f:
        pickle.dump(stored_document_data, f)

def get_document_data():
    
    import pickle
    filepath = os.path.join(SYS_DATA_DIR, 'data.pkl')
    with open(filepath, 'rb') as f:
        stored_document_data = pickle.load(f)

    files = stored_document_data['files']
    documents = stored_document_data['documents']

    return files, documents

