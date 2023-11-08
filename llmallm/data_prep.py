import os
import sys
import time

SYS_DATA_DIR = "data"

def create_document_data_1():

    from llama_index import SimpleDirectoryReader

    files = os.listdir(SYS_DATA_DIR)
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

def create_document_data_3():

    from llama_hub.file.pymu_pdf.base import PyMuPDFReader
    from llama_index import SimpleDirectoryReader

    files = os.listdir(SYS_DATA_DIR)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']
    document_titles = [os.path.splitext(f)[0] for f in files]

    start = time.time()
    documents = {}

    for file in files:
        if(not(file in documents)):
            loader = PyMuPDFReader()
            documents[file] = loader.load(file_path=f"{SYS_DATA_DIR}/{file}")

    print(f"Documents loaded : {len(documents)}")
    print(f"Memory : {sys.getsizeof(documents)}")
    print(f"Time : {time.time() - start}")

    import pickle
    filepath = os.path.join(SYS_DATA_DIR, 'data.pkl')
    stored_document_data = {'files' : files, 'documents' : documents}
    with open(filepath, 'wb') as f:
        pickle.dump(stored_document_data, f)

def create_document_data_4():

    from llmsherpa.readers import LayoutPDFReader
    from llama_index import Document

    files = os.listdir(SYS_DATA_DIR)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama 2 - Open Foundation and Fine-Tuned Chat Models.pdf']
    document_titles = [os.path.splitext(f)[0] for f in files]

    start = time.time()
    documents = {}

    for file in files:
        if(not(file in documents)):
            llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
            pdf_reader = LayoutPDFReader(llmsherpa_api_url)
            llmsherpa_docs = pdf_reader.read_pdf(f"{SYS_DATA_DIR}/{file}")
            llambda_docs = []
            for doc in llmsherpa_docs.chunks():
                llambda_docs.append(Document(text=doc.to_context_text(), extra_info={}))
            documents[file] = llambda_docs

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

