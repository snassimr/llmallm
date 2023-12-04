import os
import sys
import time

SYS_DATA_DIR = "data"


class DataPrep:

    files = []
    documents = []
    options = {'load_mode' : -1, 'transform_mode': -1}


def load_document_data_1():

    from llama_index import SimpleDirectoryReader

    files = os.listdir(SYS_DATA_DIR)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf']
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

    DataPrep.files,  DataPrep.documents = files, documents


            
def load_document_data_2():

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

    DataPrep.files,  DataPrep.documents = files, documents

def load_document_data_3():

    from llama_hub.file.pymu_pdf.base import PyMuPDFReader

    files = os.listdir(SYS_DATA_DIR)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf']
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

    DataPrep.files,  DataPrep.documents = files, documents

def load_document_data_4():

    from llmsherpa.readers import LayoutPDFReader
    from llama_index import Document

    files = os.listdir(SYS_DATA_DIR)
    files = [f for f in files if f.endswith(".pdf")]
    files = [f for f in files if f == 'Llama_2_Open_Foundation_and_Fine-Tuned_Chat_Models.pdf']
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

    DataPrep.files,  DataPrep.documents = files, documents

def transform_document(doc , transform_mode = 1):
    
    def transform_text(text: str) -> str:
        """
        """
        import re

        # Treating figures tokens
        # text = re.sub(r'Figure (\d+)', r'#Figure \1#', text)
        # text = re.sub(r'Figure\s+(\d+)', r'Figure\1', text)
        text = re.sub(r'Figure\s+(\d+)', r'<Figure\1>', text)

        return text

    if(transform_mode == 1) : pass
    if(transform_mode == 2) : doc.text = transform_text(doc.text)

    return doc

def prepare_document_data(load_mode = 1 , transform_mode = 1):
    
    if(not(DataPrep.options['load_mode']==load_mode)):
        if(load_mode == 1) : load_document_data_1()
        if(load_mode == 2) : load_document_data_2()
        if(load_mode == 3) : load_document_data_3()
        if(load_mode == 4) : load_document_data_4()
        DataPrep.options['load_mode'] = load_mode

    if(not(DataPrep.options['load_mode']==load_mode) or not(DataPrep.options['transform_mode']==transform_mode)):
        for file in DataPrep.files:
            for i, doc in enumerate(DataPrep.documents[file]):
                DataPrep.documents[file][i] = transform_document(
                    doc , 
                    transform_mode = transform_mode)

    import pickle
    filepath = os.path.join(SYS_DATA_DIR, 'data.pkl')
    documents_data = {'files' : DataPrep.files, 'documents' : DataPrep.documents}
    with open(filepath, 'wb') as f:
        pickle.dump(documents_data, f)

def get_document_data():

    import pickle
    filepath = os.path.join(SYS_DATA_DIR, 'data.pkl')
    with open(filepath, 'rb') as f:
        documents_data = pickle.load(f)

    files = documents_data['files']
    documents = documents_data['documents']

    return files, documents

