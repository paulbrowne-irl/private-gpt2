#!/usr/bin/env python3
import os
import glob
import logging
import sys
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    Docx2txtLoader

)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

# Load environment
load_dotenv()

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50
max_number_of_parts_per_run=5100 # adjust based on performance of laptop - 



# Custom document loaders
# as a class definition needs to be before the mapping file definition
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
     ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}



def load_single_document(file_path: str) -> List[Document]:
    '''Load a single document from a given file path using the predefined loader list'''

    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:

        #load the appropriate class and execute it's load method
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")





def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    # gather all files on this path as long as it is not on our list    
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    #Multithread loading
    with Pool(processes=os.cpu_count()) as pool:

        results = []

        #loop while updating progress bar
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:

            #load_single_document (defined above) is passed into this loop as a lambda function
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                
                #capture the loaded docs
                results.extend(docs)

                #update the progress bar
                pbar.update()

    return results




def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    logging.info(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)

    if not documents:
        logging.info("No new documents to load")
        exit(0)

    #process can fail on laptop if we try to ingest too many documents/chunks at a time
    logging.info(f"Found {len(documents)} new document parts from {source_directory}")

    if(len(documents)>max_number_of_parts_per_run):
         logging.info(f"truncating document list to max number of file parts per run: {max_number_of_parts_per_run}")
         documents = documents[0:max_number_of_parts_per_run]

    #Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")

    return texts




def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))

            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    #default        
    return False




def main():


    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):

        # Update and store locally vectorstore
        logging.info(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        if(len(texts)>0):
            logging.info(f"Creating embeddings. May take some minutes...")
            db.add_documents(texts)
        else:
            logging.info(f"No new documents embeddings found")
    else:
        # Create and store locally vectorstore
        logging.info("Creating new vectorstore")
        texts = process_documents()
        logging.info(f"Creating database and embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)

    # persist our vector store and release connnection    
    db.persist()
    db = None

    logging.info(f"Ingestion complete! You can now run privateGPT.py to query your documents")





if __name__ == "__main__":
    '''Running from command line, configuring logging and call main method'''

    logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("ingest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

    main()
