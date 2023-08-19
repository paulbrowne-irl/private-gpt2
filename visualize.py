# visualise the document  network 

import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS
from chromaviz import visualize_collection

#get settings to connect to database
load_dotenv() 
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

#create the chroma db connection
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

# Start a webserver to visualise the document collection
# available at http://localhost:5000
visualize_collection(db._collection)