# sample python code to inspect the documents and values in our vector db
# From  https://docs.trychroma.com/usage-guide
# pip install chromadb-client

import chromadb
from chromadb.config import Settings
#from chromadb import HttpClient
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

