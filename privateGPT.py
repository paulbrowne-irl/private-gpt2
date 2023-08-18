#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import sys
import argparse
import logging
import time

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main(questions=[]):
    
    # Parse the command line arguments
    args = parse_arguments()

    # load embeddings db connection using this and environmental settings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        
        case "GPT4All":
            # currently this is the default model
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)

        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    #retrieve an handle to the interfacr from the choosen model, passing in link to our vector DB    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    
    # Non-Interactive questions and answers
    for query in questions:

        # print the question
        logging.info("\n\n> Question:"+query)

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        
        
        logging.info(f"\n> Answer (took {round(end - start, 2)} s.):")
        logging.info(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            logging.info("\n> " + document.metadata["source"] + ":")
            logging.info(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":

    #setup logging
    logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("privateGPT.log"),
        logging.StreamHandler(sys.stdout)
    ]
    )

    #setup questions we want to ask
    
    questions =[
                "write a 1 page letter asking a company to repay their grant (SEF or Sustaining Enterprise Fund)",#+
                "Will more people have jobs in 2024 after the pandemic is finished?", #+
                "A website to attract new graduates to work in your small company",
                "Which Life Sciences still be an attractive place to work after the covid pandemic?", #?
                "which products will be more in demand in 2022?", #OK
                "which Eurozone country (France, Germany, Belgium or Netherlands) is the most popular?",#?
                "Give me a summary marketing plan to an small Irish company looking to export for the first time", #+
                "write an email to the CEO setting out the challenges in the economy",
                "Summarize the business strategy of a small company in 3 sentences in a recession",
                "what challenges are facing the economy",
                "Should a company take out a bank loan or equity",
                "List all the typical costs for an engineering company for the next two years",
                "Write an email to a client explaining why a Venture Capital fund is now able to invest in their company",
                "write a business plan for a startup with a new AI Product for self driving cars looking to export to the US",
                "A poem, why being an accountant is the best job in the wold",
                "Describe how government policy should support companies making the green transition"
                ]

    #questions=[""]

    main(questions)
