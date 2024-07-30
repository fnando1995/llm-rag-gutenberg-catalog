from langchain_community.chains import RAGChain
from langchain_community.document_loaders import SimpleWebLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import LanceDB
# import lancedb
# my_emb = OllamaEmbeddings(model="llama2")
# db = lancedb.connect("data/lance.db",)
# table = db.create_table("rag",data=[
#             {
#                 "vector": my_emb.embed_query("Hello World"),
#                 "text": "Hello World",
#                 "id": "1",
#             }
#         ],
#         mode="overwrite",)
# vectorstore = LanceDB.from_texts(texts = ["harrison worked at kensho"],embedding=my_emb,connection=db)
# retriever=vectorstore.as_retriever()
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
# # LLM
# ollama_llm = "llama2"
# model = ChatOllama(model=ollama_llm)
# print("Created Ollama model")
# def func(_dict):
#     return list(map(lambda x:x.page_content,_dict["context"]))
# # RAG chain
# chain = (
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#     | prompt
#     | model
#     | StrOutputParser()
# )


# from fastapi import FastAPI
# from fastapi.responses import RedirectResponse
# from langserve import add_routes
# app = FastAPI()
# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")
# add_routes(app, chain, path="/rag-lancedb-private")
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)