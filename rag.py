from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob
from langchain_core.documents.base import Document
from langchain_core.prompts.chat import PromptTemplate

class Loader():
    def __init__(self,persist_dir):
        self.Documents = []
        self.fulldir = persist_dir
    
    def load(self):
        list_of_paths = glob.glob(os.path.join(self.fulldir,'*','*.txt'))
        for ebook_path in list_of_paths:
            with open(ebook_path,encoding='utf-8') as f:
                text = f.read()
                self.Documents.append(Document(page_content=text))
        return self.Documents


class RAGT5():
    def __init__(self, model_name, db_directory, db_version, prompt_template, load_db_from_disk=False):
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name)
        self.model  = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.persist_dir    = os.path.join(db_directory,db_version)
        self.prompt_template    = prompt_template
        self.load_db_from_disk  = load_db_from_disk
        self.load_vector_dabase()

    def load_vector_dabase(self):
        print("loading vector database")
        if self.load_db_from_disk:
            self.vector_db = Chroma(persist_directory=self.persist_dir,
                                    embedding_function=HuggingFaceEmbeddings(model_name=self.model_name))
        else:
            docs = Loader(self.persist_dir).load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            self.vector_db = Chroma.from_documents(persist_directory=self.persist_dir,
                                                documents=splits, 
                                                embedding=HuggingFaceEmbeddings(model_name=self.model_name))

    def query_rag(self,query):
        results = self.vector_db.similarity_search_with_score(query, k=5)
        context_text = "\n--\n".join([doc.page_content for doc, _score in results])
        prompt = self.prompt_template.format(context=context_text, question=query)
        inputs_ids = self.tokenizer(prompt, return_tensors='pt',max_length=512, truncation=True, padding="max_length").input_ids
        response = self.model.generate(inputs_ids,max_new_tokens=150)
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
        return response_text 
