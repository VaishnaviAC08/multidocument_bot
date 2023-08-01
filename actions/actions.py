import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Dict, List, Text
from rasa_sdk.events import UserUtteranceReverted

os.environ['OPENAI_API_KEY'] = "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"



class ActionQueryDatabase(Action):
    def name(self) -> str:
        return "action_query_database"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_query = tracker.latest_message.get('text')
        os.environ['OPENAI_API_KEY'] = "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"
        directory = "content"
        def load_docs(directory):
            loader = DirectoryLoader(directory)
            documents = loader.load()
            return documents
        documents=load_docs(directory)
        def split_docs(documents,chunk_size=1000,chunk_overlap=20):
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            docs=text_splitter.split_documents(documents)
            return docs
        docs = split_docs(documents)
        embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")
        pinecone.init(
        api_key="dda3cf0f-f9c6-4245-9ab7-5aa8917e43e2",
        environment="asia-southeast1-gcp-free")

        index_name="document-querying"
        index= Pinecone.from_existing_index(index_name, embeddings)
        def get_similiar_docs(query, k=2, score=False):
            if score:
                similar_docs = index.similarity_search_with_score(query, k=k)
            else:
                similar_docs = index.similarity_search(query, k=k)
            return similar_docs
        model_name = "gpt-3.5-turbo"
        llm= OpenAI(model_name=model_name)
        chain = load_qa_chain(llm, chain_type= "stuff")

        def get_answer(query):
            similar_docs=get_similiar_docs(query)
            print(similar_docs)
            print('============================================================================')
            answer = chain.run(input_documents=similar_docs, question = query)
            return answer
    
       
        
        response = get_answer(user_query)
        # response = agent_executor.run(user_query)

        # # Assuming the response from the database query is a single string
        dispatcher.utter_message(text=response)

        return [UserUtteranceReverted()]