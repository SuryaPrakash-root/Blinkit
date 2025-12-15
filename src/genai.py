import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from data_processing import get_feedback_data
from dotenv import load_dotenv

class RAGChatbot:
    def __init__(self):
        print("Initializing RAG Chatbot...")
        load_dotenv()
        self.vector_store = None
        self.chain = None
        
        # Check for API Key
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            print("Warning: MISTRAL_API_KEY not found in .env. Chatbot will run in fallback mode (Retrieval Only).")
        
        # Load Data
        self.feedback_df = get_feedback_data()
        if self.feedback_df is not None and not self.feedback_df.empty:
            self._build_vector_store()
            if self.api_key:
                self._setup_chain()
        else:
            print("Error: No feedback data found.")

    def _build_vector_store(self):
        print("Building Vector Store (this may take a moment)...")
        # Use local embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Prepare text chunks
        texts = self.feedback_df.apply(lambda x: f"Category: {x['feedback_category']}. Sentiment: {x['sentiment']}. Feedback: {x['feedback_text']}", axis=1).tolist()
        
        # Create Vector Store
        self.vector_store = FAISS.from_texts(texts, embeddings)
        print("Vector Store Ready.")

    def _setup_chain(self):
        # LLM
        llm = ChatMistralAI(api_key=self.api_key, model="mistral-large-latest")
        
        # Prompt
        template = """
        You are a Customer Experience Analyst for Blinkit. 
        Use the following pieces of customer feedback to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Summarize the key issues or patterns found in the feedback.
        
        {context}
        
        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question):
        if not self.vector_store:
            return "System Error: Knowledge base not initialized."
            
        # Retrieval only fallback
        if not self.chain:
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n".join([d.page_content for d in docs])
            return f"**Analysis (LLM unavailable, showing raw matches):**\n\nRelevant Feedback:\n{context}\n\n*(To get AI summaries, please add MISTRAL_API_KEY to .env)*"
            
        try:
            return self.chain.invoke(question)
        except Exception as e:
            return f"Error generation response: {e}"

if __name__ == "__main__":
    bot = RAGChatbot()
    print(bot.ask("What are the main delivery complaints?"))
