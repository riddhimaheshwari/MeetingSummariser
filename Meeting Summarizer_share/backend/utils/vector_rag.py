# backend/utils/vector_rag.py (UPDATED with LangChain Integration)

import os
import numpy as np  # Used by FAISS internally
from dotenv import load_dotenv  # For loading OPENAI_API_KEY from .env

# --- LangChain Core Imports ---
# OpenAIEmbeddings: Converts text into numerical vectors (embeddings) for semantic search.
from langchain_openai import OpenAIEmbeddings
# ChatOpenAI: LangChain's wrapper to interact with OpenAI's chat models (like GPT-4).
from langchain_openai import ChatOpenAI
# FAISS: LangChain's integration with the FAISS library for efficient vector storage and search.
from langchain_community.vectorstores import FAISS
# RecursiveCharacterTextSplitter: Helps break long texts into smaller, manageable chunks.
from langchain.text_splitter import RecursiveCharacterTextSplitter
# create_retrieval_chain: A high-level helper to combine retrieval (finding context) and generation (LLM answering).
from langchain.chains import create_retrieval_chain
# create_stuff_documents_chain: Helps format retrieved documents and pass them to the LLM.
from langchain.chains.combine_documents import create_stuff_documents_chain
# ChatPromptTemplate: For defining the structure of the prompt sent to the LLM.
from langchain_core.prompts import ChatPromptTemplate
# MessagesPlaceholder: A special part of ChatPromptTemplate to inject chat history.
from langchain_core.prompts import MessagesPlaceholder
# HumanMessage, AIMessage: Classes to represent user and AI messages in chat history.
from langchain_core.messages import HumanMessage, AIMessage
# ConversationBufferMemory: LangChain's class to store full chat history directly.
from langchain.memory import ConversationBufferMemory
from typing import Optional
load_dotenv()  # Load environment variables from your .env file


class MeetingRAG:
    # --- Constructor: Initializes the RAG system for a specific meeting ---
    def __init__(self, meeting_id: str):
        # 1. Initialize the Embedding Model
        # This is what generates semantic numerical representations (embeddings) of your text.
        # It needs your OPENAI_API_KEY. This is a crucial upgrade from TF-IDF.
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # This will hold the FAISS vector store, which is where your embedded document chunks live.
        # It's None initially and gets set in add_document.
        self.vectorstore = None

        # 2. Initialize the Large Language Model (LLM)
        # This is the GPT model that will actually generate answers.
        # LangChain's ChatOpenAI provides a convenient interface.
        self.llm = ChatOpenAI(
            model="gpt-4-0125-preview",  # The specific GPT-4 model version
            temperature=0.4,  # Controls creativity/randomness (lower for more factual answers)
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.meeting_id = meeting_id  # Store the unique ID of the meeting this RAG instance belongs to.

        # 3. Initialize Conversational Memory (for passing history to LLM)
        # This 'ConversationBufferMemory' will store ALL the chat messages (questions and answers)
        # for the current conversation within this specific MeetingRAG instance.
        # When 'ask_question' is called, the 'main.py' backend will populate this memory
        # with the chat history from its in-memory storage (or MongoDB later).
        # 'chat_memory=[]': Starts empty, ready to be populated.
        # 'return_messages=True': Ensures messages are stored and returned as LangChain's structured HumanMessage/AIMessage objects.
        # 'memory_key="chat_history"': This name *must* match the placeholder in your prompt template (see below).
        self.memory = ConversationBufferMemory(
            chat_memory=[],
            return_messages=True,
            memory_key="chat_history"
        )

        # 4. Initialize Text Splitter for Document Chunking
        # When you have a very long meeting transcript, you can't embed it as one giant piece.
        # LLMs also have context window limits. This tool breaks the transcript into smaller,
        # overlapping "chunks" (like pages in a book).
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len  # Use Python's built-in len() function to measure chunk size
        )

    # --- Helper Functions for Converting Chat History Formats ---
    # These functions are used by the FastAPI backend (main.py) to convert chat history
    # between the simple dictionary format it stores (or will store in MongoDB)
    # and LangChain's specific HumanMessage/AIMessage object format.

    def _convert_history_to_lc_messages(self, history_list: list) -> list:
        # Converts a list of {"question": "", "answer": ""} dictionaries into LangChain's message objects.
        lc_messages = []
        for entry in history_list:
            lc_messages.append(HumanMessage(content=entry["question"]))  # Create HumanMessage from user's question
            lc_messages.append(AIMessage(content=entry["answer"]))  # Create AIMessage from AI's answer
        return lc_messages

    def _convert_lc_messages_to_history(self, lc_messages: list) -> list:
        # Converts LangChain's message objects back into your simple list of dictionaries.
        # This is used before saving the chat history from LangChain's memory back to your backend's storage.
        history_list = []
        # Loops through the messages two at a time (assuming HumanMessage then AIMessage).
        for i in range(0, len(lc_messages), 2):
            if i + 1 < len(lc_messages):  # Ensures there's a corresponding AI message for each human message
                history_list.append({
                    "question": lc_messages[i].content,
                    "answer": lc_messages[i + 1].content
                })
        return history_list

    # --- Method to Add Document (Transcript) and Manage FAISS Index ---
    def add_document(self, text: str, faiss_path: Optional[str] = None):
        # This method is called to prepare the meeting transcript for RAG.
        # It can either load an existing FAISS index (if already saved) or create a new one.

        from langchain.docstore.document import Document  # Imported here as it's primarily used in this method

        if faiss_path and os.path.exists(faiss_path):
            # If a 'faiss_path' is provided (meaning a FAISS index was saved before for this meeting)
            # AND the file actually exists, then load that existing index.
            # 'allow_dangerous_deserialization=True' is a security flag sometimes needed for
            # newer LangChain versions when loading previously saved FAISS indexes.
            self.vectorstore = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded FAISS index from {faiss_path}")
        else:
            # If no 'faiss_path' is given, or the file doesn't exist, create a brand new FAISS index.
            documents = [Document(page_content=text)]  # Wrap the raw transcript text in a LangChain Document object.
            splits = self.text_splitter.split_documents(
                documents)  # Break the document into smaller, overlapping chunks.
            # Create the FAISS index by generating embeddings for each 'split' using 'self.embeddings'.
            # These embeddings are then added to the FAISS index for fast searching.
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            print("Created new FAISS index from document.")

    # --- Method to Ask a Question (Performs RAG) ---
    async def ask_question(self, question: str) -> str:  # This method is 'async' for non-blocking operations in FastAPI
        # Check if a vector store (FAISS index) has been loaded or created.
        if not self.vectorstore:
            raise ValueError("Meeting transcript not loaded. Call add_document first.")

        # 1. Retrieval Step: Find the most relevant context from the meeting transcript.
        # 'self.vectorstore.as_retriever()' converts the FAISS index into a search tool.
        # 'search_kwargs={"k": 1}' tells the retriever to find only the 'k=1' (single) most similar document chunk.
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1})

        # 2. Define the Prompt Template for the LLM.
        # This tells the LLM how to format its input, including the system role, chat history, context, and current question.
        # 'MessagesPlaceholder("chat_history")': This is a special part. It's where LangChain will automatically
        # inject the full conversational history from 'self.memory' into the prompt.
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful meeting assistant. Use the provided context to answer the question concisely."),
            MessagesPlaceholder("chat_history"),  # THIS IS WHERE ALL THE PAST MESSAGES GO
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
        ])

        # 3. Create the Document Combination Chain.
        # This chain takes the document chunks retrieved by the 'retriever' and "stuffs" (inserts) them into the 'prompt' template.
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # 4. Create the Full Retrieval Chain.
        # This is the main RAG chain that orchestrates the entire flow:
        # Retriever (finds context) -> Document Chain (prepares prompt with context + history) -> LLM (generates answer).
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # 5. Invoke the Chain (Generate the Answer).
        # 'await retrieval_chain.ainvoke': We use 'ainvoke' because 'ask_question' is an async function,
        # ensuring non-blocking behavior for FastAPI.
        # This makes the actual single API call to OpenAI.
        # 'input': The current user question.
        # 'chat_history': The entire list of messages from 'self.memory' is loaded here.
        # LangChain then constructs ONE single prompt with all these parts and sends it to OpenAI.
        response = await retrieval_chain.ainvoke({
            "input": question,
            "chat_history": self.memory.load_memory_variables({})["chat_history"]
        })

        # This method returns the AI's answer. The saving of this answer and the question
        # into the overall meeting's chat history (in_memory_meetings or MongoDB)
        # is handled by the 'main.py' FastAPI endpoint, not here.
        return response["answer"]