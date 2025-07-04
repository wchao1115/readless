'''
This script loads the plain text of the One Big Beautiful Bill, splits it into chunks,
creates embeddings using OpenAI, and stores them in a Chroma vectorstore.
'''
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv() # Load environment variables from .env file

# Configuration
SOURCE_TEXT_PATH = os.path.join("books", "one_big_beautiful_bill.txt")
CHROMA_DB_PATH = "chroma_db_bill_text"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Ensure your OPENAI_API_KEY is set as an environment variable

# Chunking parameters (can be adjusted)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def process_and_store_text():
    '''Loads text, splits it, creates embeddings, and stores them in Chroma.'''
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to proceed.")
        print("Example: export OPENAI_API_KEY='your_api_key_here' (Linux/macOS)")
        print("         $Env:OPENAI_API_KEY='your_api_key_here' (Windows PowerShell)")
        return

    if not os.path.exists(SOURCE_TEXT_PATH):
        print(f"Error: Source text file not found at {SOURCE_TEXT_PATH}")
        print("Please ensure 'fetch_doc.py' has been run successfully.")
        return

    try:
        # 1. Load the text document
        print(f"Loading text from {SOURCE_TEXT_PATH}...")
        loader = TextLoader(SOURCE_TEXT_PATH, encoding='utf-8')
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")

        # 2. Split the text into chunks
        print(f"Splitting text into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        if not chunks:
            print("No chunks were created. Please check the source file and chunking parameters.")
            return

        # 3. Create OpenAI embeddings
        print("Initializing OpenAI embeddings model...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        print("OpenAI embeddings model initialized.")

        # 4. Create and persist the Chroma vectorstore
        print(f"Creating/loading Chroma vectorstore at {CHROMA_DB_PATH}...")
        # If the directory already exists, Chroma will try to load it.
        # If you want to ensure a fresh store, you might want to delete the CHROMA_DB_PATH directory first.
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print(f"Vectorstore created/updated and persisted at {CHROMA_DB_PATH}.")
        print("Processing complete.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_and_store_text()
