'''
This script processes documents and uploads them to Azure Cognitive Search with vector embeddings.
It replaces the original Chroma-based process_doc.py for Azure integration.
'''
import os
import json
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SearchField
)
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import uuid
import time

# Load environment variables
load_dotenv()

# Azure Configuration
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "bill-text-index")

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# Document configuration
TEXT_FILE_PATH = "books/one_big_beautiful_bill.txt"  # Bill text file path
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def create_search_index():
    """Create the Azure Cognitive Search index with vector search capabilities."""
    
    # Create the search index client
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )
    
    # Define the index schema
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # Azure OpenAI ada-002 embedding dimension
            vector_search_profile_name="default"
        )
    ]
    
    # Configure vector search
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="default",
                algorithm_configuration_name="default"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="default"
            )
        ]
    )
    
    # Create the index
    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )
    
    try:
        result = index_client.create_or_update_index(index)
        print(f"Index '{AZURE_SEARCH_INDEX_NAME}' created successfully.")
        return True
    except Exception as e:
        print(f"Error creating index: {e}")
        return False

def process_and_upload_documents():
    """Process the bill text and upload to Azure Cognitive Search."""
    
    # Check if the text file exists
    if not os.path.exists(TEXT_FILE_PATH):
        print(f"Error: Text file not found at {TEXT_FILE_PATH}")
        print("Please ensure the bill text file exists at the specified path.")
        return False
    
    # Initialize Azure OpenAI embeddings
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        print("Azure OpenAI embeddings initialized successfully.")
    except Exception as e:
        print(f"Error initializing Azure OpenAI embeddings: {e}")
        return False
    
    # Load the document
    try:
        loader = TextLoader(TEXT_FILE_PATH, encoding='utf-8')
        documents = loader.load()
        print(f"Loaded document with {len(documents)} pages.")
    except Exception as e:
        print(f"Error loading document: {e}")
        return False
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")
    
    # Create the search client
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )
    
    # Process chunks in batches
    batch_size = 10
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        documents_to_upload = []
        
        print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
        
        for j, chunk in enumerate(batch):
            try:
                # Generate embeddings for the chunk
                content = chunk.page_content
                embedding = embeddings.embed_query(content)
                
                # Create document for upload
                doc = {
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "title": f"Bill Text Chunk {i + j + 1}",
                    "source": TEXT_FILE_PATH,
                    "content_vector": embedding
                }
                
                documents_to_upload.append(doc)
                
            except Exception as e:
                print(f"Error processing chunk {i + j + 1}: {e}")
                continue
        
        # Upload the batch
        if documents_to_upload:
            try:
                result = search_client.upload_documents(documents_to_upload)
                successful_uploads = sum(1 for r in result if r.succeeded)
                print(f"Successfully uploaded {successful_uploads}/{len(documents_to_upload)} documents in this batch.")
            except Exception as e:
                print(f"Error uploading batch: {e}")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    print(f"Document processing completed. Total chunks processed: {total_chunks}")
    return True

def main():
    """Main function to orchestrate the document processing."""
    
    print("Starting Azure Cognitive Search document processing...")
    
    # Verify environment variables
    required_vars = [
        "AZURE_SEARCH_SERVICE_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    # Create the search index
    print("Creating Azure Cognitive Search index...")
    if not create_search_index():
        print("Failed to create search index. Exiting.")
        return False
    
    # Process and upload documents
    print("Processing and uploading documents...")
    if not process_and_upload_documents():
        print("Failed to process and upload documents. Exiting.")
        return False
    
    print("Azure Cognitive Search setup completed successfully!")
    print(f"Index '{AZURE_SEARCH_INDEX_NAME}' is ready for use with the chatbot.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
