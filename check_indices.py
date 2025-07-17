'''
Script to check Azure Cognitive Search indices and their document counts.
'''
import os
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Configuration
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "bill-text-index")

def check_indices():
    """Check all indices in the Azure Cognitive Search service."""
    
    # Create the search index client
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )
    
    try:
        # List all indices
        indices = index_client.list_indexes()
        
        print("Azure Cognitive Search Indices:")
        print("=" * 50)
        
        for index in indices:
            print(f"Index Name: {index.name}")
            print(f"Fields: {len(index.fields)}")
            
            # Get document count for this index
            search_client = SearchClient(
                endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
                index_name=index.name,
                credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
            )
            
            try:
                # Get document count
                result = search_client.search("*", include_total_count=True)
                doc_count = result.get_count()
                print(f"Document Count: {doc_count}")
            except Exception as e:
                print(f"Error getting document count: {e}")
            
            print("-" * 30)
        
    except Exception as e:
        print(f"Error listing indices: {e}")
        return False
    
    return True

def check_specific_index():
    """Check the specific index used by the chatbot."""
    
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )
    
    try:
        # Search for all documents
        result = search_client.search("*", include_total_count=True, top=5)
        doc_count = result.get_count()
        
        print(f"\nSpecific Index: {AZURE_SEARCH_INDEX_NAME}")
        print("=" * 50)
        print(f"Total Documents: {doc_count}")
        
        print("\nSample Documents:")
        print("-" * 30)
        
        for i, doc in enumerate(result):
            if i >= 3:  # Show only first 3 documents
                break
            print(f"ID: {doc.get('id', 'N/A')}")
            print(f"Title: {doc.get('title', 'N/A')}")
            print(f"Content Preview: {doc.get('content', 'N/A')[:100]}...")
            print(f"Source: {doc.get('source', 'N/A')}")
            print("-" * 30)
        
    except Exception as e:
        print(f"Error checking specific index: {e}")
        return False
    
    return True

def main():
    """Main function."""
    
    print("Checking Azure Cognitive Search indices...")
    
    # Verify environment variables
    if not AZURE_SEARCH_SERVICE_ENDPOINT or not AZURE_SEARCH_API_KEY:
        print("Error: Missing Azure Search credentials in environment variables.")
        return False
    
    # Check all indices
    if not check_indices():
        return False
    
    # Check specific index
    if not check_specific_index():
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
