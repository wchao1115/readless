'''
This script tests if your Azure OpenAI deployments actually work by making API calls to them.
'''
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

def test_chat_deployment():
    """Test if the chat deployment works by making a simple completion request."""
    
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [{"role": "user", "content": "Hello, test message"}],
        "max_tokens": 10
    }
    
    try:
        print(f"Testing chat deployment: {AZURE_OPENAI_CHAT_DEPLOYMENT}")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("✅ Chat deployment is working!")
            return True
        else:
            print(f"❌ Chat deployment failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing chat deployment: {e}")
        return False

def test_embedding_deployment():
    """Test if the embedding deployment works by making a simple embedding request."""
    
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_EMBEDDING_DEPLOYMENT}/embeddings?api-version={AZURE_OPENAI_API_VERSION}"
    
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "input": "test text for embedding"
    }
    
    try:
        print(f"Testing embedding deployment: {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("✅ Embedding deployment is working!")
            return True
        else:
            print(f"❌ Embedding deployment failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing embedding deployment: {e}")
        return False

def main():
    """Test both deployments."""
    
    print("Testing Azure OpenAI Deployments...")
    print(f"Endpoint: {AZURE_OPENAI_ENDPOINT}")
    print(f"API Version: {AZURE_OPENAI_API_VERSION}")
    print("=" * 50)
    
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        print("Error: Missing Azure OpenAI credentials in environment variables.")
        return False
    
    # Test chat deployment
    chat_works = test_chat_deployment()
    print()
    
    # Test embedding deployment
    embedding_works = test_embedding_deployment()
    print()
    
    print("Summary:")
    print("-" * 20)
    print(f"Chat deployment ({AZURE_OPENAI_CHAT_DEPLOYMENT}): {'✅ Working' if chat_works else '❌ Not working'}")
    print(f"Embedding deployment ({AZURE_OPENAI_EMBEDDING_DEPLOYMENT}): {'✅ Working' if embedding_works else '❌ Not working'}")
    
    if not embedding_works:
        print(f"\n⚠️  The embedding deployment '{AZURE_OPENAI_EMBEDDING_DEPLOYMENT}' is not working.")
        print("This is why your chatbot is showing the 404 DeploymentNotFound error.")
        print("You need to create this deployment in Azure OpenAI Studio.")
    
    return chat_works and embedding_works

if __name__ == "__main__":
    main()
