'''
This script checks what Azure OpenAI deployments are available in your resource.
'''
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

def check_deployments():
    """Check what deployments are available in your Azure OpenAI resource."""
    
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        print("Error: Missing Azure OpenAI credentials in environment variables.")
        return False
    
    # Try different API versions if one fails
    api_versions = [AZURE_OPENAI_API_VERSION, "2024-06-01", "2024-02-01", "2023-12-01-preview"]
    
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json"
    }
    
    print("Checking Azure OpenAI deployments...")
    print(f"Endpoint: {AZURE_OPENAI_ENDPOINT}")
    print("=" * 50)
    
    for api_version in api_versions:
        try:
            # Construct the API endpoint
            url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments?api-version={api_version}"
            print(f"Trying API version: {api_version}")
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                deployments = response.json()
                
                if "data" in deployments and deployments["data"]:
                    print("Available Deployments:")
                    print("-" * 30)
                    for deployment in deployments["data"]:
                        name = deployment.get("id", "Unknown")
                        model = deployment.get("model", "Unknown")
                        status = deployment.get("status", "Unknown")
                        print(f"Name: {name}")
                        print(f"Model: {model}")
                        print(f"Status: {status}")
                        print("-" * 30)
                    
                    # Check for required deployments
                    deployment_names = [d.get("id", "") for d in deployments["data"]]
                    deployment_models = {d.get("id", ""): d.get("model", "") for d in deployments["data"]}
                    
                    print("\nRequired Deployments Check:")
                    print("-" * 30)
                    
                    chat_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
                    embedding_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
                    
                    # Check for chat deployment (by deployment name)
                    if chat_deployment in deployment_names:
                        model = deployment_models.get(chat_deployment, "Unknown")
                        print(f"✅ Chat deployment '{chat_deployment}' found (model: {model})")
                    else:
                        print(f"❌ Chat deployment '{chat_deployment}' NOT found")
                        # Check if there's any GPT model available
                        gpt_deployments = [name for name, model in deployment_models.items() if 'gpt' in model.lower()]
                        if gpt_deployments:
                            print(f"   Available GPT deployments: {', '.join(gpt_deployments)}")
                            print(f"   You can update AZURE_OPENAI_CHAT_DEPLOYMENT to use one of these")
                    
                    # Check for embedding deployment (by deployment name)
                    if embedding_deployment in deployment_names:
                        model = deployment_models.get(embedding_deployment, "Unknown")
                        print(f"✅ Embedding deployment '{embedding_deployment}' found (model: {model})")
                    else:
                        print(f"❌ Embedding deployment '{embedding_deployment}' NOT found")
                        # Check if there's any embedding model available
                        embedding_deployments = [name for name, model in deployment_models.items() if 'embedding' in model.lower()]
                        if embedding_deployments:
                            print(f"   Available embedding deployments: {', '.join(embedding_deployments)}")
                            print(f"   You can update AZURE_OPENAI_EMBEDDING_DEPLOYMENT to use one of these")
                        else:
                            print(f"   You need to create a deployment with model 'text-embedding-ada-002'")
                    
                    return True
                    
                else:
                    print("No deployments found in your Azure OpenAI resource.")
                    print("You need to create deployments for:")
                    print("- Chat model (e.g., gpt-4o-mini)")
                    print("- Embedding model (text-embedding-ada-002)")
                    return True
            
            else:
                print(f"Error with API version {api_version}: {response.status_code}")
                print(f"Response: {response.text}")
                continue
                
        except Exception as e:
            print(f"Error with API version {api_version}: {e}")
            continue
    
    print("All API versions failed. Please check your Azure OpenAI endpoint and credentials.")
    return False

if __name__ == "__main__":
    check_deployments()
