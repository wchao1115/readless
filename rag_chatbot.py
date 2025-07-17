'''
This script creates a Gradio chatbot UI to interact with the One Big Beautiful Bill vectorstore
using Azure OpenAI as the LLM and Azure Cognitive Search as the RAG context.
'''
import os
import gradio as gr
from functools import partial
import time
import threading
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# Azure Cognitive Search Configuration
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "bill-text-index")

# Legacy configuration for backward compatibility
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Keep for fallback
LLM_MODEL = "gpt-4o-mini"

# Global variable for the RAG chain
rag_chain = None

def initialize_chatbot():
    '''Initializes the RAG chain for the chatbot using Azure OpenAI and Azure Cognitive Search.'''
    global rag_chain
    
    # Check Azure OpenAI configuration
    if not AZURE_OPENAI_API_KEY:
        print("Error: AZURE_OPENAI_API_KEY environment variable not set.")
        return False
    
    if not AZURE_OPENAI_ENDPOINT:
        print("Error: AZURE_OPENAI_ENDPOINT environment variable not set.")
        return False
    
    # Check Azure Cognitive Search configuration
    if not AZURE_SEARCH_SERVICE_ENDPOINT:
        print("Error: AZURE_SEARCH_SERVICE_ENDPOINT environment variable not set.")
        return False
    
    if not AZURE_SEARCH_API_KEY:
        print("Error: AZURE_SEARCH_API_KEY environment variable not set.")
        return False

    try:
        print("Initializing Azure OpenAI Embeddings and LLM...")
        
        # Initialize Azure OpenAI Embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        
        # Initialize Azure OpenAI LLM
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0.7
        )

        print(f"Loading Azure Cognitive Search vectorstore from {AZURE_SEARCH_SERVICE_ENDPOINT}...")
        
        # Initialize Azure Cognitive Search vectorstore
        vectorstore = AzureSearch(
            azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
            azure_search_key=AZURE_SEARCH_API_KEY,
            index_name=AZURE_SEARCH_INDEX_NAME,
            embedding_function=embeddings.embed_query
        )
        
        print("Azure vectorstore loaded successfully.")

        # Create a retriever
        retriever = vectorstore.as_retriever()

        # Define a prompt template
        prompt_template = """
        You are a knowledgeable and experienced lawyer who specializes in legislative analysis and has thoroughly studied the bill. Your role is to help people understand this important legislation by explaining it in clear, accessible language that anyone can understand.

        Key instructions for your responses:
        1. ACCURACY IS PARAMOUNT: This bill will soon become law, so never misrepresent or make up information. Only use what's provided in the context.
        2. EXPLAIN IN LAYMAN'S TERMS: Break down complex legal language into simple, everyday terms that non-lawyers can understand.
        3. BE HELPFUL AND THOROUGH: Instead of simply saying "I don't know," try to:
           - Examine related sections that might be relevant
           - Suggest more specific questions the user could ask
           - Provide context about what the bill does cover, even if their exact question isn't answered
        4. SUMMARIZE EFFECTIVELY: Help users understand the practical implications and real-world effects of different provisions.
        5. BE FAIR AND BALANCED: Present information objectively without political bias.
        6. REFER TO THE LEGISLATION: Always refer to the legislation simply as the bill in your responses, not by any specific bill number or formal title.

        When answering:
        - Start with a clear, direct answer when possible
        - Explain what the provision means in practical terms
        - If the context doesn't fully answer the question, acknowledge this but provide what relevant information you can
        - Suggest related questions or areas they might want to explore
        - Use examples or analogies when helpful for understanding

        Context from the bill: {context}

        Question: {question}

        Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        print("RAG chain initialized successfully.")
        return True
    except Exception as e:
        print(f"Error during chatbot initialization: {e}")
        rag_chain = None # Ensure chain is None if initialization fails
        return False

def chat_with_bill(question, history):
    '''Handles the chat interaction with the RAG chain for bill analysis.'''
    if rag_chain is None:
        # Try to initialize again if not already done, or if it failed previously
        if not initialize_chatbot():
            return "Analysis system is not available. Please check the console for errors (e.g., missing API key or vector database)."

    print(f"Received question: {question}")
    try:
        response = rag_chain.invoke({"query": question})
        answer = response.get("result", "Sorry, I could not find an answer.")
        # source_documents = response.get("source_documents", [])
        # if source_documents:
        #     answer += "\n\nSources:"
        #     for doc in source_documents:
        #         answer += f"\n- {doc.metadata.get('source', 'Unknown source')} (Page content snippet: {doc.page_content[:100]}...)"
        print(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error during chat processing: {e}")
        return f"An error occurred: {e}"

# Initialize the chatbot when the script starts
initialization_successful = initialize_chatbot()

# Sample questions for easy access
sample_questions = [
    "Can you give me an overview of what the bill accomplishes?",
    "What are the main tax changes in this bill and how will they affect ordinary taxpayers?",
    "How does this bill change healthcare policy and what does it mean for patients?",
    "What education funding changes does this bill make and who benefits?",
    "How does this bill address infrastructure and what projects does it fund?",
    "What are the defense and national security provisions in this legislation?",
    "How does this bill reform immigration law and what are the key changes?",
    "What environmental and climate provisions are included in this bill?",
    "How does this bill affect Social Security and Medicare?",
    "What are the small business provisions and how do they help entrepreneurs?"
]

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .message.bot, .message.user {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        line-height: 1.6 !important;
        font-size: 14px !important;
    }
    .chatbot .message {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    .small-accordion {
        margin-top: 10px;
        font-size: 0.85em;
        width: 80% !important;
        margin-left: auto;
        margin-right: auto;
    }
    .small-accordion button {
        font-size: 0.8em !important;
        padding: 1px 6px !important;
        opacity: 0.8;
        width: 100%;
        text-align: left;
        font-weight: bold !important;
    }
    .small-accordion > div:nth-child(2) {
        padding: 8px !important;
    }
    .about-accordion .label-wrap span {
        font-weight: bold !important;
    }
    /* Narrower sample question buttons with text wrapping */
    .question-button {
        width: 80% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 32px !important;
        text-align: left !important;
        font-size: 0.85em !important;
        margin-bottom: 5px !important;
    }
    """) as demo:
    gr.Markdown("# Ask The One Big Beautiful Bill Anything! ⚖️")
    gr.Markdown("I'm here to help you understand this important legislation in clear, accessible terms.")
    
    if not initialization_successful:
        gr.Markdown("""
        **⚠️ System Unavailable**
        
        The analysis system could not be initialized. Please ensure:
        - Your Azure OpenAI credentials are properly set:
          - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
          - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
          - `AZURE_OPENAI_CHAT_DEPLOYMENT`: Your chat completion deployment name (default: gpt-4o-mini)
          - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Your embedding deployment name (default: text-embedding-ada-002)
        - Your Azure Cognitive Search credentials are properly set:
          - `AZURE_SEARCH_SERVICE_ENDPOINT`: Your Azure Cognitive Search service endpoint
          - `AZURE_SEARCH_API_KEY`: Your Azure Cognitive Search API key
          - `AZURE_SEARCH_INDEX_NAME`: Your search index name (default: bill-text-index)
        - The bill text has been indexed in Azure Cognitive Search (run updated `process_doc.py` first)
        - Check the terminal for detailed error messages
        
        You may need to restart this application after resolving these issues.
        """)
    
    with gr.Row():
        # Left column for chat interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600, show_label=False)
            msg = gr.Textbox(
                placeholder="Ask me about any provision in the bill - I'll explain it in plain English...",
                label="Your Question",
                container=False
            )
            
        # Right column for sample questions and info
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Popular Questions")
            
            # Create buttons for each sample question with full text and wrapping
            question_buttons = []
            for i, question in enumerate(sample_questions):
                # Use full question text for the button label
                btn = gr.Button(
                    question, 
                    size="sm", 
                    variant="secondary", 
                    elem_classes="question-button"
                )
                question_buttons.append((btn, question))
            
            # Add About this chatbot accordion after all the sample questions
            with gr.Accordion("About this chatbot", open=False, elem_classes=["small-accordion", "about-accordion"]):
                gr.Markdown("""
                <div style="font-size: 0.85em; line-height: 1.4;">
                <strong>What I can help you with:</strong>
                <ul style="margin-top: 5px; margin-bottom: 5px; padding-left: 20px;">
                <li>Explain complex legal provisions in layman's terms</li>
                <li>Summarize key sections and their practical implications</li>
                <li>Clarify how different parts of the bill work together</li>
                <li>Answer specific questions about provisions that interest you</li>
                </ul>
                
                <strong>My commitment:</strong> I prioritize accuracy and fairness. I'll only use information from the actual bill text and will clearly indicate when I cannot answer based on available context.
                
                <em>The chatbot uses advanced RAG (Retrieval-Augmented Generation) with Azure OpenAI and Azure Cognitive Search vector database of the complete bill text.</em>
                </div>
                """)
    
    # Chat functionality
    def get_next_animation_frame(counter):
        """Get the next frame of the animation"""
        base_text = "Analyzing your question"
        max_dots = 4
        
        # Calculate dots based on counter position in the cycle
        cycle_position = counter % 8  # Full cycle is 8 frames (0-4 dots and back)
        
        if cycle_position <= 4:
            # Growing dots (0,1,2,3,4)
            dots = "." * cycle_position
        else:
            # Shrinking dots (3,2,1)
            dots = "." * (8 - cycle_position)
            
        return base_text + dots
    
    def respond(message, chat_history):
        if not message.strip():
            return chat_history, ""
        
        # Add the user's question to show it immediately
        chat_history.append((message, "Analyzing your question"))
        yield chat_history, ""
        
        # Start the actual LLM processing in a background thread
        result_container = {"response": None, "done": False}
        
        def get_llm_response():
            result_container["response"] = chat_with_bill(message, chat_history[:-1])
            result_container["done"] = True
        
        # Start LLM processing in background
        llm_thread = threading.Thread(target=get_llm_response)
        llm_thread.daemon = True
        llm_thread.start()
        
        # Animate while waiting for LLM response
        animation_counter = 0
        while not result_container["done"]:
            animation_counter += 1
            animated_text = get_next_animation_frame(animation_counter)
            chat_history[-1] = (message, animated_text)
            yield chat_history, ""
            time.sleep(0.4)  # Animation speed
        
        # Update with the final response
        chat_history[-1] = (message, result_container["response"])
        yield chat_history, ""
    
    def use_sample_question(question, chat_history):
        # Make a direct copy of respond function to ensure consistent behavior
        if not question.strip():
            return chat_history, ""
        
        # Add the user's question to show it immediately
        chat_history.append((question, "Analyzing your question"))
        yield chat_history, ""
        
        # Start the actual LLM processing in a background thread
        result_container = {"response": None, "done": False}
        
        def get_llm_response():
            result_container["response"] = chat_with_bill(question, chat_history[:-1])
            result_container["done"] = True
        
        # Start LLM processing in background
        llm_thread = threading.Thread(target=get_llm_response)
        llm_thread.daemon = True
        llm_thread.start()
        
        # Animate while waiting for LLM response
        animation_counter = 0
        while not result_container["done"]:
            animation_counter += 1
            animated_text = get_next_animation_frame(animation_counter)
            chat_history[-1] = (question, animated_text)
            yield chat_history, ""
            time.sleep(0.4)  # Animation speed
        
        # Update with the final response
        chat_history[-1] = (question, result_container["response"])
        yield chat_history, ""
    
    # Event handlers for user input
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot], 
        outputs=[chatbot, msg]
    )
    
    # Connect each sample question button to automatically submit the question
    for btn, question in question_buttons:
        btn.click(
            fn=partial(use_sample_question, question),
            inputs=[chatbot],
            outputs=[chatbot, msg]
        )
            
    # Add Azure OpenAI attribution
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            <div style="text-align: center; margin-top: 15px; padding-top: 15px; border-top: 1px solid #e0e0e0; color: #666; font-size: 0.8em;">
            AI-generated content powered by Azure OpenAI • Search powered by Azure Cognitive Search
            </div>
            """)

if __name__ == "__main__":
    if not AZURE_OPENAI_API_KEY:
        print("CRITICAL ERROR: AZURE_OPENAI_API_KEY is not set. The application will not work.")
        print("Please set the required Azure environment variables:")
        print("  PowerShell:")
        print("    $Env:AZURE_OPENAI_API_KEY='your_azure_openai_api_key_here'")
        print("    $Env:AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("    $Env:AZURE_SEARCH_SERVICE_ENDPOINT='https://your-search-service.search.windows.net'")
        print("    $Env:AZURE_SEARCH_API_KEY='your_search_api_key_here'")
        print("  Bash/Zsh:")
        print("    export AZURE_OPENAI_API_KEY='your_azure_openai_api_key_here'")
        print("    export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("    export AZURE_SEARCH_SERVICE_ENDPOINT='https://your-search-service.search.windows.net'")
        print("    export AZURE_SEARCH_API_KEY='your_search_api_key_here'")

    print("Launching Bill Analysis Interface with Azure OpenAI and Azure Cognitive Search...")
    if initialization_successful:
        print("Analysis system initialized successfully. Gradio interface launching.")
    else:
        print("Analysis system failed to initialize. Gradio will launch with an error message.")
        print("Please check for Azure OpenAI and Azure Cognitive Search credentials.")

    demo.launch()
