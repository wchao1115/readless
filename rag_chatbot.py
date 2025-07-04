'''
This script creates a Gradio chatbot UI to interact with the One Big Beautiful Bill vectorstore
using gpt-4o-mini as the LLM and the Chroma DB as a RAG context.
'''
import os
import gradio as gr
from functools import partial
import time
import threading
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
CHROMA_DB_PATH = "chroma_db_bill_text"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Ensure your OPENAI_API_KEY is set
LLM_MODEL = "gpt-4o-mini"

# Global variable for the RAG chain
rag_chain = None

def initialize_chatbot():
    '''Initializes the RAG chain for the chatbot.'''
    global rag_chain
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return False

    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: Chroma DB not found at {CHROMA_DB_PATH}")
        print("Please ensure 'process_doc.py' has been run successfully.")
        return False

    try:
        print("Initializing OpenAI Embeddings and LLM...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        llm = ChatOpenAI(temperature=0.7, model_name=LLM_MODEL, openai_api_key=OPENAI_API_KEY)

        print(f"Loading Chroma vectorstore from {CHROMA_DB_PATH}...")
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        print("Vectorstore loaded successfully.")

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
    }
    .small-accordion button {
        font-size: 0.8em !important;
        padding: 1px 6px !important;
        opacity: 0.8;
        width: 100%;
        text-align: left;
    }
    .small-accordion > div:nth-child(2) {
        padding: 8px !important;
    }
    """) as demo:
    gr.Markdown("# Ask The One Big Beautiful Bill Anything! ⚖️")
    gr.Markdown("I'm here to help you understand this important legislation in clear, accessible terms.")
    
    if not initialization_successful:
        gr.Markdown("""
        **⚠️ System Unavailable**
        
        The analysis system could not be initialized. Please ensure:
        - Your `OPENAI_API_KEY` is properly set as an environment variable
        - The bill text vector database exists (run `process_doc.py` first to create it)
        - Check the terminal for detailed error messages
        
        You may need to restart this application after resolving these issues.
        """)
    
    with gr.Row():
        # Left column for chat interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, show_label=False)
            msg = gr.Textbox(
                placeholder="Ask me about any provision in the bill - I'll explain it in plain English...",
                label="Your Question",
                container=False
            )
            
        # Right column for sample questions and info
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Popular Questions")
            
            # Create buttons for each sample question
            question_buttons = []
            for i, question in enumerate(sample_questions):
                # Create a shortened version for the button label
                short_label = question[:50] + "..." if len(question) > 50 else question
                btn = gr.Button(short_label, size="sm", variant="secondary")
                question_buttons.append((btn, question))
            
            # Add some spacing
            gr.Markdown("<br>")
            
            # Add About this chatbot accordion
            with gr.Accordion("About this chatbot", open=False, elem_classes="small-accordion"):
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
                
                <em>The chatbot uses advanced RAG (Retrieval-Augmented Generation) with a vector database of the complete bill text.</em>
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
        # Add the question to chat history
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
    
    # Event handlers
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    
    # Connect each sample question button to automatically submit the question
    for btn, question in question_buttons:
        # Use functools.partial to create a proper closure that works with generators
        btn.click(
            fn=partial(use_sample_question, question),
            inputs=[chatbot],
            outputs=[chatbot, msg]
        )
            
    # Add OpenAI attribution
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            <div style="text-align: center; margin-top: 15px; padding-top: 15px; border-top: 1px solid #e0e0e0; color: #666; font-size: 0.8em;">
            AI-generated content powered by OpenAI
            </div>
            """)

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("CRITICAL ERROR: OPENAI_API_KEY is not set. The application will not work.")
        print("Please set it as an environment variable:")
        print("  PowerShell: $Env:OPENAI_API_KEY='your_api_key_here'")
        print("  Bash/Zsh:   export OPENAI_API_KEY='your_api_key_here'")

    print("Launching Bill Analysis Interface...")
    if initialization_successful:
        print("Analysis system initialized successfully. Gradio interface launching.")
    else:
        print("Analysis system failed to initialize. Gradio will launch with an error message.")
        print("Please check for OPENAI_API_KEY and ensure the Chroma DB exists.")

    demo.launch()
