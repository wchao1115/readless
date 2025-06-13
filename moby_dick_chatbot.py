'''
This script creates a Gradio chatbot UI to interact with the Moby Dick vectorstore
using gpt-4o-mini as the LLM and the Chroma DB as a RAG context.
'''
import os
import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
CHROMA_DB_PATH = "chroma_db_moby_dick"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Ensure your OPENAI_API_KEY is set
LLM_MODEL = "gpt-4o-mini"

# Global variable for the RAG chain
rag_chain = None

def initialize_chatbot():
    '''Initializes the RAG chain for the chatbot.'''
    global rag_chain
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        # Display error in Gradio UI as well if possible, or handle gracefully
        return False

    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: Chroma DB not found at {CHROMA_DB_PATH}")
        print("Please ensure 'process_moby_dick.py' has been run successfully.")
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
        Use the following pieces of context from the book Moby Dick to answer the question at the end.
        If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
        Provide a concise answer.

        Context: {context}

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

def chat_with_moby_dick(question, history):
    '''Handles the chat interaction with the RAG chain.'''
    if rag_chain is None:
        # Try to initialize again if not already done, or if it failed previously
        if not initialize_chatbot():
            return "Chatbot is not initialized. Please check the console for errors (e.g., missing API key or vector store)."

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

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Chat with Moby Dick 🐋
    Ask any question about Herman Melville's Moby Dick.
    The chatbot uses a local vector store of the book's text for context (RAG).
    """)
    if not initialization_successful:
        gr.Markdown("""
        **Failed to initialize the chatbot.**
        Please ensure your `OPENAI_API_KEY` is set as an environment variable
        and that the `chroma_db_moby_dick` vector store exists (run `process_moby_dick.py` first).
        Check the terminal for more specific error messages.
        You might need to restart this Gradio app after resolving the issues.
        """)

    chatbot_ui = gr.ChatInterface(
        fn=chat_with_moby_dick,
        title="Moby Dick Chatbot",
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask me something about Moby Dick...", container=False, scale=7),
        examples=[
            "Who is Ishmael?",
            "What is the Pequod?",
            "Describe Captain Ahab.",
            "What is the significance of the white whale?"
        ]
    )

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("CRITICAL ERROR: OPENAI_API_KEY is not set. The application will not work.")
        print("Please set it as an environment variable:")
        print("  PowerShell: $Env:OPENAI_API_KEY='your_api_key_here'")
        print("  Bash/Zsh:   export OPENAI_API_KEY='your_api_key_here'")
        # Gradio interface will show a message, but this console message is important too.

    print("Attempting to launch Gradio interface...")
    if initialization_successful:
        print("Chatbot initialized. Gradio is launching.")
    else:
        print("Chatbot failed to initialize. Gradio will launch with an error message.")
        print("Please check for an OPENAI_API_KEY and the existence of the Chroma DB.")

    demo.launch()
