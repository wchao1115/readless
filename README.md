# Readless: Lets RAG our novels!

## 1. Overview

This project aims to make classic novels more accessible and engaging for readers. The primary motivation is to leverage Retrieval Augmented Generation (RAG) to allow users to quickly understand the plot of a story, ask specific questions, and even have conversations about the narrative. By transforming a novel like Moby Dick into an interactive experience, readers can explore its themes, characters, and plot points more dynamically. Although this project uses Moby Dick as a sample novel, the code is generic and can be adapted to RAG any book or any content that can be reliably scraped off any web page.

The technical foundation of this project is a RAG system. This system comprises several key components: a data ingestion pipeline that fetches and processes the novel's text, a vectorization process that converts text chunks into numerical representations (embeddings), a vector store for efficient similarity searching of these embeddings, and a Large Language Model (LLM) that uses the retrieved context to generate answers and engage in conversation. For this project, we used Herman Melville's "Moby Dick; or, THE WHALE" from Project Gutenberg.

## 2. What is RAG anyway?

Retrieval Augmented Generation (RAG) is a technique used to enhance the responses of Large Language Models (LLMs) by grounding them in specific, external knowledge. LLMs, while powerful, are trained on vast but general datasets, and their knowledge is fixed at the time of training. This means they might not have information on very specific, niche, or very recent topics, or they might sometimes generate plausible-sounding but incorrect information (a phenomenon known as "hallucination").

RAG addresses this by connecting the LLM to an external knowledge base. Here's a breakdown of a typical RAG process:

1.  **User Input:** The user enters a question or prompt into the chatbot or application.
2.  **Query Vectorization:** The user's input (question) is converted into a numerical representation called a vector embedding. This is typically done using an embedding model, such as one provided by OpenAI or other sentence transformers.
3.  **Context Retrieval:** The vectorized question is then used to search a specialized database (vector store) that contains pre-processed chunks of text from the knowledge base (in this case, the Moby Dick novel) and their corresponding embeddings. The system retrieves the text chunks whose embeddings are most similar to the question's embedding.
4.  **Prompt Augmentation:** The original user question and the retrieved relevant text chunks (the "context") are combined into a new, augmented prompt. This prompt is then passed to the LLM.
5.  **LLM Response Generation:** The LLM uses the provided context from the retrieved chunks to generate an informed and relevant answer to the user's original question.
6.  **Output to User:** The LLM's response is then passed back to the user through the chatbot interface.

Below is a visual representation of this RAG process:

<img src="rag_idea.jpg" width="75%">


This approach is crucial because it allows LLMs to provide more accurate, up-to-date, and contextually relevant responses, significantly reducing hallucinations and making them more reliable for specific domains or custom datasets.

## 3. How this project is created?

This project was entirely generated using GitHub Copilot in VS Code, running with Gemini 2.5 Pro (Preview). As complex as it may sound, this project only needs 3 prompts similar to the following ones to generate the 3 Python script files.

**Script 1: `fetch_moby_dick.py`**
*   **Prompt:** "Use BeautifulSoup to scrape off the Moby Dick novel from Gutenberg by converting its HTML web pages into plain text and save it in a file under the `books` folder."

**Script 2: `process_moby_dick.py`**
*   **Prompt:** "Create another Python file that loads the plain text file previously created and split them into chunks using Langchain, then vectorize the chunks and save the results in a Chroma vector store under another sub-folder"

**Script 3: `moby_dick_chatbot.py`**
*   **Prompt:** "Create a Python file that builds a Gradio chatbot UI to RAG the content previously created and stored in the Chroma vectorstore. We want a chatbot that specializes in the story of Moby Dick, the novel so the reader can ask for summarization or any questions they may have about this story."

### Key Technologies Used:

*   **Langchain:** Langchain is an open-source framework designed to simplify the development of applications powered by large language models (LLMs). It provides tools and abstractions for managing prompts, chaining LLM calls, integrating with external data sources (like vector stores), and creating agents. In this project, Langchain is used for text splitting, managing the LLM (OpenAI's gpt-4o-mini), creating embeddings, and orchestrating the RetrievalQA chain which forms the core of the RAG system.

*   **Chroma:** Chroma is an open-source embedding database (vector store) designed to store and efficiently search through vector embeddings. When text is converted into embeddings (numerical representations capturing semantic meaning), Chroma allows for fast similarity searches. In this project, after splitting Moby Dick into chunks and creating embeddings for each chunk, Chroma is used to store these embeddings. When a user asks a question, the RAG system queries Chroma to find the most relevant text chunks from the novel to provide as context to the LLM.

## 4. Setup

To set up the project environment and install all necessary dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

**Note on API Key:**

You will also need to set your OpenAI API key. There are a few ways to do this:

1.  **Environment Variable (Recommended for Production/Sharing):**
    Set it directly in your shell. This is good for temporary sessions or server environments.
    *   PowerShell: `$Env:OPENAI_API_KEY='your_api_key_here'`
    *   Bash/Zsh: `export OPENAI_API_KEY='your_api_key_here'`

2.  **.env File (Recommended for Local Development):**
    Create a file named `.env` in the root directory of the project (`c:\learn\ml\readless\.env`).
    Add your API key to this file like so:
    ```
    OPENAI_API_KEY='your_actual_api_key_here'
    ```
    The Python scripts (`process_moby_dick.py` and `moby_dick_chatbot.py`) are already configured to look for an `OPENAI_API_KEY` environment variable. To make them automatically load it from the `.env` file, you would typically add a library like `python-dotenv`.

    **IMPORTANT SECURITY NOTE:** Never commit your `.env` file or your API key directly into your version control system (like Git). The provided `.gitignore` file in this project is already configured to ignore `.env` files, preventing accidental check-ins of sensitive credentials.

## 5. Running the Chatbot

Once you have set up your environment and API key, you can run the Moby Dick chatbot:

```bash
python moby_dick_chatbot.py
```

This will launch a Gradio interface. Typically, you will see a message in your terminal indicating the application is running, often providing a local URL (e.g., `Running on local URL: http://127.0.0.1:7860`). Open this URL in your web browser to interact with the chatbot.

## 6. Regenerating the Vector Store

If you need to re-process the source text (e.g., if the `moby_dick.txt` file changes or you want to experiment with different chunking or embedding strategies), you can regenerate the Chroma vector store.

To do this, run the following script:

```bash
python process_moby_dick.py
```

This script will:
1. Load the plain text file from the `books` folder.
2. Split the text into chunks using Langchain.
3. Vectorize these chunks.
4. Save the resulting embeddings into the `chroma_db_moby_dick` vector store, overwriting any existing data in that store.

**Note:** Regenerating the vector store can take some time, depending on the size of the text and the processing power of your machine.
