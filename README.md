# AI Study Buddy

A Streamlit application that uses Retrieval-Augmented Generation (RAG) to help you study and learn from your PDF documents.

## Features

- **Document Upload**: Upload one or more PDF documents
- **Socratic Study Helper**: Chat with an AI tutor that guides your learning using the Socratic method
- **Source Citation**: All information provided by the AI includes citations to either your documents or web sources
- **Web Search Integration**: Uses Tavily API to search the web when needed
- **Quiz Generation**: Create quizzes based on your uploaded documents to test your knowledge

## Requirements

- Python 3.8+
- OpenAI API Key
- Tavily API Key (for web search functionality)

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd ai-study-buddy
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run app.py
```

## Usage

1. Open the application in your web browser (typically at http://localhost:8501)
2. Enter your OpenAI and Tavily API keys in the sidebar
3. Upload one or more PDF documents using the file uploader
4. Start chatting with the AI Study Helper about the content of your documents
5. Click "Create a Quiz for Me!" to generate a quiz based on your documents

## How It Works

1. **PDF Processing**: When you upload PDFs, the text is extracted and split into manageable chunks
2. **Vector Database**: Text chunks are converted into embeddings and stored in a ChromaDB vector database
3. **Retrieval**: When you ask a question, the most relevant chunks are retrieved from the database
4. **Generation**: The AI uses the retrieved context to generate helpful, Socratic-style responses
5. **Web Search**: If your question requires information not in your documents, the AI can search the web
6. **Quiz Generation**: The quiz feature samples diverse content from your documents to create relevant questions

## API Keys

- **OpenAI API Key**: Required for text embeddings and AI responses. Get one at [OpenAI](https://platform.openai.com/)
- **Tavily API Key**: Required for web search functionality. Get one at [Tavily](https://tavily.com/)

## Note

This application is for educational purposes. The AI will always cite its sources, whether from your documents or from web searches. 