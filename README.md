# AI Study Buddy

A Streamlit application powered by **LangChain** that uses Retrieval-Augmented Generation (RAG) with FAISS vector database to help you study and learn from multiple sources.

## Features

### 📚 Dual Learning Modes
- **Topic Search**: Enter any topic and learn from web-sourced content
- **Document Upload**: Upload one or more PDF documents for personalized learning

### 🤖 AI-Powered Study Tools
- **Socratic Study Helper**: Chat with an AI tutor that guides your learning using the Socratic method
- **Semantic Search**: Uses FAISS vector database with OpenAI embeddings for intelligent content retrieval
- **Context-Aware Conversations**: Maintains chat history for natural, flowing conversations
- **Source Citation**: All information includes citations to documents or web sources

### 🎯 Interactive Learning
- **Smart Quiz Generation**: Auto-generates multiple-choice quizzes from your content
- **Instant Feedback**: Get immediate results with detailed explanations
- **Flexible Content Sources**: Switch between topics and documents seamlessly

### 🔍 Advanced Features
- **Web Search Integration**: Uses Tavily API for comprehensive web research
- **Vector Database**: FAISS-powered semantic search for accurate information retrieval
- **Text Chunking**: Intelligent document splitting for optimal context understanding
- **Dynamic Knowledge Base**: Continuously updates vector store with new search results

## Requirements

- Python 3.8+
- OpenAI API Key (for GPT-4 and embeddings)
- Tavily API Key (for web search functionality)

### Key Dependencies
- `streamlit` - Web application framework
- `langchain` & `langchain-community` - LLM orchestration
- `langchain-openai` - OpenAI integration
- `faiss-cpu` - Vector database for semantic search
- `pypdf` - PDF document processing

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ai-study-buddy
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. Open the application in your web browser (typically at http://localhost:8501)
2. Ensure your OpenAI and Tavily API keys are set in the `.env` file or enter them in the sidebar

### 🔍 Learning from Topics
3. Select **"Search a Topic"** in the sidebar
4. Enter any topic you want to learn about
5. Click **"💬 Ask Questions"** to chat with AI about the topic
6. Or click **"🎯 Create Quiz"** to generate a quiz about the topic

### 📄 Learning from Documents
3. Select **"Upload Documents"** in the sidebar
4. Upload one or more PDF files
5. Click **"💬 Ask Questions"** to chat about your documents
6. Or click **"📝 Create Quiz"** to generate a quiz from your documents

### 💬 Chat Features
- Ask questions and get AI-powered answers with source citations
- Conversation history is maintained for context-aware responses
- Semantic search finds the most relevant information

### 🎯 Quiz Features
- Auto-generated 5-question multiple-choice quizzes
- Instant feedback with correct answers
- Score tracking and performance evaluation

## How It Works

This application leverages **LangChain** and **FAISS** for advanced AI-powered learning:

1. **PDF Processing**: Uses LangChain's `PyPDFLoader` to extract text from uploaded documents
2. **Vector Database**: FAISS vector store with OpenAI embeddings for semantic search
   - Documents are split into optimized chunks using `RecursiveCharacterTextSplitter`
   - Each chunk is embedded using OpenAI's embedding model
   - FAISS indexes enable lightning-fast similarity search
3. **Semantic Retrieval**: Finds the most relevant content using vector similarity instead of keyword matching
4. **LLM Integration**: LangChain's `ChatOpenAI` (GPT-4) for natural language understanding and generation
5. **Web Search**: LangChain's `TavilySearchResults` for comprehensive web research
   - Web content is also indexed in the vector store for semantic search
6. **Conversational AI**: Message schemas (`HumanMessage`, `SystemMessage`) maintain conversation context
7. **Smart Generation**: AI uses semantically retrieved context to generate accurate, Socratic-style responses
8. **Quiz Generation**: Creates interactive multiple-choice quizzes by retrieving relevant content from the vector store

## API Keys

- **OpenAI API Key**: Required for text embeddings and AI responses. Get one at [OpenAI](https://platform.openai.com/)
- **Tavily API Key**: Required for web search functionality. Get one at [Tavily](https://tavily.com/)

## Technical Architecture

### Vector Database
- **FAISS (Facebook AI Similarity Search)**: In-memory vector database for fast semantic search
- **OpenAI Embeddings**: Documents and web content are converted to high-dimensional vectors
- **Chunk Management**: Documents are split into 1000-character chunks with 200-character overlap
- **Dynamic Updates**: Vector store is updated in real-time as new content is added

### Retrieval-Augmented Generation (RAG)
1. User queries are embedded using the same embedding model
2. FAISS performs similarity search to find the most relevant content chunks
3. Retrieved chunks are used as context for GPT-4 to generate accurate responses
4. Sources are tracked and cited in the responses

### Session Management
- Vector stores are maintained in Streamlit session state
- Conversation history is preserved for context-aware interactions
- Separate collections for different topics/documents

## Note

This application is for educational purposes. The AI will always cite its sources, whether from your documents or from web searches.

---
*Powered by LangChain, OpenAI GPT-4, FAISS Vector Database, and Tavily Search* 