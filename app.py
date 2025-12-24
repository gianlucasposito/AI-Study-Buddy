import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.schema import Document
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# PDF processing
import PyPDF2

# Tavily Search API
from tavily import TavilyClient

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Set page configuration
st.set_page_config(layout="wide", page_title="AI Study Buddy")
st.title("AI Study Buddy")

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1])
# Display image in the center column
with col2:
    st.image("App_Image.png", width=400)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False
    
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
    
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
    
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""

if "content_source" not in st.session_state:
    st.session_state.content_source = "topic"  # "topic" or "documents"

# API key validation will be done when actually needed (when generating quiz or answering questions)
# This allows testing the UI without valid API keys

# Helper function to check API keys before making API calls
def check_api_keys():
    """Check if API keys are set before making API calls."""
    if not OPENAI_API_KEY or not TAVILY_API_KEY:
        st.error("⚠️ API keys not found! Please add valid OPENAI_API_KEY and TAVILY_API_KEY to your .env file to use this feature.")
        st.info("💡 **Tip**: You can explore the UI, but you need real API keys to generate quizzes or chat with the AI.")
        st.stop()
        return False
    return True

# Sidebar for content source selection
with st.sidebar:
    st.header("📚 Content Source")
    
    content_option = st.radio(
        "Choose how to generate your quiz:",
        ["Search a Topic", "Upload Documents"],
        help="Select whether to search the internet for a topic or upload your own documents"
    )
    
    st.session_state.content_source = "topic" if content_option == "Search a Topic" else "documents"
    
    st.divider()
    
    # Topic input or file upload based on selection
    if st.session_state.content_source == "topic":
        st.subheader("🔍 Topic Search")
        quiz_topic = st.text_input(
            "Enter a topic for your quiz:",
            placeholder="e.g., 'World War II', 'Python programming', 'Photosynthesis'",
            help="Enter any topic you'd like to learn about. We'll search the internet for information."
        )
        st.session_state.quiz_topic = quiz_topic
        uploaded_files = None
        
        if st.button("🎯 Create Quiz on This Topic", use_container_width=True):
            if quiz_topic:
                st.session_state.quiz_active = True
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.session_state.quiz_data = None
            else:
                st.warning("Please enter a topic first!")
    else:
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to create a quiz from"
        )
        st.session_state.quiz_topic = ""
        
        if uploaded_files and st.button("📝 Create Quiz from Documents", use_container_width=True):
            st.session_state.quiz_active = True
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_data = None

# Function to extract text from PDF files
def extract_text_from_pdfs(pdf_files) -> List[Document]:
    documents = []
    
    for pdf_file in pdf_files:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_path = temp_file.name
        
        # Extract text from the PDF
        text = ""
        with open(temp_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                
        # Create a Document with metadata
        doc = Document(
            page_content=text,
            metadata={"source": pdf_file.name}
        )
        documents.append(doc)
        
        # Clean up the temporary file
        os.unlink(temp_path)
    
    return documents

# Function to process documents and create vector store
@st.cache_resource
def process_documents(_documents):
    # Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for doc in _documents:
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    return vector_store

# Create Tavily search tool
def get_tavily_search_tool():
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    def tavily_search(query: str) -> str:
        """Search the web for information using Tavily API."""
        search_result = tavily_client.search(query=query, search_depth="advanced", include_images=True)
        return json.dumps(search_result, indent=2)
    
    return Tool(
        name="tavily_search",
        description="Search the web for information. Use this when you need to find information not present in the user's documents, or when looking for images or tables.",
        func=tavily_search
    )

# Function to search topic and get content for quiz
def search_topic_content(topic: str) -> str:
    """Search the web for information about a topic using Tavily."""
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # Perform comprehensive search
    search_result = tavily_client.search(
        query=topic,
        search_depth="advanced",
        max_results=10
    )
    
    # Extract and combine content from search results
    content = f"Topic: {topic}\n\n"
    for result in search_result.get('results', []):
        content += f"Source: {result.get('url', 'Unknown')}\n"
        content += f"{result.get('content', '')}\n\n"
    
    return content

# Create the Study Helper Agent
def create_study_helper_agent(vector_store):
    check_api_keys()  # Validate API keys before making calls
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # Create tools
    tools = []
    
    # Add Tavily search tool
    tavily_tool = get_tavily_search_tool()
    tools.append(tavily_tool)
    
    # Create system prompt
    system_prompt = """You are an expert Socratic tutor and study assistant. Your primary goal is to help the user understand the material, not to simply give them answers.

1. You will be given a user's question, conversation history, and context (from documents or web search).
2. **Source Citation is Mandatory:** For every piece of information you provide, you MUST cite its source.
   - If the information is from the user's documents, state: "According to your document '[document name]'..."
   - If the information is from the web, state: "According to my web search..."
3. **Socratic Method:** Guide the user with thoughtful questions that help them understand concepts deeply.
4. **Tool Usage:** If you need additional information, use the 'tavily_search' tool to find information on the internet.
"""

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Here is relevant context: {context}")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return retriever, agent_executor

# Create Quiz Master function for documents
def generate_quiz(vector_store):
    check_api_keys()  # Validate API keys before making calls
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Get a diverse sample of chunks from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    documents = retriever.get_relevant_documents("give me a diverse sample of content from all documents")
    
    # Prepare context from documents
    context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in documents])
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # System prompt for quiz generation
    system_prompt = """You are an assistant that creates educational quizzes. Based ONLY on the context provided below, generate a 5-question multiple-choice quiz.
- Each question must have 4 options (A, B, C, D).
- One option must be the correct answer.
- Return the output as a single, valid JSON object. The object should be a list where each item contains keys: 'question', 'options' (a dictionary of A, B, C, D), and 'answer' (the correct key, e.g., 'A').
"""
    
    # Generate quiz
    response = llm.invoke([
        ("system", system_prompt),
        ("human", f"Context:\n{context}\n\nGenerate a quiz based only on this context.")
    ])
    
    # Parse JSON response
    try:
        quiz_data = json.loads(response.content)
        return quiz_data
    except json.JSONDecodeError:
        st.error("Failed to generate quiz. Please try again.")
        return None

# Create Quiz Master function for topics
def generate_quiz_from_topic(topic: str):
    check_api_keys()  # Validate API keys before making calls
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Search for topic content
    context = search_topic_content(topic)
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # System prompt for quiz generation
    system_prompt = """You are an assistant that creates educational quizzes. Based on the context provided below about a topic, generate a 5-question multiple-choice quiz.
- Each question must have 4 options (A, B, C, D).
- One option must be the correct answer.
- Focus on key concepts and important facts about the topic.
- Return the output as a single, valid JSON object. The object should be a list where each item contains keys: 'question', 'options' (a dictionary of A, B, C, D), and 'answer' (the correct key, e.g., 'A').
"""
    
    # Generate quiz
    response = llm.invoke([
        ("system", system_prompt),
        ("human", f"Context:\n{context}\n\nGenerate a quiz about '{topic}' based on this context.")
    ])
    
    # Parse JSON response
    try:
        quiz_data = json.loads(response.content)
        return quiz_data
    except json.JSONDecodeError:
        st.error("Failed to generate quiz. Please try again.")
        return None

# Main application logic
# Display quiz interface or study helper based on state
if st.session_state.quiz_active:
    # Generate and display quiz
    if not st.session_state.quiz_data:
        with st.spinner("🔍 Generating your quiz..."):
            if st.session_state.content_source == "topic" and st.session_state.quiz_topic:
                st.session_state.quiz_data = generate_quiz_from_topic(st.session_state.quiz_topic)
            elif st.session_state.content_source == "documents" and uploaded_files:
                documents = extract_text_from_pdfs(uploaded_files)
                vector_store = process_documents(documents)
                st.session_state.quiz_data = generate_quiz(vector_store)
    
    if st.session_state.quiz_data:
        # Display source info
        if st.session_state.content_source == "topic":
            st.info(f"📚 Quiz Topic: **{st.session_state.quiz_topic}**")
        else:
            st.info(f"📄 Quiz based on {len(uploaded_files)} uploaded document(s)")
        
        st.header("🎯 Quiz Time!")
        st.markdown("---")
        
        # Display quiz questions
        for i, question in enumerate(st.session_state.quiz_data):
            with st.container():
                st.markdown(f"### Question {i+1}")
                st.markdown(f"**{question['question']}**")
                
                # Create radio buttons for options
                options = question['options']
                answer_key = st.radio(
                    "Select your answer:",
                    options=list(options.keys()),
                    format_func=lambda x, opts=options: f"{x}. {opts[x]}",
                    key=f"q{i}",
                    label_visibility="collapsed"
                )
                
                # Store the selected answer
                st.session_state.quiz_answers[i] = answer_key
                
                st.markdown("---")
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ Submit Quiz", use_container_width=True, type="primary"):
                st.session_state.quiz_submitted = True
        
        # Show results if submitted
        if st.session_state.quiz_submitted:
            correct_count = 0
            
            st.markdown("---")
            st.header("📊 Quiz Results")
            
            for i, question in enumerate(st.session_state.quiz_data):
                user_answer = st.session_state.quiz_answers.get(i, None)
                correct_answer = question['answer']
                
                with st.container():
                    if user_answer == correct_answer:
                        correct_count += 1
                        st.success(f"✓ **Question {i+1}:** Correct!")
                        st.markdown(f"*Your answer: {user_answer}. {question['options'][user_answer]}*")
                    else:
                        st.error(f"✗ **Question {i+1}:** Incorrect")
                        st.markdown(f"*Your answer: {user_answer}. {question['options'][user_answer]}*")
                        st.markdown(f"*Correct answer: {correct_answer}. {question['options'][correct_answer]}*")
            
            # Display final score
            score_percentage = (correct_count / len(st.session_state.quiz_data)) * 100
            
            st.markdown("---")
            if score_percentage >= 80:
                st.balloons()
                st.success(f"### 🌟 Excellent! Your Score: {correct_count}/{len(st.session_state.quiz_data)} ({score_percentage:.1f}%)")
            elif score_percentage >= 60:
                st.info(f"### 👍 Good Job! Your Score: {correct_count}/{len(st.session_state.quiz_data)} ({score_percentage:.1f}%)")
            else:
                st.warning(f"### 📖 Keep Studying! Your Score: {correct_count}/{len(st.session_state.quiz_data)} ({score_percentage:.1f}%)")
            
            # Button to return to main interface
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🏠 Back to Main", use_container_width=True):
                    st.session_state.quiz_active = False
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.rerun()
                    
elif st.session_state.content_source == "documents" and uploaded_files:
    # Document-based study helper
    documents = extract_text_from_pdfs(uploaded_files)
    vector_store = process_documents(documents)
    
    st.header("💬 Study Helper Chat")
    st.markdown(f"*Ask questions about your {len(uploaded_files)} uploaded document(s)*")
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
    
    # Get user input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Create retriever and agent
        retriever, agent_executor = create_study_helper_agent(vector_store)
        
        # Get relevant documents
        relevant_docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in relevant_docs])
        
        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Convert chat history to messages format
                chat_history = st.session_state.messages[:-1]  # Exclude the latest user message
                
                # Run agent
                response = agent_executor.invoke({
                    "input": user_input,
                    "context": context,
                    "chat_history": chat_history
                })
                
                # Display response
                st.write(response["output"])
        
        # Add AI response to chat history
        st.session_state.messages.append(AIMessage(content=response["output"]))
else:
    # Welcome message
    st.markdown("""
    ## Welcome to **AI Study Buddy**! 🎓
    
    Your personalized AI-powered learning companion that makes studying smarter and more effective.
    
    ### 🚀 Getting Started
    
    Choose how you want to learn:
    
    #### 🔍 **Search a Topic**
    - Enter any topic you want to learn about
    - We'll search the internet for comprehensive information
    - Generate an instant quiz to test your knowledge
    - Perfect for exploring new subjects quickly!
    
    #### 📄 **Upload Documents**
    - Upload your PDF study materials (textbooks, notes, articles)
    - Chat with an AI tutor about your documents
    - Get clarifications and deeper understanding
    - Generate custom quizzes from your materials
    
    ### ✨ Features
    
    - **🤖 AI Tutor**: Ask questions and get thoughtful, Socratic guidance
    - **📝 Smart Quizzes**: Automatically generated multiple-choice questions
    - **🌐 Web Search**: Integrated internet search for comprehensive learning
    - **💡 Interactive Learning**: Engage with content in a conversational way
    
    ### 👈 Select an option from the sidebar to begin!
    
    ---
    *Powered by OpenAI GPT-4 and Tavily Search*
    """)
