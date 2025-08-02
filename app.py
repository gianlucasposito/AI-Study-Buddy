import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import json

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

# Sidebar for API keys and controls
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    
    # Quiz button
    if st.button("Create a Quiz for Me!"):
        st.session_state.quiz_active = True
        st.session_state.quiz_submitted = False
        st.session_state.quiz_answers = {}

# File uploader for PDF documents
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your documents", type="pdf", accept_multiple_files=True)

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
def process_documents(_documents, _openai_api_key):
    if not _openai_api_key:
        st.error("Please provide an OpenAI API key.")
        return None
    
    # Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = _openai_api_key
    
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
def get_tavily_search_tool(api_key):
    tavily_client = TavilyClient(api_key=api_key)
    
    def tavily_search(query: str) -> str:
        """Search the web for information using Tavily API."""
        search_result = tavily_client.search(query=query, search_depth="advanced", include_images=True)
        return json.dumps(search_result, indent=2)
    
    return Tool(
        name="tavily_search",
        description="Search the web for information. Use this when you need to find information not present in the user's documents, or when looking for images or tables.",
        func=tavily_search
    )

# Create the Study Helper Agent
def create_study_helper_agent(vector_store, openai_api_key, tavily_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # Create tools
    tools = []
    
    # Add Tavily search tool if API key is provided
    if tavily_api_key:
        tavily_tool = get_tavily_search_tool(tavily_api_key)
        tools.append(tavily_tool)
    
    # Create system prompt
    system_prompt = """You are an expert Socratic tutor and study assistant. Your primary goal is to help the user understand the material in their uploaded documents, not to simply give them answers.

1. You will be given a user's question, conversation history, and a context retrieved from their documents.
2. **Source Citation is Mandatory:** For every piece of information you provide, you MUST cite its source.
   - If the information is from the user's documents, state: "According to your document '[document name]'..."
   - If the information is from the web, state: "According to my web search at [URL]..."
3. **Socratic Method:** Do not provide the direct answer immediately. Ask a leading question that points the user toward the key information in the context.
4. **Tool Usage:** If the document context is insufficient, or if the user asks for information outside their documents (including requests for images or tables), you MUST use the 'tavily_search' tool to find information on the internet.
"""

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Here is relevant context from the documents: {context}")
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

# Create Quiz Master function
def generate_quiz(vector_store, openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
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

# Main application logic
if uploaded_files:
    # Process documents
    documents = extract_text_from_pdfs(uploaded_files)
    
    # Create vector store
    if openai_api_key:
        vector_store = process_documents(documents, openai_api_key)
        
        # Display chat interface or quiz based on state
        if st.session_state.quiz_active:
            # Generate and display quiz
            if not st.session_state.quiz_data:
                with st.spinner("Generating quiz..."):
                    st.session_state.quiz_data = generate_quiz(vector_store, openai_api_key)
            
            if st.session_state.quiz_data:
                st.header("Quiz Time!")
                
                # Display quiz questions
                for i, question in enumerate(st.session_state.quiz_data):
                    st.subheader(f"Question {i+1}: {question['question']}")
                    
                    # Create radio buttons for options
                    options = question['options']
                    answer_key = st.radio(
                        "Select your answer:",
                        options=list(options.keys()),
                        format_func=lambda x: f"{x}: {options[x]}",
                        key=f"q{i}"
                    )
                    
                    # Store the selected answer
                    st.session_state.quiz_answers[i] = answer_key
                    
                    st.divider()
                
                # Submit button
                if st.button("Submit Quiz"):
                    st.session_state.quiz_submitted = True
                
                # Show results if submitted
                if st.session_state.quiz_submitted:
                    correct_count = 0
                    
                    st.header("Quiz Results")
                    for i, question in enumerate(st.session_state.quiz_data):
                        user_answer = st.session_state.quiz_answers.get(i, None)
                        correct_answer = question['answer']
                        
                        if user_answer == correct_answer:
                            correct_count += 1
                            st.success(f"Question {i+1}: Correct! Your answer: {user_answer}")
                        else:
                            st.error(f"Question {i+1}: Incorrect. Your answer: {user_answer}, Correct answer: {correct_answer}")
                    
                    # Display final score
                    score_percentage = (correct_count / len(st.session_state.quiz_data)) * 100
                    st.header(f"Your Score: {correct_count}/{len(st.session_state.quiz_data)} ({score_percentage:.1f}%)")
                    
                    # Button to return to chat
                    if st.button("Return to Study Helper"):
                        st.session_state.quiz_active = False
                        st.session_state.quiz_data = None
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}
                        st.experimental_rerun()
        else:
            # Chat interface
            st.header("Study Helper Chat")
            
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
                
                # Create retriever and agent if not already created
                retriever, agent_executor = create_study_helper_agent(vector_store, openai_api_key, tavily_api_key)
                
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
        st.error("Please provide an OpenAI API key in the sidebar.")
else:
    # Welcome message when no documents are uploaded
    st.write("""Welcome to **AI Study Buddy**, your personalized learning companion! Simply upload your PDF study materials to unlock powerful AI‑driven features that make mastering any subject easier and more engaging.

🔍 Interactive Q&A
                          
Ask questions and have real‑time conversations with your AI Study Helper—clarify concepts, explore examples, and dive deeper into your documents.

📝 Custom Quiz Generation
             
Reinforce your understanding by automatically generating tailored quizzes. Choose your preferred format—multiple choice, true/false, or short answer—and test yourself on the topics that matter most.

Let’s make learning smarter, faster, and more fun with AI Study Buddy!

""")
