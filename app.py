import streamlit as st
import os
import tempfile
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize API clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Set page configuration
st.set_page_config(layout="wide", page_title="AI Study Buddy")
st.title("AI Study Buddy")

# Display image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("App_Image.png", width=400)

# Initialize session state
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
    st.session_state.content_source = "topic"

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

# Check API keys
def check_api_keys():
    """Check if API keys are set"""
    if not OPENAI_API_KEY or not TAVILY_API_KEY:
        st.error("⚠️ API keys not found! Please add OPENAI_API_KEY and TAVILY_API_KEY to your .env file.")
        st.stop()
        return False
    return True

# Extract text from PDFs using LangChain
def extract_text_from_pdfs(pdf_files):
    """Extract text from uploaded PDF files using LangChain"""
    all_text = ""
    
    for pdf_file in pdf_files:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_path = temp_file.name
        
        # Use LangChain's PyPDFLoader
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        
        # Extract text from documents
        for doc in documents:
            all_text += doc.page_content + "\n"
        
        os.unlink(temp_path)
    
    return all_text

# Search web using LangChain's Tavily tool
def search_web(query):
    """Search the web using Tavily API with LangChain"""
    check_api_keys()
    
    # Use LangChain's TavilySearchResults
    search = TavilySearchResults(
        max_results=10,
        api_key=TAVILY_API_KEY
    )
    results = search.invoke({"query": query})
    
    # Combine search results into text
    content = ""
    for item in results:
        content += f"{item.get('content', '')}\n\n"
    
    return content

# Generate quiz using LangChain
def generate_quiz(content, topic=""):
    """Generate a quiz from content using LangChain's ChatOpenAI"""
    check_api_keys()
    
    # Initialize LangChain's ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY
    )
    
    prompt = f"""Based on the following content, create a 5-question multiple-choice quiz.

Content:
{content[:8000]}

Generate exactly 5 questions with 4 options each (A, B, C, D).
Return ONLY valid JSON in this format:
[
  {{
    "question": "Question text here?",
    "options": {{
      "A": "Option A text",
      "B": "Option B text",
      "C": "Option C text",
      "D": "Option D text"
    }},
    "answer": "A"
  }}
]
"""
    
    # Use LangChain's message format
    messages = [
        SystemMessage(content="You are a quiz generator. Return only valid JSON."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    try:
        quiz_data = json.loads(response.content)
        return quiz_data
    except:
        st.error("Failed to generate quiz. Please try again.")
        return None

# Answer questions using LangChain
def answer_question(question, context, chat_history):
    """Answer a question based on context using LangChain's ChatOpenAI"""
    check_api_keys()
    
    # Initialize LangChain's ChatOpenAI
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Build conversation history using LangChain messages
    messages = [
        SystemMessage(content="""You are a helpful study tutor. Answer questions based on the provided context.
- Be clear and concise
- Use the Socratic method to guide understanding
- Cite your sources when providing information
- If you need more information, say so""")
    ]
    
    # Add chat history (last 3 exchanges)
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(SystemMessage(content=msg["content"]))
    
    # Add current question with context
    messages.append(HumanMessage(content=f"Context:\n{context[:4000]}\n\nQuestion: {question}"))
    
    response = llm.invoke(messages)
    
    return response.content

# Sidebar
with st.sidebar:
    st.header("📚 Content Source")
    
    content_option = st.radio(
        "Choose how to generate your quiz:",
        ["Search a Topic", "Upload Documents"],
        help="Select whether to search the internet or upload documents"
    )
    
    st.session_state.content_source = "topic" if content_option == "Search a Topic" else "documents"
    
    st.divider()
    
    if st.session_state.content_source == "topic":
        st.subheader("🔍 Topic Search")
        quiz_topic = st.text_input(
            "Enter a topic:",
            placeholder="e.g., Python programming, Photosynthesis",
            help="Enter any topic you'd like to learn about"
        )
        st.session_state.quiz_topic = quiz_topic
        uploaded_files = None
        
        st.markdown("**Choose an action:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💬 Ask Questions", use_container_width=True):
                if quiz_topic:
                    st.session_state.quiz_active = False
                    st.session_state.messages = []  # Reset chat for new topic
                    st.rerun()
                else:
                    st.warning("Please enter a topic!")
        
        with col2:
            if st.button("🎯 Create Quiz", use_container_width=True):
                if quiz_topic:
                    st.session_state.quiz_active = True
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_data = None
                else:
                    st.warning("Please enter a topic!")
    else:
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files"
        )
        st.session_state.quiz_topic = ""
        
        if uploaded_files:
            st.markdown("**Choose an action:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💬 Ask Questions", use_container_width=True, key="doc_chat"):
                    st.session_state.quiz_active = False
                    st.session_state.messages = []  # Reset chat for new documents
                    st.rerun()
            
            with col2:
                if st.button("📝 Create Quiz", use_container_width=True, key="doc_quiz"):
                    st.session_state.quiz_active = True
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_data = None

# Main content
if st.session_state.quiz_active:
    # Generate and display quiz
    if not st.session_state.quiz_data:
        with st.spinner("🔍 Generating your quiz..."):
            if st.session_state.content_source == "topic" and st.session_state.quiz_topic:
                # Search web for topic
                content = search_web(st.session_state.quiz_topic)
                st.session_state.document_text = content
                st.session_state.quiz_data = generate_quiz(content, st.session_state.quiz_topic)
            elif st.session_state.content_source == "documents" and uploaded_files:
                # Extract text from PDFs
                content = extract_text_from_pdfs(uploaded_files)
                st.session_state.document_text = content
                st.session_state.quiz_data = generate_quiz(content)
    
    if st.session_state.quiz_data:
        # Display source
        if st.session_state.content_source == "topic":
            st.info(f"📚 Quiz Topic: **{st.session_state.quiz_topic}**")
        else:
            st.info(f"📄 Quiz from {len(uploaded_files)} document(s)")
        
        st.header("🎯 Quiz Time!")
        st.markdown("---")
        
        # Display questions
        for i, question in enumerate(st.session_state.quiz_data):
            with st.container():
                st.markdown(f"### Question {i+1}")
                st.markdown(f"**{question['question']}**")
                
                options = question['options']
                answer_key = st.radio(
                    "Select your answer:",
                    options=list(options.keys()),
                    format_func=lambda x, opts=options: f"{x}. {opts[x]}",
                    key=f"q{i}",
                    label_visibility="collapsed"
                )
                
                st.session_state.quiz_answers[i] = answer_key
                st.markdown("---")
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ Submit Quiz", use_container_width=True, type="primary"):
                st.session_state.quiz_submitted = True
        
        # Show results
        if st.session_state.quiz_submitted:
            correct_count = 0
            
            st.markdown("---")
            st.header("📊 Quiz Results")
            
            for i, question in enumerate(st.session_state.quiz_data):
                user_answer = st.session_state.quiz_answers.get(i, None)
                correct_answer = question['answer']
                
                if user_answer == correct_answer:
                    correct_count += 1
                    st.success(f"✓ **Question {i+1}:** Correct!")
                    st.markdown(f"*Your answer: {user_answer}. {question['options'][user_answer]}*")
                else:
                    st.error(f"✗ **Question {i+1}:** Incorrect")
                    st.markdown(f"*Your answer: {user_answer}. {question['options'][user_answer]}*")
                    st.markdown(f"*Correct answer: {correct_answer}. {question['options'][correct_answer]}*")
            
            # Final score
            score = (correct_count / len(st.session_state.quiz_data)) * 100
            
            st.markdown("---")
            if score >= 80:
                st.balloons()
                st.success(f"### 🌟 Excellent! Score: {correct_count}/{len(st.session_state.quiz_data)} ({score:.1f}%)")
            elif score >= 60:
                st.info(f"### 👍 Good Job! Score: {correct_count}/{len(st.session_state.quiz_data)} ({score:.1f}%)")
            else:
                st.warning(f"### 📖 Keep Studying! Score: {correct_count}/{len(st.session_state.quiz_data)} ({score:.1f}%)")
            
            # Back button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🏠 Back to Main", use_container_width=True):
                    st.session_state.quiz_active = False
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.rerun()

elif st.session_state.content_source == "documents" and uploaded_files:
    # Chat interface for documents
    if not st.session_state.document_text:
        with st.spinner("📄 Processing documents..."):
            st.session_state.document_text = extract_text_from_pdfs(uploaded_files)
    
    st.header("💬 Study Helper Chat")
    st.markdown(f"*Ask questions about your {len(uploaded_files)} uploaded document(s)*")
    st.markdown("---")
    
    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.write(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(
                    user_input,
                    st.session_state.document_text,
                    st.session_state.messages
                )
                st.write(response)
        
        # Add AI response
        st.session_state.messages.append({"role": "assistant", "content": response})

elif st.session_state.content_source == "topic" and st.session_state.quiz_topic:
    # Chat interface for topics
    st.header("💬 Study Helper Chat")
    st.markdown(f"*Ask questions about: **{st.session_state.quiz_topic}***")
    st.markdown("---")
    
    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.write(msg["content"])
    
    # Chat input
    user_input = st.chat_input(f"Ask a question about {st.session_state.quiz_topic}...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Search web for context if needed
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get fresh context from web
                search_query = f"{st.session_state.quiz_topic} {user_input}"
                context = search_web(search_query)
                
                response = answer_question(
                    user_input,
                    context,
                    st.session_state.messages
                )
                st.write(response)
        
        # Add AI response
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to **AI Study Buddy**! 🎓
    
    Your AI-powered learning companion that makes studying simple and effective.
    
    ### 🚀 Getting Started
    
    Choose how you want to learn:
    
    #### 🔍 **Search a Topic**
    - Enter any topic you want to learn about
    - We'll search the internet for information
    - Generate a quiz to test your knowledge
    - Chat with AI about the topic
    
    #### 📄 **Upload Documents**
    - Upload your PDF study materials
    - Chat with AI about your documents
    - Generate custom quizzes from your materials
    
    ### ✨ Features
    
    - **🤖 AI Tutor**: Ask questions and get helpful answers
    - **📝 Smart Quizzes**: Auto-generated multiple-choice questions
    - **🌐 Web Search**: Internet search for comprehensive learning
    - **💡 Simple & Fast**: Clean interface, direct API calls
    
    ### 👈 Select an option from the sidebar to begin!
    
    ---
    *Powered by LangChain, OpenAI GPT-4, and Tavily Search*
    """)
