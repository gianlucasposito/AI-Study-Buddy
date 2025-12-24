# AI Study Buddy - Setup Instructions

## Quick Start Guide

### 1. Install Dependencies

First, install all required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Create Environment File

Create a `.env` file in the project root directory with your API keys:

```bash
# On Windows (PowerShell)
echo "OPENAI_API_KEY=your_actual_openai_key_here" > .env
echo "TAVILY_API_KEY=your_actual_tavily_key_here" >> .env

# On Mac/Linux
echo "OPENAI_API_KEY=your_actual_openai_key_here" > .env
echo "TAVILY_API_KEY=your_actual_tavily_key_here" >> .env
```

**Or manually create a `.env` file** with this content:

```
OPENAI_API_KEY=your_actual_openai_key_here
TAVILY_API_KEY=your_actual_tavily_key_here
```

### 3. Get Your API Keys

#### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and paste it in your `.env` file

#### Tavily API Key
1. Go to https://tavily.com/
2. Sign up for an account
3. Navigate to your dashboard
4. Copy your API key and paste it in your `.env` file

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## What's New

### ✨ Major Changes

1. **Environment Variables**: API keys are now stored securely in a `.env` file instead of being entered in the UI
2. **Topic Search**: You can now generate quizzes on any topic by searching the internet (no documents needed!)
3. **Flexible Learning**: Choose between:
   - **Search a Topic**: Enter any subject and we'll find information online
   - **Upload Documents**: Use your own PDF files for personalized study
4. **Improved UI**: Better layout, clearer instructions, and more engaging visuals

### 📚 How to Use

#### Option 1: Search a Topic
1. Select "Search a Topic" in the sidebar
2. Enter a topic (e.g., "Ancient Rome", "Machine Learning", "Photosynthesis")
3. Click "Create Quiz on This Topic"
4. Take the quiz and test your knowledge!

#### Option 2: Upload Documents
1. Select "Upload Documents" in the sidebar
2. Upload one or more PDF files
3. Either:
   - Chat with the AI tutor about your documents
   - Click "Create Quiz from Documents" to generate a quiz

## Troubleshooting

### Error: "API keys not found"
- Make sure your `.env` file is in the same directory as `app.py`
- Check that the API keys are correctly formatted (no extra spaces or quotes)
- Restart the Streamlit app after creating the `.env` file

### Quiz generation fails
- Ensure your API keys are valid and have sufficient credits
- For topic searches, make sure you have an active internet connection
- For documents, ensure your PDFs contain readable text (not scanned images)

## Support

If you encounter any issues, please check:
1. All dependencies are installed correctly
2. Your `.env` file exists and contains valid API keys
3. You're using Python 3.8 or higher

Enjoy learning with AI Study Buddy! 🎓

