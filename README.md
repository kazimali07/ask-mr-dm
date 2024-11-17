# Ask Mr. Data Modeler 🤖

An intelligent AI-powered assistant that helps you understand and analyze data models through natural conversation. Built with Streamlit and powered by OpenAI's language models.

## 🌟 Features

- **PDF Document Processing**: Automatically extracts and processes text from uploaded PDF files
- **Intelligent Conversations**: Engage in natural dialogue about your data models and documentation
- **Vector-Based Search**: Uses FAISS for efficient information retrieval
- **Persistent Knowledge**: Saves processed documents in a vector store for quick future access
- **User-Friendly Interface**: Clean Streamlit interface for easy interaction
- **Real-Time Responses**: Get immediate AI-powered responses to your questions

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd ask-mr-dm
   ```

2. **Set Up Environment**
   ```bash
   # Create and activate a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the App**
   - Open your browser and go to `http://localhost:8501`
   - Start uploading PDFs and asking questions!

## 📋 Requirements

- Python 3.x
- OpenAI API key
- Required Python packages (installed via requirements.txt):
  - streamlit
  - langchain
  - openai
  - python-dotenv
  - PyPDF2
  - faiss-cpu
  - certifi

## 💡 How It Works

1. **Document Processing**:
   - Upload PDF documents through the Streamlit interface
   - The app extracts text and splits it into manageable chunks
   - Text chunks are converted into vector embeddings using OpenAI's embedding model

2. **Vector Store**:
   - Embeddings are stored in a FAISS vector store
   - The vector store is saved locally for persistence
   - Efficient similarity search enables quick information retrieval

3. **Conversation Chain**:
   - Uses LangChain's ConversationalRetrievalChain
   - Maintains conversation context for more natural interactions
   - Combines retrieved context with OpenAI's language model for accurate responses

## 🔒 Security Note

- Never commit your `.env` file or expose your API keys
- The vector store contains processed text from your documents; ensure you have appropriate permissions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

This project is licensed under the terms included in the LICENSE file.

## 🆘 Troubleshooting

- Ensure all dependencies are correctly installed
- Check that your OpenAI API key is valid and properly set in the `.env` file
- For PDF processing issues, ensure your PDFs are text-based and not scanned images
- If the vector store isn't loading, check file permissions and storage space
