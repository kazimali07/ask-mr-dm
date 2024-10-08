# Ask Mr. Data Modeler

## Overview
"Ask Mr. Data Modeler" is a Streamlit application that allows users to interact with a conversational AI model. The application processes PDF documents to extract text and creates a vector store for efficient retrieval of information. Users can ask questions, and the AI will respond based on the content of the uploaded PDFs.

## Features
- Load PDF documents from a specified directory.
- Extract text from PDF files and split it into manageable chunks.
- Create a FAISS vector store for efficient text retrieval.
- Engage in a conversational interface with the AI model.
- Upload additional PDF documents through the sidebar.

## Requirements
- Python 3.x
- Streamlit
- PyPDF2
- Langchain
- Certifi
- Dotenv

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload PDF documents using the sidebar or load them from the specified directory.

4. Type your questions in the chat interface and receive responses from the AI model.

## Notes
- Ensure that the PDF documents are in the correct format for text extraction.
- The application saves the vector store to a file named `vector_store.pkl` for future use.

## License
This project is licensed under the MIT License.
