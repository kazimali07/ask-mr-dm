# Ask Mr. Data Modeler

Ask Mr. Data Modeler is a Streamlit application that allows users to upload PDF documents and interact with them using OpenAI's GPT-3.5-turbo model. The application processes the PDFs, creates a vector store, and enables conversational retrieval of information from the documents.

## Features

- Upload multiple PDF documents.
- Extract text from PDFs and split it into manageable chunks.
- Create a vector store from the text chunks using OpenAI embeddings.
- Interact with the documents using a conversational interface powered by GPT-3.5-turbo.
- Maintain conversation history for context-aware responses.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ask-mr-dm.git
    cd ask-mr-dm
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload your PDF documents using the sidebar.

4. Type your questions in the chat input and interact with the documents.

## Project Structure

- `app.py`: The main application file.
- `requirements.txt`: List of dependencies required for the project.
- `pdf-docs/`: Directory to store PDF documents (ensure this directory exists).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.