# QueryPDF
Turn Documents into Conversations.
AI PDF Chatbot using RAG

An AI-powered PDF chatbot that allows users to upload PDF documents and ask questions about their content using natural language.

## 🚀 Features

- Upload PDF documents.
- Extract text from PDF files.
- Split text into smaller chunks.
- Generate text embeddings using Sentence Transformers.
- Store and search embeddings using FAISS.
- Retrieve relevant document chunks using semantic similarity search.
- Generate context-based answers using the Gemini API.
- Display source page numbers with answers.
- Interactive chat interface built with Streamlit.

## 🛠️ Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Google Gemini API
- NumPy

## ⚙️ Installation

1. Clone the repository:

   `git clone YOUR_REPOSITORY_URL`

2. Navigate to the project directory:

   `cd PDF-Chatbot`

3. Install dependencies:

   `pip install -r requirements.txt`

4. Create a `.env` file and add your Gemini API key:

   `GEMINI_API_KEY=your_api_key_here`

5. Run the application:

   `streamlit run app.py`

## 🧠 How It Works

1. User uploads a PDF.
2. Text is extracted using PyPDF.
3. Extracted text is divided into overlapping chunks.
4. Sentence Transformers converts chunks into embeddings.
5. FAISS stores the embeddings and retrieves relevant chunks.
6. Gemini uses the retrieved context to generate answers.
7. The chatbot displays the answer with source page references.

## 🔮 Future Improvements

- Support multiple PDF uploads.
- Add OCR support for scanned PDFs.
- Implement conversation memory.
- Improve retrieval accuracy.
- Add persistent vector storage.
- Deploy the application online.

## 👩‍💻 Author

Dimple Chauhan
