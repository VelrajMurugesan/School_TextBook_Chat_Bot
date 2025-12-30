# 📘 School Textbook AI Chatbot

An intelligent RAG (Retrieval-Augmented Generation) chatbot designed to help students and teachers interact with textbook content using AI. The chatbot uses persistent FAISS vector indexing for fast retrieval and local HuggingFace models for privacy-focused, offline-capable question answering.

## 🎥 Demo Video

**A complete demo video is available in [Release v1](https://github.com/VelrajMurugesan/School_TextBook_Chat_Bot/releases/tag/v1.0)** - Watch it to see the chatbot in action!

## ✨ Features

- **📚 Multi-PDF Support**: Upload and index multiple textbook PDFs
- **💾 Persistent Indexing**: FAISS index is cached locally - no need to re-upload textbooks
- **🎯 Smart Retrieval**: Uses cosine similarity with confidence scoring
- **👥 Dual Modes**: 
  - **Student Mode**: Simple, student-friendly explanations
  - **Teacher Mode**: Detailed explanations with examples
- **🔍 Source Highlighting**: See exactly which pages and sections contain the answer
- **📊 Confidence Metrics**: Get accuracy confidence scores for each answer
- **🔒 Privacy-Focused**: Runs entirely locally using HuggingFace models
- **⚡ Fast Performance**: O(log n) retrieval complexity with FAISS

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: google/flan-t5-base (local, offline)
- **PDF Processing**: PyPDFLoader, langchain-text-splitters
- **Python**: 3.12 compatible

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VelrajMurugesan/School_TextBook_Chat_Bot.git
   cd School_TextBook_Chat_Bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app_persistent_faiss_rag_fixed.py
   ```

The app will open in your default web browser at `http://localhost:8501`

## 🚀 Usage

### First Time Setup

1. **Upload PDFs**: Click "Upload textbook PDF" and select your textbook PDF files
2. **Wait for Indexing**: The first upload will index the PDFs (this may take a few minutes)
3. **Index is Cached**: Subsequent runs will automatically load the cached index - no need to re-upload!

### Asking Questions

1. **Select Mode**: Choose between "Student" or "Teacher" mode
2. **Type Your Question**: Enter your question in the chat input
3. **Get Answers**: The chatbot will:
   - Retrieve relevant passages from your textbooks
   - Generate an answer based on the context
   - Show confidence score and source highlights
   - Display page references for verification

### Features Explained

- **Confidence Score**: Indicates how well the retrieved content matches your question (0-100%)
- **Source Highlights**: Shows the exact sentences from your textbooks that contain relevant information
- **Page References**: See which PDF and page number each answer comes from

## 📋 Requirements

See `requirements.txt` for the complete list. Key dependencies:

- `streamlit>=1.32.0`
- `langchain-community>=0.0.34`
- `langchain-text-splitters>=0.0.1`
- `faiss-cpu>=1.7.4`
- `sentence-transformers>=2.6.1`
- `transformers>=4.39.0`
- `torch>=2.1.0`

## 🔧 Configuration

The app uses the following default settings (configurable in the code):

- **Chunk Size**: 600 characters
- **Chunk Overlap**: 200 characters
- **Top-K Retrieval**: 10 documents
- **Similarity Threshold**: 0.30
- **Word Overlap Threshold**: 1

## 📁 Project Structure

```
School_TextBook_Chat_Bot/
├── app_persistent_faiss_rag_fixed.py  # Main application file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── faiss_index_cached/                 # Cached FAISS index (auto-generated)
├── pdf_hash.txt                        # PDF hash for cache validation
└── README.md                           # This file
```

## 🎯 How It Works

1. **PDF Processing**: PDFs are loaded and split into chunks using RecursiveCharacterTextSplitter
2. **Embedding**: Each chunk is converted to embeddings using sentence-transformers
3. **Indexing**: Embeddings are stored in a FAISS vector database for fast similarity search
4. **Query Processing**: 
   - User question is embedded
   - Similar chunks are retrieved using cosine similarity
   - Top-K most relevant chunks are selected
5. **Answer Generation**: 
   - Retrieved chunks are used as context
   - Local LLM (flan-t5-base) generates the answer
   - Confidence is calculated based on similarity scores
6. **Response Display**: Answer, confidence, and source highlights are shown to the user

## 🔒 Privacy & Security

- **100% Local Processing**: All AI models run locally on your machine
- **No Data Transmission**: Your PDFs and questions never leave your computer
- **Cached Index**: FAISS index is stored locally for fast access
- **No External APIs**: No calls to OpenAI, Anthropic, or other cloud services

## 🐛 Troubleshooting

### Index Not Loading
- Delete the `faiss_index_cached/` folder and `pdf_hash.txt` to rebuild the index
- Ensure you have write permissions in the project directory

### Slow Performance
- First-time indexing can take several minutes for large PDFs
- Subsequent queries are fast due to cached index
- Consider using GPU if available (modify requirements to use `faiss-gpu`)

### Out of Syllabus Errors
- The chatbot only answers questions based on uploaded textbook content
- Ensure your question relates to the uploaded PDFs
- Try rephrasing your question with different keywords

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Velraj Murugesan**

## 🙏 Acknowledgments

- HuggingFace for providing open-source models
- LangChain for document processing utilities
- Facebook AI Research for FAISS
- Streamlit for the web framework

---

**Note**: For a complete walkthrough and demonstration, check out the [demo video in Release v1](https://github.com/VelrajMurugesan/School_TextBook_Chat_Bot/releases/tag/v1.0)!

