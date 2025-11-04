# 💬 Mini RAG Chatbot

An intelligent document-based Q&A chatbot powered by Retrieval-Augmented Generation (RAG) technology.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Live Demo

🔗 [Try it here](YOUR-DEPLOYMENT-URL-HERE) *(update after deployment)*

## ✨ Features

- **💬 Conversational Memory** - Remembers last 10 messages for natural conversations
- **📚 Document Q&A** - Ask questions, get answers from your documents
- **🔄 Easy Updates** - Click a button to rebuild index after adding documents
- **📊 Adjustable Retrieval** - Control how many document chunks to use (Top-K)
- **🎨 Modern UI** - Professional, animated interface with gradient design
- **🤖 Smart Detection** - Distinguishes casual chat from information requests

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** OpenAI GPT-4o-mini
- **Vector Database:** FAISS
- **Embeddings:** Sentence Transformers (paraphrase-MiniLM-L6-v2)
- **Document Processing:** PyPDF

## 📋 Requirements

- Python 3.9+
- OpenAI API key

## 🚀 Local Development

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/mini-rag-chatbot.git
cd mini-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

### 4. Add your documents

Place your documents in the `data/docs/` folder:
- Supported formats: PDF, TXT, MD
- Optional: Create `data/faq.json` for Q&A pairs

### 5. Build the vector store

```bash
python ingest.py
```

### 6. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 📁 Project Structure

```
mini-rag-chatbot/
├── app.py                 # Streamlit UI
├── rag.py                 # RAG logic
├── ingest.py              # Vector store builder
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (not in repo)
├── .gitignore            # Git ignore file
├── data/
│   ├── docs/             # Your documents (PDF, TXT, MD)
│   └── faq.json          # Optional Q&A pairs
└── vectorstore/          # Generated embeddings
    ├── faiss.index
    └── metadata.json
```

## 🎯 Usage

1. **Ask Questions:** Type your question in the chat input
2. **Adjust Settings:** Use sidebar to change Top-K chunks (1-10)
3. **Update Documents:** Add files to `data/docs/` and click "Rebuild Index"
4. **View Sources:** Check which documents were used for each answer

## 🔧 Configuration

### Top-K Chunks
- **K=1-2:** Focused, precise answers
- **K=4-6:** Balanced (recommended)
- **K=8-10:** Comprehensive answers with more context

### LLM Provider
- **auto:** Uses OpenAI if configured, else offline mode
- **openai:** Force OpenAI usage
- **stub:** Offline extractive mode only

## 📊 Features in Detail

### Conversation Memory
- Remembers last 10 messages
- Natural follow-up questions
- No repetitive responses

### Small Talk Detection
- Recognizes greetings and casual chat
- Routes appropriately (casual vs information)
- Natural conversation flow

### Document Processing
- Chunks documents intelligently
- 750 characters per chunk with 150 character overlap
- Supports multiple file formats

### Vector Search
- FAISS for fast similarity search
- Cosine similarity scoring
- Relevance gating to filter irrelevant results

## 🚀 Deployment

### Streamlit Community Cloud (FREE)

1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Create new app
4. Add secrets (OpenAI API key)
5. Deploy!

See [DEPLOY_STREAMLIT_CLOUD.md](DEPLOY_STREAMLIT_CLOUD.md) for detailed instructions.

### Other Options
- Docker + any cloud provider
- Heroku
- AWS/Azure/GCP
- Railway

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI](https://openai.com/)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/YOUR-USERNAME/mini-rag-chatbot](https://github.com/YOUR-USERNAME/mini-rag-chatbot)

---

⭐ Star this repo if you find it helpful!
