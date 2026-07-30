# 📝 AI Question Paper Generator

An **AI-powered Question Paper Generator** that automatically creates professional examination papers from uploaded **PDF** or **TXT** study materials using **Retrieval-Augmented Generation (RAG)**.

Instead of manually creating question papers, simply upload your study material, select the chapters, choose the question type, and let AI generate a well-structured exam paper in seconds.

---

## 🚀 Features

* 📄 Upload **PDF** or **TXT** documents
* 🔍 **OCR support** for scanned PDFs
* 🧠 Uses **Retrieval-Augmented Generation (RAG)** for accurate question generation
* 📚 Generate questions from:

  * Entire document
  * Specific chapter
  * Multiple chapters
* 📝 Four question modes:

  * Multiple Choice Questions (MCQs)
  * Short Questions
  * Long Questions
  * Mixed Question Paper
* 🎯 Difficulty Levels:

  * Easy
  * Medium
  * Hard
* 📖 Bloom's Taxonomy support
* 🏫 Professional exam paper formatting
* 📅 Automatically includes:

  * Institution Name
  * Subject Name
  * Exam Date
  * Time Duration
  * Total Marks
  * Chapters Covered
  * Instructions
* 📥 Export generated papers as:

  * PDF
  * TXT
* ✅ Generate Answer Keys
* ⚡ Fast semantic search using **FAISS**
* 🎨 Interactive web interface built with **Streamlit**

---

## 🧠 How It Works

```text
Upload PDF / TXT
        │
        ▼
Extract Text (OCR if needed)
        │
        ▼
Detect Chapters
        │
        ▼
Split into Smart Chunks
        │
        ▼
Generate Embeddings
        │
        ▼
Store in FAISS Vector Database
        │
        ▼
User Selects Chapters & Question Type
        │
        ▼
Retrieve Relevant Content
        │
        ▼
Groq LLM Generates Questions
        │
        ▼
Export Professional Test Paper
```

---

## 💡 Why RAG?

Unlike traditional AI systems that rely on general knowledge, this application uses **Retrieval-Augmented Generation (RAG)** to retrieve only the relevant content from the uploaded document before generating questions.

This approach helps:

* Improve accuracy
* Reduce hallucinations
* Generate document-specific questions
* Produce more reliable examination papers

---

## 🛠 Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### AI & Machine Learning

* LangChain
* Groq LLM
* Hugging Face Embeddings
* Sentence Transformers

### Vector Database

* FAISS

### Document Processing

* PyMuPDF
* OCR (Tesseract)

### Export

* ReportLab
* Text Export

---

## 📂 Project Structure

```text
AI-Test-Paper-Generator/
│
├── app.py
├── config.py
├── requirements.txt
│
├── rag/
│   ├── loader.py
│   ├── parser.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── generator.py
│   └── prompts.py
│
├── utils/
│   ├── pdf_export.py
│   └── text_export.py
│
├── uploads/
├── outputs/
└── README.md
```

---

## ✨ Example Workflow

1. Upload a PDF or TXT study material.
2. Process the document.
3. Select one or more chapters.
4. Choose the question type:

   * MCQs
   * Short Questions
   * Long Questions
   * Mixed Paper
5. Select difficulty and number of questions.
6. Generate the exam paper.
7. Download the paper as PDF or TXT.

---

## 📸 Screenshots

> Add screenshots or a GIF here to showcase:

* Document upload
* Chapter selection
* Question generation
* Generated exam paper
* PDF export

---

## 🎯 Future Improvements

* Multi-language support
* Automatic chapter detection improvements
* Image and diagram understanding
* Teacher-defined question templates
* Difficulty balancing using AI
* Student performance analytics
* Multi-document knowledge base
* Cloud deployment with authentication
* API support for integration with Learning Management Systems (LMS)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests to improve the project.

---

## ⭐ If You Like This Project

If you found this project helpful, please consider giving it a **⭐ Star** on GitHub. It motivates me to build and share more AI-powered open-source projects.

---

## 👨‍💻 Author

**Irfan Hyder**

AI Engineer | Machine Learning | Data Science | Python

Passionate about building practical AI applications using **Machine Learning, Generative AI, Retrieval-Augmented Generation (RAG), and Automation** to solve real-world problems.

If you like this project, connect with me and explore my other AI projects!
