ChatBot-for-Persian-Website
This project is a Persian RAG-based Chatbot built with Streamlit, Hugging Face LLMs, and SentenceTransformer embeddings. It lets users upload Persian documents ( PDF,  DOCX,  Excel,  CSV,  TXT,  JSON), automatically preprocesses them into embeddings, and then allows interactive question answering in Persian. It also support English language too.

 Persian RAG Chatbot

A **Persian language chatbot** powered by **Retrieval-Augmented Generation (RAG)**.  
It can answer questions about your uploaded documents (PDF, Word, Excel, CSV, TXT, JSON) in **Persian**.  
If the document is **tabular (CSV/Excel)**, the chatbot can even **generate and execute pandas code** automatically to compute answers.  

---
 Features

-  **Supports multiple file formats**: PDF, DOCX, TXT, JSON, CSV, Excel.
-  **Retrieval-Augmented Generation**: finds the most relevant context from your documents.
-  **Persian-first chatbot** (answers in Persian, English fallback if needed).
-  **Smart data analysis**: auto-generates `pandas` code for tabular datasets.
-  **Knowledge base**: all embeddings stored in a single `knowledge_base.json`.
-  **Web interface with Streamlit**.

---

