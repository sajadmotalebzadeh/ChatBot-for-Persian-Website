import streamlit as st
import os
import json
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import pandas as pd
import fitz  # PyMuPDF
import docx
import io

from contextlib import redirect_stdout

# --- Configuration and API Key Handling ---
def get_hf_api_key():

    """
    Retrieves the Hugging Face API key from Streamlit secrets.
    """
    try:

        api_key = st.secrets["HUGGINGFACE_API_KEY"]
        return api_key
    except (KeyError, Exception) as e:
        st.error("🛑 **خطای کلید API:** کلید API Hugging Face شما یافت نشد.")
        st.error("لطفاً فایل .streamlit/secrets.toml را ساخته و کلید خود را در آن قرار دهید.")
        st.code("HUGGINGFACE_API_KEY = \"hf_YOUR_TOKEN_HERE\"", language="toml")
        st.stop()

# --- Model and Data Loading (Cached for performance) ---
@st.cache_resource
def get_embedding_model():
    """
    Loads and caches the SentenceTransformer model for embeddings.
    """
    st.write("در حال بارگذاری مدل امبدینگ (e5-large)... این کار فقط یک بار انجام می‌شود.")
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    return model

@st.cache_resource
def get_inference_client(_api_key):
    """
    Initializes and caches the Hugging Face Inference Client.
    """
    return InferenceClient(token=_api_key)

@st.cache_data(show_spinner=False)
def load_knowledge_base(filepath="knowledge_base.json"):
    """
    Loads the pre-processed knowledge base from the JSON file.
    """

    if not os.path.exists(filepath):
        st.error(f"فایل پایگاه دانش '{filepath}' یافت نشد.")
        st.error("لطفاً ابتدا اسکریپت 'embedding_generator.py' را برای پردازش فایل‌های خود اجرا کنید.")
        st.stop()

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- RAG and Data Analysis Functions ---
def find_relevant_context(question_embedding, knowledge_base, top_k=5):
    """
    Finds the most relevant text chunks and their source files.
    """

    doc_embeddings = np.array([item['embedding'] for item in knowledge_base])
    question_embedding = np.array(question_embedding)
    similarities = np.dot(doc_embeddings, question_embedding)
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    return [knowledge_base[i] for i in top_k_indices]

def generate_rag_answer(client, chat_history, context_chunks):
    """Generates a standard RAG answer for text-based questions, now with chat history."""
    system_prompt = st.session_state.rag_system_prompt
    context_text = ""
    source_files = set()
   
    for chunk in context_chunks:
        context_text += f"Content from '{chunk['source']}':\n{chunk['content']}\n\n"
        source_files.add(chunk['source'])
    current_user_prompt = chat_history[-1]['content']
    sanitized_history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history[:-1]]
    messages_for_api = [{"role": "system", "content": system_prompt}]
    messages_for_api.extend(sanitized_history)
    messages_for_api.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {current_user_prompt}"})
   
    try:
        response = client.chat_completion(messages_for_api, model="meta-llama/Llama-4-Scout-17B-16E-Instruct", max_tokens=1024)
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        st.error(f"خطا در تولید پاسخ RAG: {e}")
        return "متاسفانه در تولید پاسخ خطایی رخ داد."

def generate_pandas_code(client, chat_history, df_columns):
    """Generates pandas code to answer a data analysis question, now with chat history."""
    system_prompt = f"""
You are an expert Python programmer specializing in the pandas library. Your task is to write Python code to answer the user's latest question, using the conversation history for context.
You have access to a pandas DataFrame named 'df'. The DataFrame has the following columns: {', '.join(df_columns)}.
Also , you have to know about SQL language and can detect the relationships between tables using Joins and similar columns.
**CRITICAL RULE:** All Persian column names and text values **MUST** be enclosed in single quotes (e.g., df['کشور'] == 'ایران'). Never use Persian characters like '،' in the code.
- Your code must be a single, executable line or a small script that PRINTS the final answer.
- If the question is not a data analysis question, return the exact string "FALLBACK_TO_RAG".
- Do not include any explanation or comments.
---
"""

    sanitized_history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
    messages_for_api = [{"role": "system", "content": system_prompt}]
    messages_for_api.extend(sanitized_history)

    try:
        response = client.chat_completion(messages_for_api, model="meta-llama/Meta-Llama-3-8B-Instruct", max_tokens=200)
        code = response.choices[0].message.content.strip().replace("`", "")
        if code.startswith("python"):
            code = code[6:].strip()
        return code
    except Exception as e:
        return f"print('Error generating code: {e}')"

def execute_pandas_code(df, code):
    """Safely executes the generated pandas code and captures the output."""
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            exec(code, {"df": df, "pd": pd})


        result = output.getvalue()
        return f"پاسخ محاسبه شده: **{result.strip()}**"
    except Exception as e:
        return f"خطا در اجرای کد: {e}"

# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="چت‌بات فایل‌های فارسی", page_icon="🤖", layout="wide")
    st.title("🤖 چت‌بات پرسش و پاسخ")
    st.markdown("سوالات خود را در مورد اسناد بارگذاری شده بپرسید.")
    api_key = get_hf_api_key()
    embedding_model = get_embedding_model()
    client = get_inference_client(api_key)
    knowledge_base = load_knowledge_base()
    if 'rag_system_prompt' not in st.session_state:
        st.session_state.rag_system_prompt = """
        
You are an expert AI assistant. Your goal is to answer questions based on the provided text context and the conversation history.
- Use the conversation history to understand follow-up questions.
- Answer ONLY with information found in the context.
- If the user's question is a title or header, return the entire section of text that follows it, without summarization.
- All answers must be in Persian, but if the user asked English, translate yout answer to English and answer with English.  
- If the answer is not in the context, state clearly: "اطلاعاتی در این مورد در فایل یافت نشد."
"""

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "سلام! چطور می‌توانم به شما کمک کنم؟"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("سوال خود را اینجا بنویسید..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            answer = ""
            with st.spinner("در حال تحلیل سوال... ⚙️"):
                question_with_prefix = "query: " + prompt
                question_embedding = embedding_model.encode(question_with_prefix)
                context_chunks = find_relevant_context(question_embedding, knowledge_base)
                primary_source = context_chunks[0]['source'] if context_chunks else None
                is_tabular = primary_source and primary_source.endswith(('.csv', '.xlsx'))
            if is_tabular:
                with st.spinner("منبع داده جدولی شناسایی شد. در حال تولید کد..."):
                    data_file_path = os.path.join("data", primary_source)
                    if os.path.exists(data_file_path):
                        df = pd.read_excel(data_file_path) if primary_source.endswith('.xlsx') else pd.read_csv(data_file_path)

                        code_to_execute = generate_pandas_code(client, st.session_state.messages, df.columns)
                        if "FALLBACK_TO_RAG" not in code_to_execute:
                            with st.spinner("در حال اجرای کد و محاسبه پاسخ... 📊"):
                                answer = execute_pandas_code(df, code_to_execute)
                            st.markdown(answer)
                        else:
                            with st.spinner("سوال تحلیلی نیست، در حال جستجو در محتوا... 📄"):
                                answer = generate_rag_answer(client, st.session_state.messages, context_chunks)
                            st.markdown(answer)
                    else:
                        st.error(f"فایل منبع '{primary_source}' در پوشه 'data' یافت نشد.")
                        answer = "متاسفانه فایل منبع برای تحلیل یافت نشد."
            else: # Standard RAG for non-tabular files
                with st.spinner("در حال جستجو در اسناد... 📄"):
                    answer = generate_rag_answer(client, st.session_state.messages, context_chunks)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
if __name__ == "__main__":
    main()
