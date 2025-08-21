import os
import json
import fitz  # PyMuPDF
import pandas as pd
import docx
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
from elasticsearch.exceptions import NotFoundError
from elasticsearch.exceptions import NotFoundError

# --- Model Loading ---
def get_embedding_model():   
    """
    Loads the SentenceTransformer model for embeddings.
    """
    print("در حال بارگذاری مدل امبدینگ (e5-large)... این کار ممکن است چند دقیقه طول بکشد.")
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    print("مدل امبدینگ با موفقیت بارگذاری شد.")
    return model

# --- Text Extraction and Chunking Functions ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    doc.close()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n\n"
    return text

def process_tabular_data(df, rows_per_chunk=20):
    chunks = []
    for i in range(0, len(df), rows_per_chunk):
        chunk_df = df.iloc[i:i + rows_per_chunk]
        chunk_str = chunk_df.to_string(index=False, header=True)
        chunks.append(chunk_str)
    return chunks

def extract_chunks_from_csv(file_path):
    df = pd.read_csv(file_path)
    return process_tabular_data(df)

def extract_chunks_from_excel(file_path):
    df = pd.read_excel(file_path)
    return process_tabular_data(df)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False)

def recursive_text_splitter(text, max_chunk_size):
    if len(text) <= max_chunk_size: return [text]
    split_pos = text.rfind('. ', 0, max_chunk_size)
    if split_pos == -1: split_pos = max_chunk_size
    chunk = text[:split_pos+1]
    remaining = text[split_pos+1:]
    return [chunk] + recursive_text_splitter(remaining, max_chunk_size)

def get_text_chunks(text, min_chunk_size=50, max_chunk_size=700):
    initial_chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) >= min_chunk_size]
    final_chunks = []
    for chunk in initial_chunks:
        if len(chunk) > max_chunk_size:
            sub_chunks = recursive_text_splitter(chunk, max_chunk_size)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    return final_chunks

# --- Metadata Extraction ---
def get_file_metadata(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return {"نوع فایل": "CSV", "تعداد رکوردها": len(df), "ستون‌ها": df.columns.tolist()}
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            return {"نوع فایل": "Excel", "تعداد رکوردها": len(df), "ستون‌ها": df.columns.tolist()}
    except Exception:
        return None

    return None

# --- Main Processing Logic ---
def main():
    """
    Scans the 'data' directory, processes all supported files,
    and saves the embeddings and metadata to a single JSON file.
    """
    data_dir = "data"
    output_file = "knowledge_base.json"
  
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"پوشه '{data_dir}' ایجاد شد. لطفاً فایل‌های خود را در آن قرار دهید و دوباره اسکریپت را اجرا کنید.")
        return
    
    files_to_process = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    if not files_to_process:
        print(f"پوشه '{data_dir}' خالی است. هیچ فایلی برای پردازش وجود ندارد.")
        return
    
    embedding_model = get_embedding_model()
    knowledge_base = []
    
    print(f"\nشروع پردازش {len(files_to_process)} فایل از پوشه '{data_dir}'...")
    
    for filename in tqdm(files_to_process, desc="Processing files"):
        file_path = os.path.join(data_dir, filename)
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        try:
            metadata = get_file_metadata(file_path)
           
            text_extractors = {
                ".pdf": extract_text_from_pdf, ".docx": extract_text_from_docx,
                ".txt": extract_text_from_txt, ".json": extract_text_from_json,
            }
            chunk_extractors = {
                ".csv": extract_chunks_from_csv, ".xlsx": extract_chunks_from_excel,
            }
            chunks = []
            if extension in chunk_extractors:
                chunks = chunk_extractors[extension](file_path)
            elif extension in text_extractors:
                text = text_extractors[extension](file_path)
                chunks = get_text_chunks(text)
            else:
                print(f"\nفرمت فایل {filename} پشتیبانی نمی‌شود. از این فایل صرف نظر می‌شود.")
                continue
            if not chunks:
                print(f"\nهیچ محتوای مناسبی در فایل {filename} یافت نشد.")
                continue
            passage_chunks = ["passage: " + chunk for chunk in chunks]
            embeddings = embedding_model.encode(passage_chunks, show_progress_bar=False)
            for i, chunk in enumerate(chunks):
                knowledge_base.append({
                    'source': filename,
                    'content': chunk,
                    'embedding': embeddings[i].tolist(),
                    'metadata': metadata
                })
        except Exception as e:
            print(f"\nخطا در پردازش فایل {filename}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ پردازش با موفقیت انجام شد.")
    print(f"پایگاه دانش با {len(knowledge_base)} بخش در فایل '{output_file}' ذخیره شد.")
    print("اکنون می‌توانید 'app.py' را برای شروع چت‌بات اجرا کنید.")

if __name__ == "__main__":
    main()

