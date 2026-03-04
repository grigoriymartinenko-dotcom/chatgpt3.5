import io
import os
import pickle
import torch
import pytesseract
import pdfplumber
import fitz
import easyocr
import numpy as np
import cv2
import faiss
import re
from sympy import symbols, Eq, solve, sympify, expand

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# =====================================================
# TESSERACT PATH (WINDOWS)
# =====================================================
pytesseract.pytesseract.tesseract_cmd = r"F:\AI\Tesseract\tesseract.exe"

# =====================================================
# FASTAPI INIT
# =====================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD LLM (QWEN)
# =====================================================
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Model loaded.")

# =====================================================
# LOAD EMBEDDING MODEL (RAG)
# =====================================================
print("Loading embedding model...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-base")
print("Embedding model ready.")

# =====================================================
# INIT EASY OCR
# =====================================================
print("Loading EasyOCR...")
reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
print("EasyOCR ready.")

# =====================================================
# MEMORY
# =====================================================
short_memory = []

# =====================================================
# RAG ENGINE
# =====================================================
class RAGEngine:
    def __init__(self):
        self.dimension = 768
        self.index = None
        self.text_chunks = []

    def chunk_text(self, text, chunk_size=800, overlap=150):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def build_index(self, text):
        self.text_chunks = self.chunk_text(text)
        if not self.text_chunks:
            print("No text to index.")
            return
        embeddings = embed_model.encode(
            ["passage: " + c for c in self.text_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype("float32"))
        print("RAG index built. Total vectors:", self.index.ntotal)

    def search(self, query, top_k=3):
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = embed_model.encode(
            ["query: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])
        return results

    def save(self, path="rag_store"):
        if self.index is None:
            return
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.text_chunks, f)
        print("RAG saved.")

    def load(self, path="rag_store"):
        try:
            self.index = faiss.read_index(os.path.join(path, "index.faiss"))
            with open(os.path.join(path, "chunks.pkl"), "rb") as f:
                self.text_chunks = pickle.load(f)
            print("RAG loaded from disk.")
        except:
            print("No existing RAG index found.")

rag = RAGEngine()
rag.load()

# =====================================================
# Функция для пошагового решения школьных выражений
# =====================================================
x = symbols('x')

# =====================================================
# Функция для пошагового решения выражений с промежуточными результатами
# =====================================================
def solve_and_explain(text):
    """
    Пошаговое объяснение арифметики с промежуточными результатами.
    Убирает лишние символы, обрабатывает выражения и показывает шаги.
    """
    explanations = []

    # Чистим текст от лишних символов, оставляем цифры, +-*/^(),x и пробелы
    clean_text = re.sub(r"[^0-9\+\-\*/\^\(\)x=.\n]", "", text)

    # Разбиваем на выражения по переносу строки или '='
    parts = re.split(r'[=\n]+', clean_text)

    for part in parts:
        expr_str = part.strip()
        if not expr_str:
            continue

        try:
            # Простое число
            if re.fullmatch(r'-?\d+(\.\d+)?', expr_str):
                explanations.append(f"Число: {expr_str} → это результат предыдущего вычисления.")
                continue

            # Выражение с x
            if 'x' in expr_str:
                if '=' in expr_str:
                    left, right = expr_str.split('=', 1)
                    x = symbols('x')
                    eq = Eq(parse_expr(left), parse_expr(right))
                    sol = solve(eq, x)
                    explanations.append(
                        f"Уравнение: {expr_str}\nРешение: {sol}"
                    )
                continue

            # Арифметическое выражение
            tokens = re.findall(r'\d+\.\d+|\d+|[\+\-\*/\(\)]', expr_str)
            tmp_tokens = tokens[:]

            # 1️⃣ умножение и деление
            i = 0
            while i < len(tmp_tokens):
                if tmp_tokens[i] in ('*', '/'):
                    a = float(tmp_tokens[i-1])
                    b = float(tmp_tokens[i+1])
                    res = a*b if tmp_tokens[i]=='*' else a/b
                    tmp_tokens[i-1:i+2] = [str(res)]
                    i = 0
                else:
                    i += 1
            step1 = " ".join(tmp_tokens)

            # 2️⃣ сложение и вычитание
            i = 0
            while i < len(tmp_tokens):
                if tmp_tokens[i] in ('+', '-'):
                    a = float(tmp_tokens[i-1])
                    b = float(tmp_tokens[i+1])
                    res = a+b if tmp_tokens[i]=='+' else a-b
                    tmp_tokens[i-1:i+2] = [str(res)]
                    i = 0
                else:
                    i += 1
            final_result = tmp_tokens[0]

            explanations.append(
                f"Выражение: {expr_str}\n"
                f"Пошаговое объяснение:\n"
                f"1. Сначала умножение и деление → {step1}\n"
                f"2. Затем сложение и вычитание → {final_result}\n"
                f"Результат: {final_result}"
            )

        except Exception as e:
            explanations.append(f"{expr_str} → не удалось разобрать ({e})")

    return "\n\n".join(explanations)

# =====================================================
# CHAT
# =====================================================
@app.post("/chat")
async def chat(data: dict):
    user_message = data.get("message", "")
    short_memory.append(f"user: {user_message}")
    prompt = "\n".join(short_memory) + "\nassistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def generate():
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response.split("assistant:")[-1].strip()
        short_memory.append(f"assistant: {answer}")
        yield f"data: {answer}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# =====================================================
# RAG UPLOAD (PDF / IMAGE / TXT)
# =====================================================
@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename.lower()
        text = ""

        if filename.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    # 🔹 сразу вычисляем и объясняем формулы
                    page_text = solve_and_explain(page_text)
                    text += page_text + "\n"

            if not text.strip():
                pdf_document = fitz.open(stream=contents, filetype="pdf")
                for page in pdf_document:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    page_text = pytesseract.image_to_string(img, lang="rus+eng")
                    page_text = solve_and_explain(page_text)
                    text += page_text + "\n"

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(contents))
            page_text = pytesseract.image_to_string(image, lang="rus+eng").strip()
            if not page_text:
                result = reader.readtext(np.array(image), detail=0)
                page_text = "\n".join(result)
            page_text = solve_and_explain(page_text)
            text += page_text

        else:
            page_text = contents.decode("utf-8")
            page_text = solve_and_explain(page_text)
            text += page_text

        if not text.strip():
            return JSONResponse({"error": "Файл пустой или текст не распознан"}, status_code=400)

        rag.build_index(text)
        rag.save()
        return JSONResponse({"status": "Document indexed successfully"})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =====================================================
# RAG CHAT с объяснением
# =====================================================
@app.post("/rag/chat")
async def rag_chat(data: dict):
    question = data.get("message", "")
    contexts = rag.search(question, top_k=4)
    combined_context = "\n\n".join(contexts)

    # 🔹 решаем и объясняем формулы из контекста
    combined_context = solve_and_explain(combined_context)

    prompt = f"""
You are an AI assistant.
Answer ONLY using the provided context.
Explain the solution STEP BY STEP.
If the answer is not in the context, say you don't know.

Context:
{combined_context}

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def generate():
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()
        yield f"data: {answer}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# =====================================================
# IMAGE OCR
# =====================================================
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image, lang="rus+eng").strip()
        if len(text) < 5:
            result = reader.readtext(np.array(image), detail=0)
            text = "\n".join(result).strip()
        if not text:
            text = "[OCR ничего не распознал]"

        # 🔹 вычисляем формулы и объясняем
        text = solve_and_explain(text)

        short_memory.append(f"system (image): {text}")
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =====================================================
# PDF OCR / EXTRACT
# =====================================================
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = ""

        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                page_text = solve_and_explain(page_text)
                text += page_text + "\n"

        if not text.strip():
            pdf_document = fitz.open(stream=contents, filetype="pdf")
            for page_number, page in enumerate(pdf_document):
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                page_text = pytesseract.image_to_string(img, lang="rus+eng")
                if not page_text.strip():
                    result = reader.readtext(np.array(img), detail=0)
                    page_text = "\n".join(result)
                page_text = solve_and_explain(page_text)
                text += f"\n--- Page {page_number+1} ---\n{page_text}"

        if not text.strip():
            text = "[PDF пустой или текст не распознан]"

        short_memory.append(f"system (pdf): {text}")
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)