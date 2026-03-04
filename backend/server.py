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
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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

    def chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def build_index(self, text):
        print("Chunking document...")
        self.text_chunks = self.chunk_text(text)

        print("Creating embeddings...")
        embeddings = embed_model.encode(
            ["passage: " + chunk for chunk in self.text_chunks],
            convert_to_numpy=True,
            show_progress_bar=True
        )

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))

        print("RAG index built.")

    def search(self, query, top_k=3):
        if self.index is None:
            return []

        query_embedding = embed_model.encode(
            ["query: " + query],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results

    def save(self, path="rag_store"):
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
# CHAT (STANDARD)
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
                max_new_tokens=300,
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
# RAG UPLOAD DOCUMENT
# =====================================================
@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = contents.decode("utf-8")

        rag.build_index(text)
        rag.save()

        return JSONResponse({"status": "Document indexed successfully"})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =====================================================
# RAG CHAT
# =====================================================
@app.post("/rag/chat")
async def rag_chat(data: dict):

    question = data.get("message", "")

    contexts = rag.search(question, top_k=3)

    combined_context = "\n\n".join(contexts)

    prompt = f"""
You are an AI assistant.
Answer ONLY using the provided context.
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
                max_new_tokens=300,
                do_sample=False
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

        gray = image.convert("L")
        gray = gray.resize((gray.width * 2, gray.height * 2))
        gray_np = np.array(gray)
        _, thresh = cv2.threshold(gray_np, 150, 255, cv2.THRESH_BINARY)

        text_tesseract = pytesseract.image_to_string(
            thresh,
            lang="rus+eng",
            config="--oem 3 --psm 6"
        ).strip()

        final_text = text_tesseract

        if len(text_tesseract) < 5:
            img_np = np.array(image)
            result = reader.readtext(img_np, detail=0)
            text_easy = "\n".join(result).strip()
            final_text = text_easy if text_easy else "[OCR ничего не распознал]"

        short_memory.append(f"system (image): {final_text}")

        return JSONResponse({"text": final_text})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =====================================================
# PDF EXTRACT
# =====================================================
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = ""

        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            pdf_document = fitz.open(stream=contents, filetype="pdf")

            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))

                page_text = pytesseract.image_to_string(
                    img,
                    lang="rus+eng"
                )

                if not page_text.strip():
                    img_np = np.array(img)
                    result = reader.readtext(img_np, detail=0)
                    page_text = "\n".join(result)

                text += f"\n--- Page {page_number+1} ---\n"
                text += page_text

        if not text.strip():
            text = "[PDF пустой или текст не распознан]"

        short_memory.append(f"system (pdf): {text}")

        return JSONResponse({"text": text})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)