import io
import torch
import pytesseract
import pdfplumber
import fitz  # PyMuPDF

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================
# WINDOWS TESSERACT PATH
# =============================
pytesseract.pytesseract.tesseract_cmd = r"F:\AI\Tesseract\tesseract.exe"

# =============================
# FASTAPI
# =============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# LOAD MODEL
# =============================
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

print("Model loaded.")

# =============================
# MEMORY
# =============================
short_memory = []

# =============================
# CHAT STREAM
# =============================
@app.post("/chat")
async def chat(data: dict):

    user_message = data.get("message")
    short_memory.append(f"user: {user_message}")

    prompt = "\n".join(short_memory) + "\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def generate():
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response.split("assistant:")[-1].strip()

        short_memory.append(f"assistant: {answer}")

        yield f"data: {answer}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# =====================================================
# IMAGE OCR (RUS + ENG)
# =====================================================
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        print("Image loaded")

        text = pytesseract.image_to_string(
            image,
            lang="rus+eng"
        )

        if not text.strip():
            text = "[OCR ничего не распознал]"

        print("OCR RESULT:")
        print(text)

        short_memory.append(f"system (image): {text}")

        return JSONResponse({"text": text})

    except Exception as e:
        print("OCR ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# =====================================================
# PDF SMART EXTRACT
# 1) пробуем текстовый слой
# 2) если пусто → OCR каждой страницы
# =====================================================
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = ""

        # ---------- 1. Пробуем текстовый слой ----------
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # ---------- 2. Если пусто → OCR ----------
        if not text.strip():
            print("No text layer found. Running OCR on PDF pages...")

            pdf_document = fitz.open(stream=contents, filetype="pdf")

            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                pix = page.get_pixmap(dpi=300)

                img = Image.open(io.BytesIO(pix.tobytes()))
                page_text = pytesseract.image_to_string(
                    img,
                    lang="rus+eng"
                )

                text += f"\n--- Page {page_number+1} ---\n"
                text += page_text

        if not text.strip():
            text = "[PDF пустой или не удалось распознать текст]"

        print("PDF RESULT:")
        print(text[:500])

        short_memory.append(f"system (pdf): {text}")

        return JSONResponse({"text": text})

    except Exception as e:
        print("PDF ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=500)