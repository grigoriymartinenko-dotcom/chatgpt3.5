import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
from config import *
from memory.short_memory import ShortMemory

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    trust_remote_code=True
)

print("Configuring 4bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model in 4bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
print("Model loaded successfully.")

# Инициализация short memory
short_memory = ShortMemory(max_len=10)


def build_messages(user_message: str):
    """
    Формируем сообщения для prompt:
    - системный
    - все предыдущие сообщения из short memory
    """
    # добавляем сообщение пользователя
    short_memory.add_message("user", user_message)

    # формируем полный контекст
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(short_memory.get_messages())
    return messages


def stream_generate(user_message: str):
    """
    Потоковая генерация ответа
    """
    messages = build_messages(user_message)

    # Qwen использует chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""
    for token in streamer:
        response += token
        yield token

    # Добавляем ответ ассистента в short memory
    short_memory.add_message("assistant", response)