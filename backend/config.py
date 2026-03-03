import torch

# Модель Qwen открытая, подходит для 8GB GPU
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Генерация
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Устройство
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Системный prompt
SYSTEM_PROMPT = """
You are an advanced AI assistant.
Be precise and helpful.
If you don't know something, say you don't know.
"""