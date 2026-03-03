from collections import deque

class ShortMemory:
    def __init__(self, max_len=10):
        """
        Храним последние N сообщений (user + assistant)
        """
        self.memory = deque(maxlen=max_len)

    def add_message(self, role: str, content: str):
        """
        Добавляем сообщение в память
        role: "user" или "assistant"
        """
        self.memory.append({"role": role, "content": content})

    def get_messages(self):
        """
        Возвращаем все сообщения для prompt
        """
        return list(self.memory)

    def clear(self):
        """
        Очистка памяти
        """
        self.memory.clear()