import React, { useState } from "react";

export default function ChatInput({ sendMessage, clearMemory }) {
  const [text, setText] = useState("");

  const handleSend = () => {
    if (!text.trim()) return;
    sendMessage(text.trim());
    setText("");
  };

  return (
    <div className="flex gap-2 mt-2">
      <input
        className="flex-1 p-2 border rounded"
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Введите сообщение..."
      />
      <button
        className="px-3 py-2 bg-blue-600 text-white rounded"
        onClick={handleSend}
      >
        Send
      </button>
      <button
        className="px-3 py-2 bg-red-600 text-white rounded"
        onClick={clearMemory}
      >
        Clear Memory
      </button>
    </div>
  );
}