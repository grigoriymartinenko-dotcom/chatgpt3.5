import React, { useEffect, useRef } from "react";

export default function ChatHistory({ messages }) {
  const containerRef = useRef(null);

  // Авто-прокрутка вниз при добавлении нового сообщения
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto p-2 border bg-white rounded mb-2"
    >
      {messages.map((msg, idx) => (
        <div
          key={idx}
          className={`my-1 ${
            msg.role === "user" ? "text-right text-blue-700" : "text-left text-gray-800"
          }`}
        >
          <strong>{msg.role}:</strong>{" "}
          <span className={msg.role === "assistant" ? "animate-pulse" : ""}>
            {msg.content}
          </span>
        </div>
      ))}
    </div>
  );
}