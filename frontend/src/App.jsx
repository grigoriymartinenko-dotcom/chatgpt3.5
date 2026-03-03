import React, { useState } from "react";

export default function App() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);
  const [file, setFile] = useState(null);

  // =========================
  // CHAT (STREAM FIXED)
  // =========================
  const sendMessage = async () => {
    if (!message) return;

    const userMessage = { role: "user", content: message };
    setChat((prev) => [...prev, userMessage]);
    setMessage("");

    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let assistantText = "";

    // добавляем пустое сообщение ассистента
    setChat((prev) => [...prev, { role: "assistant", content: "" }]);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      for (let line of lines) {
        if (line.startsWith("data: ")) {
          const clean = line.replace("data: ", "").trim();
          assistantText += clean + " ";

          // обновляем последнее сообщение ассистента
          setChat((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: "assistant",
              content: assistantText,
            };
            return updated;
          });
        }
      }
    }
  };

  // =========================
  // UPLOAD IMAGE
  // =========================
  const uploadImage = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/upload_image", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    setChat((prev) => [
      ...prev,
      {
        role: "system",
        content: `📷 OCR TEXT:\n${data.text}`,
      },
    ]);
  };

  // =========================
  // UPLOAD PDF
  // =========================
  const uploadPDF = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/upload_pdf", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    setChat((prev) => [
      ...prev,
      {
        role: "system",
        content: `📄 PDF TEXT:\n${data.text}`,
      },
    ]);
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Local Qwen Chat + Memory + File Upload</h1>

      <textarea
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Введите сообщение..."
        rows={4}
        style={{ width: "100%" }}
      />

      <br />
      <button onClick={sendMessage}>Отправить</button>

      <hr />

      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={uploadPDF}>Загрузить PDF</button>
      <button onClick={uploadImage}>Загрузить изображение</button>

      <hr />

      <h2>Chat</h2>
      {chat.map((msg, index) => (
        <div key={index} style={{ marginBottom: 10 }}>
          <b>{msg.role}:</b>
          <pre style={{ whiteSpace: "pre-wrap" }}>{msg.content}</pre>
        </div>
      ))}
    </div>
  );
}