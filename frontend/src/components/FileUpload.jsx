import React, { useState } from "react";

export default function FileUpload({ onUpload }) {
  const [fileName, setFileName] = useState("");

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setFileName(file.name);

    const formData = new FormData();
    formData.append("file", file);

    let url = "";
    if (file.type === "application/pdf") url = "http://127.0.0.1:8000/upload/pdf";
    else if (file.type.startsWith("image/")) url = "http://127.0.0.1:8000/upload/image";
    else {
      alert("Поддерживаются только PDF и изображения");
      return;
    }

    const res = await fetch(url, { method: "POST", body: formData });
    const data = await res.json();
    alert(`Файл загружен: ${file.name}\n${data.status}`);
    onUpload(); // обновляем memory
  };

  return (
    <div className="flex items-center gap-2 mb-2">
      <input type="file" onChange={handleFileChange} className="border p-1 rounded bg-white" />
      {fileName && <span className="text-gray-700">{fileName}</span>}
    </div>
  );
}