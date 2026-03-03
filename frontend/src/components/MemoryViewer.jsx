import React from "react";

export default function MemoryViewer({ memory }) {
  return (
    <div className="p-2 border rounded mb-2 bg-gray-50 h-32 overflow-y-auto">
      <strong>Short Memory:</strong>
      {memory.length === 0 && <div className="text-gray-500">Memory is empty</div>}
      {memory.map((msg, idx) => (
        <div key={idx} className="text-sm text-gray-700">
          <strong>{msg.role}:</strong> {msg.content}
        </div>
      ))}
    </div>
  );
}