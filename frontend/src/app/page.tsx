"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [paths, setPaths] = useState<string[]>([]);
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    const body = new FormData();
    Array.from(files).forEach((file) => body.append("files", file));
    const response = await fetch(`${API}/upload`, { method: "POST", body });
    const data = await response.json();
    setPaths(data.files ?? []);
  }

  async function analyze() {
    if (!query || !paths.length) return;
    setBusy(true);
    try {
      const response = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, image_paths: paths }),
      });
      setResult(await response.json());
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 1000, margin: "40px auto", padding: 24, fontFamily: "Arial" }}>
      <h1>SatQuery AI</h1>
      <p>Agentic remote-sensing analysis prototype</p>
      <input type="file" multiple accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(e) => upload(e.target.files)} />
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a satellite-image question..."
        rows={4}
        style={{ width: "100%", marginTop: 20, padding: 12 }}
      />
      <button onClick={analyze} disabled={busy || !paths.length || !query} style={{ marginTop: 12, padding: "10px 18px" }}>
        {busy ? "Analyzing..." : "Analyze"}
      </button>
      {result && (
        <pre style={{ marginTop: 24, padding: 16, background: "#f4f4f4", overflow: "auto" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}
