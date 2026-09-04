"use client";

import { useEffect, useRef, useState } from "react";

const API = "http://127.0.0.1:8000";

type Evidence = {
  type?: string;
  path?: string;
};

type TraceStep = {
  stage?: string;
  status?: string;
  tool?: string;
  model?: string;
};

type AnalysisResult = {
  answer?: string;
  confidence?: number;
  task?: string;
  tool?: string;
  model?: string;
  selected_tools?: string[];
  evidence?: Evidence[];
  trace?: TraceStep[];
  fusion?: {
    architecture?: string;
    optical_encoder?: string;
    sar_encoder?: string;
    reasoning_model?: string;
  };
};

function getUploadedPath(data: any): string {
  const first = data?.files?.[0];

  if (typeof first === "string") {
    return first;
  }

  if (first?.path) {
    return first.path;
  }

  if (first?.file_path) {
    return first.file_path;
  }

  throw new Error("Backend did not return an uploaded file path.");
}

function evidenceLabel(type?: string) {
  if (!type) return "Evidence";

  return type
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function evidenceUrl(path?: string) {
  if (!path) return "";

  return `${API}/artifact/${encodeURI(path)}`;
}

export default function Home() {
  const opticalInput = useRef<HTMLInputElement>(null);
  const sarInput = useRef<HTMLInputElement>(null);

  const [backendOnline, setBackendOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  const [modelBusy, setModelBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  const [opticalPath, setOpticalPath] = useState("");
  const [sarPath, setSarPath] = useState("");

  const [opticalName, setOpticalName] = useState("");
  const [sarName, setSarName] = useState("");

  const [query, setQuery] = useState(
    "Use the optical and SAR images together to identify built-up areas and water-covered regions. Explain the relevant evidence and provide confidence."
  );

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function refreshStatus() {
    try {
      const [healthResponse, modelResponse] = await Promise.all([
        fetch(`${API}/health`, { cache: "no-store" }),
        fetch(`${API}/model-status`, { cache: "no-store" }),
      ]);

      if (!healthResponse.ok) {
        throw new Error("Backend unavailable.");
      }

      setBackendOnline(true);

      if (modelResponse.ok) {
        const modelData = await modelResponse.json();
        setModelLoaded(Boolean(modelData?.loaded));
      } else {
        setModelLoaded(false);
      }
    } catch {
      setBackendOnline(false);
      setModelLoaded(false);
    }
  }

  useEffect(() => {
    refreshStatus();

    const timer = setInterval(refreshStatus, 3000);

    return () => clearInterval(timer);
  }, []);

  async function loadModel() {
    setError("");
    setMessage("Loading GeoQwen onto the GPU...");
    setModelBusy(true);

    try {
      const response = await fetch(`${API}/model-load`, {
        method: "POST",
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || "GeoQwen loading failed.");
      }

      await refreshStatus();

      setMessage("GeoQwen is loaded and ready.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "GeoQwen loading failed."
      );
    } finally {
      setModelBusy(false);
    }
  }

  async function unloadModel() {
    setError("");
    setMessage("Unloading GeoQwen from GPU memory...");
    setModelBusy(true);

    try {
      const response = await fetch(`${API}/model-unload`, {
        method: "POST",
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || "GeoQwen unload failed.");
      }

      await refreshStatus();

      setMessage("GeoQwen has been unloaded.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "GeoQwen unload failed."
      );
    } finally {
      setModelBusy(false);
    }
  }

  async function uploadImage(
    file: File,
    type: "optical" | "sar"
  ) {
    setError("");
    setMessage(
      `Uploading ${type === "optical" ? "optical" : "SAR"} image...`
    );
    setUploadBusy(true);

    try {
      const formData = new FormData();
      formData.append("files", file);

      const response = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || "Image upload failed.");
      }

      const uploadedPath = getUploadedPath(data);

      if (type === "optical") {
        setOpticalPath(uploadedPath);
        setOpticalName(file.name);
      } else {
        setSarPath(uploadedPath);
        setSarName(file.name);
      }

      setMessage(
        `${type === "optical" ? "Optical" : "SAR"} image uploaded successfully.`
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Image upload failed."
      );
    } finally {
      setUploadBusy(false);
    }
  }

  async function analyze() {
    setError("");
    setMessage("");

    if (!backendOnline) {
      setError("Backend is offline.");
      return;
    }

    if (!modelLoaded) {
      setError("Load GeoQwen before running analysis.");
      return;
    }

    if (!query.trim()) {
      setError("Enter an analysis query.");
      return;
    }

    if (!opticalPath || !sarPath) {
      setError("Upload both Optical and SAR images.");
      return;
    }

    setResult(null);
    setAnalyzing(true);
    setMessage("SatQuery AI is analyzing the imagery...");

    try {
      const response = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query.trim(),
          image_paths: [opticalPath, sarPath],
        }),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(data?.detail || "Analysis failed.");
      }

      setResult(data);
      setMessage("Analysis completed successfully.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Analysis failed."
      );
    } finally {
      setAnalyzing(false);
    }
  }

  const evidence = result?.evidence ?? [];

  return (
    <main className="app-shell">

      {/* HEADER */}
      <header className="topbar">
        <div>
          <div className="brand">SATQUERY AI</div>

          <div className="subtitle">
            Interactive Vision-Language Assistant for Multimodal Remote Sensing
          </div>
        </div>

        <div className="system-status">

          <span
            className={`status-dot ${
              backendOnline ? "online" : "offline"
            }`}
          />

          <span>
            Backend {backendOnline ? "Online" : "Offline"}
          </span>

          <span className="divider" />

          <span
            className={`status-dot ${
              modelLoaded ? "online" : "offline"
            }`}
          />

          <span>
            GeoQwen {modelLoaded ? "Loaded" : "Not Loaded"}
          </span>

        </div>
      </header>

      {/* INPUT AREA */}
      <section className="input-grid">

        {/* IMAGE INPUT */}
        <div className="panel">

          <div className="panel-title">
            <span>01</span>
            IMAGE INPUT
          </div>

          <div className="upload-card">

            <div>
              <div className="label">
                OPTICAL IMAGE
              </div>

              <div className="filename">
                {opticalName || "No optical image selected"}
              </div>
            </div>

            <button
              className="upload-button"
              onClick={() => opticalInput.current?.click()}
              disabled={uploadBusy}
            >
              {uploadBusy ? "Uploading..." : "Upload Optical"}
            </button>

            <input
              ref={opticalInput}
              type="file"
              hidden
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              onChange={(event) => {
                const file = event.target.files?.[0];

                if (file) {
                  uploadImage(file, "optical");
                }

                event.target.value = "";
              }}
            />

          </div>

          <div className="upload-card">

            <div>
              <div className="label">
                SAR IMAGE
              </div>

              <div className="filename">
                {sarName || "No SAR image selected"}
              </div>
            </div>

            <button
              className="upload-button"
              onClick={() => sarInput.current?.click()}
              disabled={uploadBusy}
            >
              {uploadBusy ? "Uploading..." : "Upload SAR"}
            </button>

            <input
              ref={sarInput}
              type="file"
              hidden
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              onChange={(event) => {
                const file = event.target.files?.[0];

                if (file) {
                  uploadImage(file, "sar");
                }

                event.target.value = "";
              }}
            />

          </div>

          {/* MODEL CONTROL */}
          <div className="model-box">

            <div>
              <div className="label">
                GEOQWEN MODEL
              </div>

              <div
                className={`model-state ${
                  modelLoaded ? "ready" : "not-ready"
                }`}
              >
                {modelLoaded
                  ? "READY ON GPU"
                  : "MODEL NOT LOADED"}
              </div>
            </div>

            <div className="model-buttons">

              <button
                className="load-button"
                onClick={loadModel}
                disabled={
                  !backendOnline ||
                  modelLoaded ||
                  modelBusy
                }
              >
                {modelBusy && !modelLoaded
                  ? "Loading..."
                  : "Load GeoQwen"}
              </button>

              <button
                className="unload-button"
                onClick={unloadModel}
                disabled={
                  !backendOnline ||
                  !modelLoaded ||
                  modelBusy
                }
              >
                {modelBusy && modelLoaded
                  ? "Unloading..."
                  : "Unload GeoQwen"}
              </button>

            </div>

          </div>

        </div>

        {/* QUERY */}
        <div className="panel">

          <div className="panel-title">
            <span>02</span>
            ANALYSIS QUERY
          </div>

          <textarea
            className="query-box"
            value={query}
            onChange={(event) =>
              setQuery(event.target.value)
            }
            placeholder="Ask SatQuery AI about the imagery..."
          />

          <button
            className="analyze-button"
            onClick={analyze}
            disabled={
              analyzing ||
              uploadBusy ||
              !backendOnline ||
              !modelLoaded
            }
          >
            {analyzing
              ? "Analyzing Imagery..."
              : "Analyze Imagery"}
          </button>

          {message && (
            <div className="message-box">
              {message}
            </div>
          )}

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

        </div>

      </section>

      {/* RESULT */}
      <section className="panel">

        <div className="panel-title">
          <span>03</span>
          ANALYSIS RESULT
        </div>

        <div className="answer-label">
          ANSWER
        </div>

        <div className="answer-box">
          {result?.answer ||
            "Run an analysis to receive an evidence-grounded answer from SatQuery AI."}
        </div>

        <div className="metrics">

          <div className="metric">
            <span>CONFIDENCE</span>
            <strong>
              {result?.confidence !== undefined
                ? `${Math.round(result.confidence * 100)}%`
                : "--"}
            </strong>
          </div>

          <div className="metric">
            <span>TASK</span>
            <strong>
              {result?.task ||
                result?.tool ||
                "--"}
            </strong>
          </div>

          <div className="metric">
            <span>MODEL</span>
            <strong>
              {result?.fusion?.reasoning_model ||
                result?.model ||
                "GeoQwen"}
            </strong>
          </div>

        </div>

      </section>

      {/* EVIDENCE */}
      <section className="panel">

        <div className="panel-title">
          <span>04</span>
          EVIDENCE
        </div>

        <div className="evidence-grid">

          {evidence.length > 0 ? (

            evidence.map((item, index) => {

              const src = evidenceUrl(item.path);
              const isRaster =
                item.path?.toLowerCase().endsWith(".tif") ||
                item.path?.toLowerCase().endsWith(".tiff");

              return (
                <div
                  className="evidence-card"
                  key={`${item.type}-${index}`}
                >

                  <div className="evidence-title">
                    {evidenceLabel(item.type)}
                  </div>

                  {src && !isRaster ? (
                    <img
                      className="evidence-image"
                      src={src}
                      alt={evidenceLabel(item.type)}
                    />
                  ) : (
                    <div className="evidence-content">

                      <strong>
                        {item.type === "optical_image"
                          ? "Optical GeoTIFF"
                          : item.type === "sar_image"
                          ? "SAR GeoTIFF"
                          : "Raster Evidence"}
                      </strong>

                      <span>
                        Generated by backend
                      </span>

                    </div>
                  )}

                </div>
              );
            })

          ) : (

            <>
              <div className="evidence-card">
                <div className="evidence-title">
                  OPTICAL
                </div>

                <div className="evidence-content">
                  Optical evidence appears here
                </div>
              </div>

              <div className="evidence-card">
                <div className="evidence-title">
                  SAR
                </div>

                <div className="evidence-content">
                  SAR evidence appears here
                </div>
              </div>

              <div className="evidence-card">
                <div className="evidence-title">
                  AGREEMENT MAP
                </div>

                <div className="evidence-content">
                  Fusion agreement map
                </div>
              </div>

              <div className="evidence-card">
                <div className="evidence-title">
                  DISAGREEMENT MAP
                </div>

                <div className="evidence-content">
                  Fusion disagreement map
                </div>
              </div>
            </>

          )}

        </div>

      </section>

      {/* TRACE */}
      <section className="panel">

        <div className="panel-title">
          <span>05</span>
          EXECUTION TRACE
        </div>

        <div className="trace-list">

          {result?.trace && result.trace.length > 0 ? (

            result.trace.map((step, index) => (

              <div
                className="trace-row"
                key={`${step.stage}-${index}`}
              >

                <span className="check">
                  ✓
                </span>

                <strong>
                  {step.stage || `Step ${index + 1}`}
                </strong>

                <span
                  className={
                    step.status === "success"
                      ? "complete"
                      : "trace-status"
                  }
                >
                  {step.status || "completed"}
                </span>

              </div>

            ))

          ) : (

            <div className="trace-empty">
              Execution trace will appear after analysis.
            </div>

          )}

        </div>

      </section>

      <footer>
        SPACE TECHNOLOGY • SIH26167 • SatQuery AI
      </footer>

    </main>
  );
}
