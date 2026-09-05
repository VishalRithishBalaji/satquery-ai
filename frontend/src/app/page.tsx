"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Cpu,
  Image as ImageIcon,
  Info,
  Loader2,
  Power,
  PowerOff,
  RefreshCw,
  Route,
  Satellite,
  ShieldCheck,
  Sparkles,
  Waves,
  XCircle,
} from "lucide-react";
import { PolarAngleAxis, RadialBar, RadialBarChart } from "recharts";

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
  task?: string;
  tools?: string[];
  operation?: string;
  attributes?: string[];
  confidence?: number;
  error?: string;
};

type TaskIntent = {
  operation?: string;
  attributes?: string[];
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
  task_intent?: TaskIntent;
  query_grounding_score?: number;
  low_query_sensitivity?: boolean;
  retried_for_query_sensitivity?: boolean;
  fusion?: {
    architecture?: string;
    optical_encoder?: string;
    sar_encoder?: string;
    reasoning_model?: string;
  };
};

const QUERY_PRESETS: { label: string; query: string }[] = [
  {
    label: "Optical + SAR fusion",
    query:
      "Use the optical and SAR images together to identify built-up areas and water-covered regions. Explain the relevant evidence and provide confidence.",
  },
  {
    label: "Built-up areas",
    query: "Identify built-up areas in this scene and explain the supporting evidence.",
  },
  {
    label: "Water bodies",
    query: "Identify water-covered regions in this scene and explain the supporting evidence.",
  },
  {
    label: "What changed?",
    query: "What changed between these images? Focus on any new or removed structures.",
  },
];

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

function traceDetail(step: TraceStep): string {
  const parts: string[] = [];

  if (step.task) parts.push(`task: ${step.task}`);
  if (step.operation) parts.push(`operation: ${step.operation}`);
  if (step.attributes && step.attributes.length > 0) parts.push(`focus: ${step.attributes.join(", ")}`);
  if (step.tool) parts.push(`tool: ${step.tool}`);
  if (step.model) parts.push(`model: ${step.model}`);
  if (step.confidence !== undefined) parts.push(`confidence: ${Math.round(step.confidence * 100)}%`);
  if (step.error) parts.push(`error: ${step.error}`);

  return parts.join(" · ");
}

function traceIcon(step: TraceStep) {
  if (step.status === "error") return <XCircle size={16} />;

  switch (step.stage) {
    case "validation":
      return <ShieldCheck size={16} />;
    case "routing":
      return <Route size={16} />;
    case "tool_execution":
      return <Cpu size={16} />;
    default:
      return <CheckCircle2 size={16} />;
  }
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

function ConfidenceGauge({ value }: { value?: number }) {
  const hasValue = value !== undefined;
  const pct = hasValue ? Math.round((value as number) * 100) : 0;
  const data = [{ name: "confidence", value: pct }];

  return (
    <div className="gauge-wrap">
      <RadialBarChart
        width={116}
        height={108}
        cx={58}
        cy={58}
        innerRadius={40}
        outerRadius={54}
        barSize={9}
        data={data}
        startAngle={90}
        endAngle={-270}
      >
        <defs>
          <linearGradient id="confidenceGradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#58d7ff" />
            <stop offset="100%" stopColor="#7b91ff" />
          </linearGradient>
        </defs>
        <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
        <RadialBar background={{ fill: "#0d2036" }} dataKey="value" cornerRadius={8} fill="url(#confidenceGradient)" />
      </RadialBarChart>
      <div className="gauge-value">{hasValue ? `${pct}%` : "--"}</div>
    </div>
  );
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

  const [opticalPreview, setOpticalPreview] = useState("");
  const [sarPreview, setSarPreview] = useState("");

  const [draggingOptical, setDraggingOptical] = useState(false);
  const [draggingSar, setDraggingSar] = useState(false);

  const [query, setQuery] = useState(QUERY_PRESETS[0].query);

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [resultVersion, setResultVersion] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

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

  useEffect(() => {
    return () => {
      if (opticalPreview) URL.revokeObjectURL(opticalPreview);
      if (sarPreview) URL.revokeObjectURL(sarPreview);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

    const previewUrl = URL.createObjectURL(file);

    if (type === "optical") {
      if (opticalPreview) URL.revokeObjectURL(opticalPreview);
      setOpticalPreview(previewUrl);
      setOpticalName(file.name);
    } else {
      if (sarPreview) URL.revokeObjectURL(sarPreview);
      setSarPreview(previewUrl);
      setSarName(file.name);
    }

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
      } else {
        setSarPath(uploadedPath);
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

  function handleDrop(event: React.DragEvent<HTMLDivElement>, type: "optical" | "sar") {
    event.preventDefault();
    if (type === "optical") setDraggingOptical(false);
    else setDraggingSar(false);

    const file = event.dataTransfer.files?.[0];
    if (file) uploadImage(file, type);
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
      setResultVersion((v) => v + 1);
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

  async function copyAnswer() {
    if (!result?.answer) return;

    try {
      await navigator.clipboard.writeText(result.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard access can be denied by the browser; not worth surfacing as an error.
    }
  }

  const evidence = result?.evidence ?? [];

  return (
    <main className="app-shell">

      {/* HEADER */}
      <header className="topbar">
        <div className="brand-row">
          <div className="brand-icon">
            <Satellite size={22} />
          </div>

          <div>
            <div className="brand">SatQuery AI</div>

            <div className="subtitle">
              Interactive Vision-Language Assistant for Multimodal Remote Sensing
            </div>
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
            Image Input
          </div>

          <div
            className={`dropzone ${draggingOptical ? "dragging" : ""} ${opticalName ? "filled" : ""}`}
            onClick={() => opticalInput.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDraggingOptical(true);
            }}
            onDragLeave={() => setDraggingOptical(false)}
            onDrop={(event) => handleDrop(event, "optical")}
          >
            {opticalPreview ? (
              <img className="dropzone-thumb" src={opticalPreview} alt="Optical preview" />
            ) : (
              <div className="dropzone-icon">
                <ImageIcon size={22} />
              </div>
            )}

            <div className="dropzone-text">
              <div className="label">OPTICAL IMAGE</div>
              <strong>{opticalName || "Drop image here or click to browse"}</strong>
              <span>GeoTIFF, PNG or JPG</span>
            </div>

            {uploadBusy && !opticalPath && <Loader2 className="spin" size={18} />}

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

          <div
            className={`dropzone ${draggingSar ? "dragging" : ""} ${sarName ? "filled" : ""}`}
            onClick={() => sarInput.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDraggingSar(true);
            }}
            onDragLeave={() => setDraggingSar(false)}
            onDrop={(event) => handleDrop(event, "sar")}
          >
            {sarPreview ? (
              <img className="dropzone-thumb" src={sarPreview} alt="SAR preview" />
            ) : (
              <div className="dropzone-icon">
                <Waves size={22} />
              </div>
            )}

            <div className="dropzone-text">
              <div className="label">SAR IMAGE</div>
              <strong>{sarName || "Drop image here or click to browse"}</strong>
              <span>GeoTIFF, PNG or JPG</span>
            </div>

            {uploadBusy && !sarPath && <Loader2 className="spin" size={18} />}

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
                <Cpu size={14} />
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
                {modelBusy && !modelLoaded ? <Loader2 className="spin" size={15} /> : <Power size={15} />}
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
                {modelBusy && modelLoaded ? <Loader2 className="spin" size={15} /> : <PowerOff size={15} />}
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
            Analysis Query
          </div>

          <div className="preset-row">
            {QUERY_PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                className="preset-chip"
                onClick={() => setQuery(preset.query)}
              >
                {preset.label}
              </button>
            ))}
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
            {analyzing ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
            {analyzing
              ? "Analyzing Imagery..."
              : "Analyze Imagery"}
          </button>

          {message && (
            <div className="message-box">
              <Info size={15} />
              {message}
            </div>
          )}

          {error && (
            <div className="error-box">
              <AlertTriangle size={15} />
              {error}
            </div>
          )}

        </div>

      </section>

      {/* RESULT */}
      <section className="panel">

        <div className="panel-title">
          <span>03</span>
          Analysis Result
        </div>

        <div key={resultVersion} className="result-fade">

          <div className="answer-header">
            <div className="answer-label">ANSWER</div>

            {result?.answer && (
              <button className="icon-button" onClick={copyAnswer} type="button">
                <Copy size={13} />
                {copied ? "Copied" : "Copy"}
              </button>
            )}
          </div>

          {(result?.task_intent?.operation || (result?.task_intent?.attributes?.length ?? 0) > 0 || result?.low_query_sensitivity) && (
            <div className="query-focus">
              <span className="query-focus-label">DETECTED FOCUS</span>

              {result?.task_intent?.operation && (
                <span className="badge operation">
                  {evidenceLabel(result.task_intent.operation)}
                </span>
              )}

              {result?.task_intent?.attributes?.map((attribute) => (
                <span className="badge" key={attribute}>
                  {evidenceLabel(attribute)}
                </span>
              ))}

              {result?.low_query_sensitivity && (
                <span className="badge warning">
                  <AlertTriangle size={11} />
                  Low query sensitivity - verify answer
                </span>
              )}

              {result?.retried_for_query_sensitivity && (
                <span className="badge">
                  <RefreshCw size={11} />
                  Auto-retried for query focus
                </span>
              )}
            </div>
          )}

          <div className="answer-box">
            {result?.answer ||
              "Run an analysis to receive an evidence-grounded answer from SatQuery AI."}
          </div>

          <div className="metrics">

            <div className="metric gauge-card">
              <span>CONFIDENCE</span>
              <ConfidenceGauge value={result?.confidence} />
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

            <div className="metric">
              <span>QUERY GROUNDING</span>
              <strong>
                {result?.query_grounding_score !== undefined
                  ? `${Math.round(result.query_grounding_score * 100)}%`
                  : "--"}
              </strong>
            </div>

          </div>

        </div>

      </section>

      {/* EVIDENCE */}
      <section className="panel">

        <div className="panel-title">
          <span>04</span>
          Evidence
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
                    <ImageIcon size={13} />
                    {evidenceLabel(item.type)}
                  </div>

                  {src && !isRaster ? (
                    <a className="evidence-image-link" href={src} target="_blank" rel="noreferrer">
                      <img
                        className="evidence-image"
                        src={src}
                        alt={evidenceLabel(item.type)}
                      />
                    </a>
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
                  <ImageIcon size={13} />
                  OPTICAL
                </div>

                <div className="evidence-content">
                  Optical evidence appears here
                </div>
              </div>

              <div className="evidence-card">
                <div className="evidence-title">
                  <Waves size={13} />
                  SAR
                </div>

                <div className="evidence-content">
                  SAR evidence appears here
                </div>
              </div>

              <div className="evidence-card">
                <div className="evidence-title">
                  <CheckCircle2 size={13} />
                  AGREEMENT MAP
                </div>

                <div className="evidence-content">
                  Fusion agreement map
                </div>
              </div>

              <div className="evidence-card">
                <div className="evidence-title">
                  <XCircle size={13} />
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
          Execution Trace
        </div>

        <div className="trace-list">

          {result?.trace && result.trace.length > 0 ? (

            result.trace.map((step, index) => {
              const failed = step.status === "error";
              const detail = traceDetail(step);

              return (
                <div
                  className="trace-row"
                  key={`${step.stage}-${index}`}
                >

                  <span className={`check${failed ? " failed" : ""}`}>
                    {traceIcon(step)}
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

                  {detail && (
                    <div className="trace-detail">
                      {detail}
                    </div>
                  )}

                </div>
              );
            })

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
