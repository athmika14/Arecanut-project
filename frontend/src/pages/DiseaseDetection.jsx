import React, { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Module.css";

// ── Constants ──────────────────────────────────────────────────────────────
const API_URL       = "/api/disease/predict";
const MAX_MB        = 16;
const MAX_BYTES     = MAX_MB * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

// ── Helpers ────────────────────────────────────────────────────────────────

/** Round a 0–1 score to a percentage string, e.g. "91.23 %" */
function pct(score) {
  return (score * 100).toFixed(2) + " %";
}

/** Validate a File object; return an error string or null. */
function validateFile(file) {
  if (!ALLOWED_TYPES.has(file.type)) {
    return `Unsupported file type "${file.type}". Please upload a JPG, PNG, or WEBP image.`;
  }
  if (file.size > MAX_BYTES) {
    return `File is too large (${(file.size / 1048576).toFixed(1)} MB). Maximum is ${MAX_MB} MB.`;
  }
  return null;
}

// ── Disease vs. Healthy classification ────────────────────────────────────
const DISEASE_CLASSES = new Set([
  "Mahali_Koleroga",
  "Stem_bleeding",
  "bud borer",
  "stem cracking",
  "yellow leaf disease",
]);

const HEALTHY_CLASSES = new Set([
  "Healthy_Nut",
  "Healthy_Leaf",
  "Healthy_Trunk",
  "healthy_foot",
]);

/**
 * Determine whether the predicted class is a disease or a healthy condition.
 * Returns { isDisease: boolean, isHealthy: boolean, status: "disease"|"healthy"|"unknown" }.
 */
function classifyPrediction(predictedClass) {
  if (DISEASE_CLASSES.has(predictedClass)) {
    return { isDisease: true,  isHealthy: false, status: "disease"  };
  }
  if (HEALTHY_CLASSES.has(predictedClass)) {
    return { isDisease: false, isHealthy: true,  status: "healthy"  };
  }
  // Unknown class — fall back to neutral green theme
  return { isDisease: false, isHealthy: false, status: "unknown"  };
}

// ── Sub-components ─────────────────────────────────────────────────────────

function StatusBadge({ status }) {
  if (status === "disease") {
    return (
      <span className="status-badge status-badge--disease" role="status">
        <span className="status-badge-icon" aria-hidden="true">⚠️</span>
        Disease Detected
      </span>
    );
  }
  if (status === "healthy") {
    return (
      <span className="status-badge status-badge--healthy" role="status">
        <span className="status-badge-icon" aria-hidden="true">✓</span>
        No Disease Detected
      </span>
    );
  }
  return null;
}

function ResultCard({ result }) {
  const { predicted_class, confidence, all_scores, note } = result;

  // Determine disease vs. healthy from the actual predicted class
  const { status } = classifyPrediction(predicted_class);

  // Sort all_scores highest → lowest for display
  const sorted = Object.entries(all_scores).sort((a, b) => b[1] - a[1]);

  // Card-level modifier class: result-card--disease or result-card--healthy
  const cardClass = `result-card result-card--${status}`;

  return (
    <div className={cardClass} role="region" aria-label="Prediction result">

      {/* ── Header row: disease name (left) + status badge (right) ── */}
      <div className="result-header">
        <span className="result-icon" aria-hidden="true">🔬</span>
        <div className="result-header-text">
          <p className="result-label">Detected condition</p>
          <h2 className="result-class">{predicted_class}</h2>
          <p className="result-confidence">
            Model score:{" "}
            <strong>{pct(confidence)}</strong>
          </p>
        </div>
        <StatusBadge status={status} />
      </div>

      {/* ── Confidence bar for the top class ── */}
      <div className="result-confidence-bar-section">
        <div
          className="result-confidence-bar"
          role="img"
          aria-label={`Confidence ${(confidence * 100).toFixed(1)} percent`}
        >
          <div
            className="result-confidence-bar-fill"
            style={{ width: `${(confidence * 100).toFixed(1)}%` }}
          />
        </div>
      </div>

      {/* ── AI advisory note ── */}
      <p className="result-advisory">
        <span className="result-advisory-icon" aria-hidden="true">🤖</span>
        This is an AI-generated result and should not replace expert
        diagnosis by a qualified agricultural specialist.
      </p>

      {/* ── All class scores — always visible ── */}
      <div className="result-scores-section">
        <h3 className="result-scores-heading">All class scores</h3>
        <ul className="score-list" aria-label="All class scores">
          {sorted.map(([cls, score]) => {
            const isTop = cls === predicted_class;
            return (
              <li
                key={cls}
                className={isTop ? "score-item score-item--top" : "score-item"}
              >
                <span className="score-name">
                  {isTop && (
                    <span className="score-badge" aria-hidden="true">●</span>
                  )}
                  {cls}
                </span>
                <div
                  className="score-bar-wrap"
                  role="img"
                  aria-label={`${(score * 100).toFixed(1)} percent`}
                >
                  <div
                    className="score-bar"
                    style={{ width: `${Math.max(score * 100, 1).toFixed(1)}%` }}
                  />
                </div>
                <span className="score-value">{pct(score)}</span>
              </li>
            );
          })}
        </ul>
      </div>

      {/* ── Disclaimer ── */}
      <p className="result-note" aria-label="Disclaimer">
        ⚠️ {note}
      </p>
    </div>
  );
}

function ErrorBox({ message, isContentGuard = false }) {
  return (
    <div className={`error-box${isContentGuard ? " error-box--guard" : ""}`} role="alert">
      <span className="error-icon" aria-hidden="true">
        {isContentGuard ? "🌿" : "❌"}
      </span>
      <p>{message}</p>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

function DiseaseDetection() {
  const navigate = useNavigate();

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview,      setPreview]      = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [result,       setResult]       = useState(null);   // API success payload
  const [error,        setError]        = useState(null);   // string | null
  const [isGuardError, setIsGuardError] = useState(false);  // true when content guard rejected

  // ── File selection ─────────────────────────────────────────────────────

  const applyFile = useCallback((file) => {
    if (!file) return;
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError(null);
    setResult(null);
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  }, []);

  const handleFileChange = (e) => applyFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    applyFile(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleRemove = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setIsGuardError(false);
  };

  // ── API call ──────────────────────────────────────────────────────────

  const handleAnalyse = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setResult(null);
    setError(null);
    setIsGuardError(false);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(API_URL, {
        method:  "POST",
        body:    formData,
        // Do NOT set Content-Type — the browser sets it with the boundary
      });

      const data = await response.json();

      if (!response.ok) {
        // Server returned a structured error object
        const isGuard = data.rejected_by_guard === true;
        setIsGuardError(isGuard);
        setError(
  data.error ||
  data.detail ||
  `Server error (${response.status}).`
);
      } else {
        setResult(data);
      }
    } catch (networkErr) {
      setError(
        "Could not reach the backend. Make sure the Flask server is running on port 5000."
      );
    } finally {
      setLoading(false);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="module-page">
      {/* ── Header band ───────────────────────────────────────────────── */}
      <div className="module-header disease-header">
        <button className="back-btn" onClick={() => navigate("/")}>
          ← Back
        </button>
        <h1 className="module-title">
          <span aria-hidden="true">🔬</span> Disease Detection
        </h1>
        <p className="module-subtitle">
          Upload an arecanut plant or leaf image to detect diseases.
        </p>
      </div>

      {/* ── Body ──────────────────────────────────────────────────────── */}
      <div className="module-body">

        {/* Upload area */}
        <div
          className="upload-area"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          aria-label="Image upload area"
        >
          {preview ? (
            <div className="preview-container">
              <img
                src={preview}
                alt="Selected image preview"
                className="image-preview"
              />
              <button
                className="remove-btn"
                onClick={handleRemove}
                aria-label="Remove selected image"
              >
                ✕ Remove
              </button>
            </div>
          ) : (
            <label className="upload-label" htmlFor="disease-upload">
              <div className="upload-icon" aria-hidden="true">📁</div>
              <p className="upload-text">Drag &amp; drop an image here</p>
              <p className="upload-hint">or click to browse</p>
              <p className="upload-formats">Supported: JPG, PNG, WEBP (max {MAX_MB} MB)</p>
            </label>
          )}
          <input
            id="disease-upload"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="file-input"
            onChange={handleFileChange}
            aria-label="Choose image file"
          />
        </div>

        {/* Analyse button */}
        <button
          className={`analyse-btn disease-analyse-btn${loading ? " analyse-btn--loading" : ""}`}
          onClick={handleAnalyse}
          disabled={!selectedFile || loading}
          aria-busy={loading}
          title={!selectedFile ? "Please upload an image first" : ""}
        >
          {loading ? "Analysing…" : "Analyse for Disease"}
        </button>

        {/* Error */}
        {error && <ErrorBox message={error} isContentGuard={isGuardError} />}

        {/* Result */}
        {result && <ResultCard result={result} />}

        {/* Initial hint — only shown before any interaction */}
        {!result && !error && !loading && (
          <div className="notice-box">
            <span className="notice-icon" aria-hidden="true">ℹ️</span>
            <div>
              <strong>How it works</strong>
              <p>
                Upload a clear photo of an arecanut plant or leaf. The model
                will classify it into one of 9 conditions and return a
                confidence score.
              </p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

export default DiseaseDetection;
