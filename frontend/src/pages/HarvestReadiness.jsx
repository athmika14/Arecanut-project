import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Module.css";

function HarvestReadiness() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleDragOver = (e) => e.preventDefault();

  return (
    <div className="module-page">
      <div className="module-header harvest-header">
        <button className="back-btn" onClick={() => navigate("/")}>
          ← Back
        </button>
        <h1 className="module-title">
          <span>🌾</span> Harvest Readiness
        </h1>
        <p className="module-subtitle">
          Upload an arecanut bunch image to check its maturity stage.
        </p>
      </div>

      <div className="module-body">
        {/* Upload Area */}
        <div
          className="upload-area"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {preview ? (
            <div className="preview-container">
              <img src={preview} alt="Selected preview" className="image-preview" />
              <button
                className="remove-btn"
                onClick={() => { setSelectedFile(null); setPreview(null); }}
              >
                ✕ Remove
              </button>
            </div>
          ) : (
            <label className="upload-label" htmlFor="harvest-upload">
              <div className="upload-icon">📁</div>
              <p className="upload-text">Drag &amp; drop an image here</p>
              <p className="upload-hint">or click to browse</p>
              <p className="upload-formats">Supported: JPG, PNG, WEBP</p>
            </label>
          )}
          <input
            id="harvest-upload"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="file-input"
            onChange={handleFileChange}
          />
        </div>

        {/* Analyse Button */}
        <button
          className="analyse-btn harvest-analyse-btn"
          disabled={!selectedFile}
          title={!selectedFile ? "Please upload an image first" : ""}
        >
          Check Harvest Readiness
        </button>

        {/* Model Not Ready Notice */}
        <div className="notice-box">
          <span className="notice-icon">ℹ️</span>
          <div>
            <strong>ML Model Coming Soon</strong>
            <p>
              The harvest readiness model is not yet trained. Once the model is
              ready and connected to the backend, the maturity stage prediction
              will appear here.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HarvestReadiness;
