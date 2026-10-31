import React from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Home.css";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-container">

      {/* ── Hero Section — split layout: text left, illustration right ── */}
      <section className="hero">
        <div className="hero-text">
          <h1 className="hero-title">
            AI-Powered Arecanut Plant Health
          </h1>
          <p className="hero-subtitle">
            Upload a plant image to identify potential disease conditions using AI.
          </p>
          <p className="hero-description">
            Get an instant analysis of your arecanut plant's health. The model
            identifies disease patterns and returns a model score — helping
            you act early and protect your crop.
          </p>
          <button
            className="hero-cta-btn"
            onClick={() => navigate("/disease")}
            aria-label="Go to Disease Detection"
          >
            Check Your Plant →
          </button>
        </div>

        {/* SVG illustration of an arecanut palm plantation */}
        <div className="hero-visual" aria-hidden="true">
          <svg
            viewBox="0 0 400 400"
            xmlns="http://www.w3.org/2000/svg"
            className="hero-illustration"
          >
            {/* Sky / background circle */}
            <circle cx="200" cy="180" r="150" fill="#F7F4E9" opacity="0.15" />

            {/* Palm trunks */}
            <rect x="95"  y="150" width="10" height="220" rx="4" fill="#214E34" opacity="0.85" />
            <rect x="195" y="130" width="10" height="240" rx="4" fill="#214E34" />
            <rect x="295" y="160" width="10" height="210" rx="4" fill="#214E34" opacity="0.8" />

            {/* Palm fronds — left palm */}
            <g fill="#73966B">
              <ellipse cx="100" cy="145" rx="45" ry="8"  transform="rotate(-25 100 145)" />
              <ellipse cx="100" cy="145" rx="45" ry="8"  transform="rotate(25 100 145)" />
              <ellipse cx="100" cy="140" rx="40" ry="7"  transform="rotate(-55 100 140)" />
              <ellipse cx="100" cy="140" rx="40" ry="7"  transform="rotate(55 100 140)" />
            </g>
            {/* Arecanuts on left trunk */}
            <g fill="#C77B43">
              <circle cx="92" cy="170" r="6" />
              <circle cx="105" cy="175" r="6" />
              <circle cx="88" cy="180" r="5" />
            </g>

            {/* Palm fronds — centre palm (taller, more prominent) */}
            <g fill="#73966B">
              <ellipse cx="200" cy="125" rx="55" ry="10" transform="rotate(-20 200 125)" />
              <ellipse cx="200" cy="125" rx="55" ry="10" transform="rotate(20 200 125)" />
              <ellipse cx="200" cy="118" rx="50" ry="9"  transform="rotate(-50 200 118)" />
              <ellipse cx="200" cy="118" rx="50" ry="9"  transform="rotate(50 200 118)" />
              <ellipse cx="200" cy="112" rx="35" ry="7"  transform="rotate(-80 200 112)" />
              <ellipse cx="200" cy="112" rx="35" ry="7"  transform="rotate(80 200 112)" />
            </g>
            {/* Arecanuts on centre trunk */}
            <g fill="#C77B43">
              <circle cx="192" cy="155" r="7" />
              <circle cx="208" cy="160" r="7" />
              <circle cx="187" cy="166" r="6" />
              <circle cx="213" cy="168" r="6" />
            </g>

            {/* Palm fronds — right palm */}
            <g fill="#73966B">
              <ellipse cx="300" cy="155" rx="45" ry="8"  transform="rotate(-25 300 155)" />
              <ellipse cx="300" cy="155" rx="45" ry="8"  transform="rotate(25 300 155)" />
              <ellipse cx="300" cy="150" rx="40" ry="7"  transform="rotate(-55 300 150)" />
              <ellipse cx="300" cy="150" rx="40" ry="7"  transform="rotate(55 300 150)" />
            </g>
            {/* Arecanuts on right trunk */}
            <g fill="#C77B43">
              <circle cx="292" cy="180" r="6" />
              <circle cx="305" cy="185" r="6" />
            </g>

            {/* Ground line */}
            <rect x="0" y="370" width="400" height="30" fill="#214E34" opacity="0.12" />
          </svg>
        </div>
      </section>

      {/* ── Feature Cards ── */}
      <section className="features-section">
        <h2 className="features-heading">What it does</h2>
        <div className="features-grid">

          <div className="feature-item">
            <div className="feature-icon" aria-hidden="true">🔬</div>
            <h3 className="feature-title">Disease Identification</h3>
            <p className="feature-desc">
              Analyses the uploaded image and identifies disease conditions
              commonly affecting arecanut plants.
            </p>
          </div>

          <div className="feature-item">
            <div className="feature-icon" aria-hidden="true">📊</div>
            <h3 className="feature-title">Model Score</h3>
            <p className="feature-desc">
              Each prediction comes with a model score so you know how
              confident the model is about its result.
            </p>
          </div>

          <div className="feature-item">
            <div className="feature-icon" aria-hidden="true">⚡</div>
            <h3 className="feature-title">Fast Analysis</h3>
            <p className="feature-desc">
              Results in seconds. Just upload a clear photo of the plant
              or leaf — no special equipment needed.
            </p>
          </div>

        </div>
      </section>

      {/* ── Call to Action card ── */}
      <section className="cta-section">
        <div className="cta-card">
          <div className="cta-card-icon" aria-hidden="true">🔬</div>
          <div className="cta-card-body">
            <h3 className="cta-card-title">Disease Detection</h3>
            <p className="cta-card-desc">
              Upload an arecanut plant or leaf image. The ML model will
              analyse it and return the predicted health or disease condition
              along with a model score.
            </p>
            <ul className="cta-features">
              <li>✓ Identify common arecanut diseases</li>
              <li>✓ Model score with every prediction</li>
              <li>✓ Supports JPG, PNG, and WEBP images</li>
            </ul>
            <button
              className="card-btn disease-btn"
              onClick={() => navigate("/disease")}
              aria-label="Go to Disease Detection module"
            >
              Check Disease →
            </button>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="home-footer">
        <p>ArecaNut AI &copy; 2026</p>
      </footer>
    </div>
  );
}

export default Home;
