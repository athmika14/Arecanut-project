import React from "react";
import { Link, useLocation } from "react-router-dom";
import "../styles/Navbar.css";

function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/" className="brand-link">
          <span className="brand-icon">🌿</span>
          <span className="brand-name">ArecaNut AI</span>
        </Link>
      </div>

      <ul className="navbar-links">
        <li>
          <Link
            to="/"
            className={`nav-link ${location.pathname === "/" ? "active" : ""}`}
          >
            Home
          </Link>
        </li>
        <li>
          <Link
            to="/disease"
            className={`nav-link ${location.pathname === "/disease" ? "active" : ""}`}
          >
            Disease Detection
          </Link>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;
