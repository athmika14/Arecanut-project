import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import DiseaseDetection from "./pages/DiseaseDetection";

function App() {
  return (
    <Router>
      <Navbar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/disease" element={<DiseaseDetection />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
