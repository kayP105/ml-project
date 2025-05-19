// Results.jsx
import React from "react";

const Results = ({ title, result }) => {
  if (!result) return null;

  const isFake = result.prediction === "Fake";
  const confidenceColor =
    result.confidence > 70 ? "green" : result.confidence > 40 ? "orange" : "red";

  return (
    <div className={`results-container ${isFake ? "fake" : "real"}`}>
      <h2>{title}</h2>
      <div className="result-item">
        <span className="label">Prediction:</span>
        <span className={`value ${isFake ? "fake" : "real"}`}>
          {result.prediction} News
        </span>
      </div>
      <div className="result-item">
        <span className="label">Confidence:</span>
        <span className="value" style={{ color: confidenceColor }}>
          {result.confidence}%
        </span>
      </div>
      <div className="analyzed-text">
        <h3>Analyzed Text:</h3>
        <p>{result.text}</p>
      </div>
    </div>
  );
};

export default Results;
