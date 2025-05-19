// NewsForm.jsx
import React, { useState } from "react";

const NewsForm = ({ onAnalyze }) => {
  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;

    setIsLoading(true);
    try {
      const distilbertResponse = await fetch("http://localhost:5000/predict/distilbert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const tfidfResponse = await fetch("http://localhost:5000/predict/tfidf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const distilbertResult = await distilbertResponse.json();
      const tfidfResult = await tfidfResponse.json();

      onAnalyze({ distilbert: distilbertResult, tfidf: tfidfResult });
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="news-form">
      <div className="form-group">
        <label htmlFor="news-text">Enter News Text:</label>
        <textarea
          id="news-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste news article text here..."
          rows="8"
          required
        />
      </div>
      <button type="submit" disabled={isLoading}>
        {isLoading ? "Analyzing..." : "Analyze News"}
      </button>
    </form>
  );
};

export default NewsForm;
