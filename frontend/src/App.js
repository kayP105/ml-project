import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import NewsForm from "./components/NewsForm";
import Results from "./components/Results";
import SplashScreen from "./components/SplashScreen";
import "./App.css";

function App() {
  const [isSplashVisible, setIsSplashVisible] = useState(true);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsSplashVisible(false);
    }, 6000); // Display splash screen for 3 seconds
    return () => clearTimeout(timer);
  }, []);

  if (isSplashVisible) {
    return <SplashScreen />;
  }

  return (
    <div className="App">
      <Header />
      <main>
        <div className="content-container">
          <NewsForm onAnalyze={setResult} />
          {result && (
            <div className="results-wrapper">
              <Results title="DistilBERT Model" result={result.distilbert} />
              <Results title="TF-IDF Model" result={result.tfidf} />
            </div>
          )}
        </div>
      </main>
      <footer>
        <p>© {new Date().getFullYear()} Fake News Detector | Machine Learning Project</p>
      </footer>
    </div>
  );
}

export default App;
