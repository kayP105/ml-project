import React from 'react';
// import './About.css';

const About = () => {
    return (
        <div className="about-container">
            <h2>About Fake News Detector</h2>
            <p>
                This application uses machine learning to analyze news articles and determine 
                their likelihood of being fake or real. The model was trained on a dataset 
                containing thousands of verified real and fake news articles.
            </p>
            <h3>How it works:</h3>
            <ul>
                <li>Paste the text of a news article in the input field</li>
                <li>Click "Analyze News" to process the text</li>
                <li>View the results showing prediction and confidence level</li>
            </ul>
            <h3>Technical Details:</h3>
            <p>
                The backend uses a Passive Aggressive Classifier with TF-IDF vectorization 
                to analyze text patterns characteristic of fake news. The model achieves 
                approximately 95% accuracy on test datasets.
            </p>
        </div>
    );
};

export default About;