import React from 'react';
// import './Header.css';

const Header = () => {
    return (
        <header className="header">
            <img src="/news.gif" alt="Decoration" className="header-gif before" />
            <h1>Fake News Detector</h1>
            <p>Analyze news articles for authenticity using AI</p>
        </header>
    );
};

export default Header;
