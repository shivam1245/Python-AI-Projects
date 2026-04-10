# SentimentTextAnalysis

This project performs sentiment analysis on any article using the `TextBlob` library and the `newspaper3k` library to extract and analyze text from a given article. It outputs a summary of the article and provides the sentiment polarity, which is a measure of the article's sentiment.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Creating `requirements.txt`](#creating-requirements-txt)
- [Usage](#usage)
- [Sentiment Analysis](#sentiment-analysis)
- [Customization](#customization)
- [License](#license)

## Overview

This project extracts text from a given URL (article) and performs sentiment analysis to determine whether the content is positive, neutral, or negative. The sentiment score is provided in the form of a polarity value ranging from -1 to 1, where:
- **1** indicates highly positive sentiment,
- **-1** indicates highly negative sentiment,
- **0** indicates neutral sentiment.

Additionally, the program generates a short summary using the first five sentences of the article.

## Project Structure

```
.
├── main.py                     # Main script for sentiment analysis and article summary
├── requirements.txt            # List of project dependencies
└── README.md                   # Project documentation
```

## Requirements

To run this project, you will need the following Python libraries:

- `newspaper3k` - For article extraction and parsing.
- `TextBlob` - For natural language processing and sentiment analysis.
- `nltk` - Required for text tokenization (part of the `TextBlob` functionality).

These dependencies are listed in the `requirements.txt` file for easy installation.

## Setup

Ensure you have Python installed on your system. To get started, clone this repository and install the necessary libraries:

```bash
git clone <repo-url>
cd SentimentTextAnalysis
pip install -r requirements.txt
```

### Creating `requirements.txt`

To create your own `requirements.txt` file, follow these steps:

1. **Activate your virtual environment** (if you are using one):
   ```bash
   source venv/bin/activate  # For MacOS/Linux
   .\venv\Scripts\activate   # For Windows
   ```

2. **Install the required packages** for the project:
   ```bash
   pip install newspaper3k textblob nltk
   ```

3. **Freeze the installed packages into a `requirements.txt` file**:
   ```bash
   pip freeze > requirements.txt
   ```

This will create a `requirements.txt` file with the exact versions of the libraries you have installed.

Sample `requirements.txt`:
```
nltk==3.9.1
textblob==0.18.0.post0
newspaper3k==0.2.8
lxml_html_clean
```

## Usage

To analyze the sentiment of an article, follow these steps:

1. **Provide an Article URL:**
   - Update the `url` variable in the script with the URL of the article you want to analyze.
   - Example:
     ```python
     url = "https://en.wikipedia.org/wiki/Science"
     ```

2. **Run the Script:**
   - Run the Python script to download the article, extract its text, summarize the first five sentences, and analyze its sentiment.
   
   Example command:
   ```bash
   python main.py
   ```

### Sample Output:
```bash
Article Text:

<Entire article's text>

Article Summary (First 5 sentences):
<Summary of the article>

Sentiment Polarity: 0.15
```

The **polarity score** indicates the sentiment:
- **Positive sentiment** if the score is closer to **1**.
- **Negative sentiment** if the score is closer to **-1**.
- **Neutral sentiment** if the score is **0**.

## Sentiment Analysis

### How Sentiment Analysis Works:

The script uses the `TextBlob` library to perform sentiment analysis on the article's text. The `TextBlob` object takes the article text as input and calculates the sentiment polarity based on the positive or negative nature of the words in the article.

The following method extracts the sentiment polarity:
```python
sentiments = blob.sentiment.polarity
```

Polarity ranges from:
- **-1** (very negative) to
- **1** (very positive).

The script prints this sentiment polarity to the console after summarizing the article.

## Customization

You can customize the following aspects of the script:
- **Article URL:** Modify the `url` variable to analyze different articles.
- **Summary Length:** Change the number of sentences used in the summary by adjusting this line:
  ```python
  summary = ' '.join([str(sentence) for sentence in blob.sentences[:5]])
  ```
  For example, to summarize the first 10 sentences, use `[:10]` instead of `[:5]`.
