"""
Analysis of the text on any article
:Positive if closure to 1
:Negative if closure to -1
:Neutral if 0
"""
from textblob import TextBlob
from newspaper import Article
import nltk

# Download the 'punkt' tokenizer for sentence splitting
nltk.download('punkt')

# Yov can provide the article URL or text

url = "https://en.wikipedia.org/wiki/Science"
article = Article(url)


article.download()
article.parse()

text = article.text
print("Article Text:")

blob = TextBlob(text)

summary = ' '.join([str(sentence) for sentence in blob.sentences[:5]])
print("\nArticle Summary (First 5 sentences):")
print(summary)

# Perform sentiment analysis on the summary
sentiments = blob.sentiment.polarity  # Polarity score between -1 and 1
print("\nSentiment Polarity:", sentiments)
