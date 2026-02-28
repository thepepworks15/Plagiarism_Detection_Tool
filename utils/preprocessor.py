"""
Text Preprocessing Module
=========================
Handles tokenization, stopword removal, stemming, and text normalization
for plagiarism detection.

Algorithm Explanation:
1. Text Cleaning: Remove special characters, extra whitespace, URLs, emails
2. Tokenization: Split text into individual words (tokens)
3. Stopword Removal: Filter out common words (the, is, at, etc.) that don't
   carry semantic meaning
4. Stemming: Reduce words to their root form (running -> run) using
   Porter Stemmer algorithm
5. Lemmatization: Alternative to stemming that produces valid dictionary words
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


class TextPreprocessor:
    """Preprocesses text for plagiarism comparison."""

    def __init__(self, language='english'):
        self.language = language
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))

    def clean_text(self, text):
        """Remove special characters, URLs, emails, and normalize whitespace."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        # Remove special characters but keep sentence boundaries
        text = re.sub(r'[^\w\s.,!?;:\-]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        """Split text into individual word tokens."""
        return word_tokenize(text.lower())

    def sentence_tokenize(self, text):
        """Split text into sentences."""
        return sent_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove common stopwords from token list."""
        return [t for t in tokens if t not in self.stop_words and t not in string.punctuation]

    def stem(self, tokens):
        """Apply Porter Stemming to reduce words to root form."""
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize(self, tokens):
        """Apply lemmatization to produce valid dictionary root words."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text, use_stemming=True):
        """
        Full preprocessing pipeline.

        Steps:
            1. Clean raw text
            2. Tokenize into words
            3. Remove stopwords
            4. Apply stemming or lemmatization

        Args:
            text: Raw input text
            use_stemming: If True use stemming, else use lemmatization

        Returns:
            list: Processed tokens
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = self.stem(tokens)
        else:
            tokens = self.lemmatize(tokens)
        return tokens

    def get_sentences(self, text):
        """Get cleaned sentences from text."""
        cleaned = self.clean_text(text)
        return self.sentence_tokenize(cleaned)

    def get_ngrams(self, tokens, n=3):
        """Generate n-grams from token list."""
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
