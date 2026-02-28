"""
AI-Generated Text Detector
===========================
Detects whether text is AI-generated or human-written using
statistical and stylometric features + a trained classifier.

Features extracted:
- Average sentence length
- Vocabulary richness (type-token ratio)
- Punctuation density
- Average word length
- Perplexity proxy (bigram repetition ratio)
- Sentence length variance (AI text tends to be more uniform)
- Hapax legomena ratio (words appearing only once)
- Function word ratio
- Conjunction density
- Comma-to-sentence ratio
"""

import re
import math
import pickle
import os
import numpy as np
from collections import Counter


FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
    'either', 'neither', 'each', 'every', 'all', 'any', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
    'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how', 'what',
    'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'me',
    'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
    'it', 'its', 'they', 'them', 'their',
}


def extract_features(text):
    """
    Extract stylometric features from text for AI detection.

    Returns a numpy array of 15 features.
    """
    if not text or len(text.strip()) < 20:
        return np.zeros(15)

    text = text.strip()

    # Basic tokenization
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) > 2]
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    if not words or not sentences:
        return np.zeros(15)

    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    n_chars = len(text)
    unique_words = set(words)
    word_counts = Counter(words)

    # 1. Average sentence length (words per sentence)
    sent_lengths = [len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0

    # 2. Sentence length standard deviation (AI tends to be more uniform)
    sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

    # 3. Vocabulary richness (type-token ratio)
    ttr = len(unique_words) / n_words if n_words > 0 else 0

    # 4. Average word length
    avg_word_len = np.mean([len(w) for w in words])

    # 5. Word length standard deviation
    word_len_std = np.std([len(w) for w in words])

    # 6. Hapax legomena ratio (words appearing exactly once / total words)
    hapax = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_ratio = hapax / n_words if n_words > 0 else 0

    # 7. Function word ratio
    func_count = sum(1 for w in words if w in FUNCTION_WORDS)
    func_ratio = func_count / n_words if n_words > 0 else 0

    # 8. Punctuation density (punctuation chars / total chars)
    punct_count = sum(1 for c in text if c in '.,;:!?-()[]{}"\'"')
    punct_density = punct_count / n_chars if n_chars > 0 else 0

    # 9. Comma density (commas per sentence)
    comma_count = text.count(',')
    comma_per_sent = comma_count / n_sentences

    # 10. Conjunction density
    conjunctions = {'and', 'but', 'or', 'nor', 'yet', 'so', 'for', 'both', 'either', 'neither', 'while', 'although', 'however', 'moreover', 'furthermore', 'additionally'}
    conj_count = sum(1 for w in words if w in conjunctions)
    conj_density = conj_count / n_words if n_words > 0 else 0

    # 11. Bigram repetition ratio (repeated bigrams / total bigrams)
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(1 for b, c in bigram_counts.items() if c > 1)
    bigram_rep_ratio = repeated_bigrams / max(len(bigrams), 1)

    # 12. Sentence starter diversity (unique first words / total sentences)
    starters = [re.findall(r'\b[a-zA-Z]+\b', s)[0].lower() for s in sentences if re.findall(r'\b[a-zA-Z]+\b', s)]
    starter_diversity = len(set(starters)) / max(len(starters), 1)

    # 13. Average paragraph length (approximated by double newlines)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    avg_para_len = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else n_words

    # 14. Lexical density (content words / total words)
    lexical_density = 1.0 - func_ratio

    # 15. Text length (log-normalized)
    log_length = math.log(n_words + 1)

    return np.array([
        avg_sent_len,       # 0
        sent_len_std,       # 1
        ttr,                # 2
        avg_word_len,       # 3
        word_len_std,       # 4
        hapax_ratio,        # 5
        func_ratio,         # 6
        punct_density,      # 7
        comma_per_sent,     # 8
        conj_density,       # 9
        bigram_rep_ratio,   # 10
        starter_diversity,  # 11
        avg_para_len,       # 12
        lexical_density,    # 13
        log_length,         # 14
    ])


FEATURE_NAMES = [
    'avg_sent_len', 'sent_len_std', 'type_token_ratio', 'avg_word_len',
    'word_len_std', 'hapax_ratio', 'func_word_ratio', 'punct_density',
    'comma_per_sent', 'conj_density', 'bigram_rep_ratio', 'starter_diversity',
    'avg_para_len', 'lexical_density', 'log_length'
]


class AIDetector:
    """Detect AI-generated text using a trained classifier."""

    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), '..', 'models', 'ai_detector.pkl')

        if os.path.exists(self.model_path):
            self.load()

    def train(self, texts, labels):
        """
        Train the detector on labeled data.

        Args:
            texts: List of text strings
            labels: List of labels (1 = AI-generated, 0 = human-written)

        Returns:
            dict: Training metrics
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        # Extract features
        X = np.array([extract_features(t) for t in texts])
        y = np.array(labels)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(X_scaled, y)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')

        return {
            'cv_accuracy_mean': float(np.mean(cv_scores)),
            'cv_accuracy_std': float(np.std(cv_scores)),
            'cv_scores': cv_scores.tolist(),
            'n_samples': len(texts),
            'feature_importances': dict(zip(FEATURE_NAMES,
                                            self.model.feature_importances_.tolist()))
        }

    def predict(self, text):
        """
        Predict whether text is AI-generated.

        Returns:
            dict: prediction result with label, confidence, features
        """
        if self.model is None:
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'message': 'Model not trained. Run train_model.py first.'
            }

        features = extract_features(text)
        X = self.scaler.transform(features.reshape(1, -1))
        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]

        return {
            'label': 'AI-Generated' if pred == 1 else 'Human-Written',
            'is_ai': bool(pred == 1),
            'confidence': float(max(proba)),
            'ai_probability': float(proba[1]),
            'human_probability': float(proba[0]),
        }

    def save(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)

    def load(self):
        """Load trained model from disk."""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
