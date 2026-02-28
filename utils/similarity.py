"""
Similarity Detection Algorithms Module
=======================================
Implements multiple plagiarism detection algorithms for comparing text documents.

Algorithms Implemented:
1. Cosine Similarity - Measures angle between TF-IDF vectors
2. Jaccard Similarity - Measures set overlap between token sets
3. N-gram Analysis - Compares overlapping word sequences
4. Semantic Similarity - Uses sentence transformers for meaning-based comparison

Algorithm Explanations:
-----------------------

COSINE SIMILARITY:
    Converts documents to TF-IDF vectors (Term Frequency - Inverse Document Frequency)
    and measures the cosine of the angle between them.
    - TF(t,d) = (Number of times term t appears in document d) / (Total terms in d)
    - IDF(t) = log(Total documents / Documents containing term t)
    - Cosine = (A . B) / (||A|| * ||B||)
    - Range: 0 (completely different) to 1 (identical)
    - Best for: Overall document similarity

JACCARD SIMILARITY:
    Measures the overlap between two sets of tokens.
    - J(A,B) = |A intersection B| / |A union B|
    - Range: 0 (no common tokens) to 1 (identical token sets)
    - Best for: Detecting copy-paste with minor word changes

N-GRAM ANALYSIS:
    Compares sequences of N consecutive words (shingles).
    - Generates all possible n-grams from both documents
    - Calculates overlap ratio
    - Higher N = more specific matching (fewer false positives)
    - Best for: Detecting reordered text and paraphrasing

SEMANTIC SIMILARITY:
    Uses pre-trained sentence transformer models to capture meaning.
    - Encodes sentences into dense vector embeddings
    - Compares embeddings using cosine similarity
    - Detects paraphrased content that lexical methods miss
    - Best for: Detecting intelligent paraphrasing
"""

import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'config.json')


def _load_config():
    """Load optimized weights and threshold from config.json if available."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return None


class SimilarityAnalyzer:
    """Computes similarity between documents using multiple algorithms."""

    def __init__(self):
        self._semantic_model = None
        self._config = _load_config()

    @property
    def semantic_model(self):
        """Lazy-load semantic model only when needed."""
        if self._semantic_model is None:
            from sentence_transformers import SentenceTransformer
            self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._semantic_model

    # ------------------------------------------------------------------
    # 1. Cosine Similarity (TF-IDF based)
    # ------------------------------------------------------------------
    def cosine_similarity(self, text1, text2):
        """
        Calculate cosine similarity using TF-IDF vectors.

        Process:
        1. Create TF-IDF matrix from both documents
        2. Compute cosine of angle between document vectors
        3. Return similarity score

        Args:
            text1, text2: Raw text strings

        Returns:
            float: Similarity score between 0 and 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = sklearn_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])

    # ------------------------------------------------------------------
    # 2. Jaccard Similarity
    # ------------------------------------------------------------------
    def jaccard_similarity(self, tokens1, tokens2):
        """
        Calculate Jaccard similarity between two token sets.

        Formula: |A ∩ B| / |A ∪ B|

        Args:
            tokens1, tokens2: Lists of preprocessed tokens

        Returns:
            float: Similarity score between 0 and 1
        """
        set1 = set(tokens1)
        set2 = set(tokens2)

        if not set1 and not set2:
            return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # 3. N-gram Analysis
    # ------------------------------------------------------------------
    def ngram_similarity(self, tokens1, tokens2, n=3):
        """
        Calculate similarity based on overlapping n-grams.

        Process:
        1. Generate all n-grams (sequences of n consecutive tokens)
        2. Find common n-grams between documents
        3. Calculate overlap ratio

        Args:
            tokens1, tokens2: Lists of preprocessed tokens
            n: Size of n-grams (default 3 for trigrams)

        Returns:
            dict: Contains similarity score and matching n-grams
        """
        def get_ngrams(tokens, size):
            return [tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]

        ngrams1 = set(get_ngrams(tokens1, n))
        ngrams2 = set(get_ngrams(tokens2, n))

        if not ngrams1 and not ngrams2:
            return {'score': 0.0, 'matching_ngrams': [], 'total_ngrams_doc1': 0, 'total_ngrams_doc2': 0}

        common = ngrams1.intersection(ngrams2)
        total = ngrams1.union(ngrams2)

        score = len(common) / len(total) if total else 0.0

        return {
            'score': score,
            'matching_ngrams': [' '.join(ng) for ng in list(common)[:50]],
            'total_ngrams_doc1': len(ngrams1),
            'total_ngrams_doc2': len(ngrams2),
            'common_count': len(common)
        }

    # ------------------------------------------------------------------
    # 4. Semantic Similarity (Sentence Transformers)
    # ------------------------------------------------------------------
    def semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity using sentence transformer embeddings.

        Process:
        1. Split documents into sentences
        2. Encode sentences into dense vector embeddings
        3. Compute pairwise cosine similarity between all sentence pairs
        4. Identify highly similar sentence pairs (potential plagiarism)
        5. Calculate overall semantic similarity score

        Args:
            text1, text2: Raw text strings

        Returns:
            dict: Contains overall score and matched sentence pairs
        """
        from nltk.tokenize import sent_tokenize

        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)

        if not sentences1 or not sentences2:
            return {'score': 0.0, 'matched_pairs': []}

        # Encode all sentences
        embeddings1 = self.semantic_model.encode(sentences1, convert_to_numpy=True)
        embeddings2 = self.semantic_model.encode(sentences2, convert_to_numpy=True)

        # Compute pairwise similarity matrix
        sim_matrix = sklearn_cosine(embeddings1, embeddings2)

        # Find highly similar sentence pairs (threshold > 0.7)
        matched_pairs = []
        threshold = 0.7
        for i in range(len(sentences1)):
            for j in range(len(sentences2)):
                if sim_matrix[i][j] > threshold:
                    matched_pairs.append({
                        'source_sentence': sentences1[i],
                        'matched_sentence': sentences2[j],
                        'similarity': float(sim_matrix[i][j])
                    })

        # Overall score: average of max similarities for each source sentence
        max_similarities = np.max(sim_matrix, axis=1)
        overall_score = float(np.mean(max_similarities))

        # Sort matched pairs by similarity (descending)
        matched_pairs.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'score': min(overall_score, 1.0),
            'matched_pairs': matched_pairs[:30],
            'total_sentences_doc1': len(sentences1),
            'total_sentences_doc2': len(sentences2)
        }

    # ------------------------------------------------------------------
    # Combined Analysis
    # ------------------------------------------------------------------
    def full_analysis(self, text1, text2, tokens1, tokens2, include_semantic=True):
        """
        Run all similarity algorithms and produce a combined report.

        Args:
            text1, text2: Raw text strings
            tokens1, tokens2: Preprocessed token lists
            include_semantic: Whether to include semantic analysis (slower)

        Returns:
            dict: Complete similarity analysis results
        """
        results = {
            'cosine_similarity': self.cosine_similarity(text1, text2),
            'jaccard_similarity': self.jaccard_similarity(tokens1, tokens2),
            'ngram_analysis': self.ngram_similarity(tokens1, tokens2, n=3),
        }

        if include_semantic:
            results['semantic_similarity'] = self.semantic_similarity(text1, text2)
        else:
            results['semantic_similarity'] = {'score': 0.0, 'matched_pairs': [], 'skipped': True}

        # Calculate weighted overall plagiarism score using optimized config
        if self._config and include_semantic:
            weights = self._config.get('similarity_weights_with_semantic', {
                'cosine': 0.25, 'jaccard': 0.15, 'ngram': 0.15, 'semantic': 0.45
            })
        elif self._config and not include_semantic:
            weights = self._config.get('similarity_weights', {
                'cosine': 0.20, 'jaccard': 0.30, 'ngram': 0.50, 'semantic': 0.0
            })
        elif include_semantic:
            weights = {'cosine': 0.25, 'jaccard': 0.15, 'ngram': 0.15, 'semantic': 0.45}
        else:
            weights = {'cosine': 0.20, 'jaccard': 0.30, 'ngram': 0.50, 'semantic': 0.0}

        overall = (
            weights['cosine'] * results['cosine_similarity'] +
            weights['jaccard'] * results['jaccard_similarity'] +
            weights['ngram'] * results['ngram_analysis']['score'] +
            weights['semantic'] * results['semantic_similarity']['score']
        )

        results['overall_plagiarism_score'] = round(overall * 100, 2)

        # Classification
        score_pct = results['overall_plagiarism_score']
        if score_pct < 15:
            results['classification'] = 'Original'
        elif score_pct < 30:
            results['classification'] = 'Low Similarity'
        elif score_pct < 50:
            results['classification'] = 'Moderate Similarity'
        elif score_pct < 75:
            results['classification'] = 'High Similarity'
        else:
            results['classification'] = 'Very High Similarity (Likely Plagiarism)'

        return results
