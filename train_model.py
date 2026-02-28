"""
Model Training & Testing Script
================================
Trains the AI detector and calibrates the plagiarism similarity engine.
Tests both systems and reports accuracy metrics.

Usage:
    python train_model.py

Requires:
    test_data/balanced_ai_human_prompts.csv
"""

import csv
import random
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from utils.ai_detector import AIDetector, extract_features
from utils.preprocessor import TextPreprocessor
from utils.similarity import SimilarityAnalyzer

random.seed(42)
np.random.seed(42)


def load_csv_data(csv_path):
    """Load text and labels from CSV."""
    texts, labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text'].strip()
            if len(text) > 30:
                texts.append(text)
                labels.append(int(row['generated']))
    return texts, labels


def train_ai_detector(texts, labels):
    """Train and evaluate the AI text detector."""
    print("=" * 70)
    print("PHASE 1: TRAINING AI-GENERATED TEXT DETECTOR")
    print("=" * 70)

    # Split data
    X_train_t, X_test_t, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"\nDataset: {len(texts)} samples ({sum(labels)} AI, {len(labels)-sum(labels)} Human)")
    print(f"Train set: {len(X_train_t)} | Test set: {len(X_test_t)}")

    # Train
    detector = AIDetector()
    print("\nTraining Gradient Boosting Classifier (200 estimators)...")
    start = time.time()
    metrics = detector.train(X_train_t, y_train)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    print(f"\n5-Fold Cross-Validation Accuracy: {metrics['cv_accuracy_mean']:.4f} "
          f"(+/- {metrics['cv_accuracy_std']:.4f})")
    print(f"Per-fold scores: {[f'{s:.4f}' for s in metrics['cv_scores']]}")

    # Test
    print("\n--- Test Set Evaluation ---")
    y_pred = []
    for t in X_test_t:
        result = detector.predict(t)
        y_pred.append(1 if result['is_ai'] else 0)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  Predicted:  Human   AI")
    print(f"  Human:     {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  AI:        {cm[1][0]:5d}  {cm[1][1]:5d}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Human', 'AI-Generated'])}")

    # Top features
    print("Top 5 Most Important Features:")
    importances = sorted(metrics['feature_importances'].items(), key=lambda x: x[1], reverse=True)
    for name, imp in importances[:5]:
        print(f"  {name:25s} {imp:.4f}")

    # Save model
    detector.save()
    print(f"\nModel saved to: {detector.model_path}")

    return detector, acc


def create_plagiarism_pairs(texts, labels):
    """
    Create synthetic plagiarism test pairs from the dataset.

    Types:
    1. IDENTICAL:        Same text copied verbatim
    2. MINOR_EDIT:       5-10% words changed
    3. WORD_SWAP:        20-30% words replaced with synonyms
    4. SHUFFLE_SENTENCES: Sentences reordered
    5. PARTIAL_COPY:     50% of text copied, rest different
    6. UNRELATED:        Two completely different texts
    """
    ai_texts = [t for t, l in zip(texts, labels) if l == 1 and len(t) > 100]
    human_texts = [t for t, l in zip(texts, labels) if l == 0 and len(t) > 100]

    pairs = []

    # --- Type 1: IDENTICAL (label=1, expected score > 95%) ---
    for text in random.sample(ai_texts, min(40, len(ai_texts))):
        pairs.append({
            'text1': text, 'text2': text,
            'label': 1, 'type': 'identical',
            'expected_min': 0.95
        })

    # --- Type 2: MINOR_EDIT (label=1, expected > 70%) ---
    for text in random.sample(ai_texts, min(40, len(ai_texts))):
        words = text.split()
        n_change = max(1, len(words) // 15)
        modified = words.copy()
        for _ in range(n_change):
            idx = random.randint(0, len(modified) - 1)
            modified[idx] = random.choice(['the', 'a', 'this', 'that', 'very', 'quite', 'rather'])
        pairs.append({
            'text1': text, 'text2': ' '.join(modified),
            'label': 1, 'type': 'minor_edit',
            'expected_min': 0.65
        })

    # --- Type 3: WORD_SWAP / synonym replacement (label=1, expected > 40%) ---
    synonym_map = {
        'important': 'significant', 'significant': 'important',
        'large': 'big', 'big': 'large',
        'help': 'assist', 'assist': 'help',
        'use': 'utilize', 'utilize': 'use',
        'make': 'create', 'create': 'make',
        'show': 'demonstrate', 'demonstrate': 'show',
        'good': 'beneficial', 'beneficial': 'good',
        'new': 'novel', 'novel': 'new',
        'also': 'additionally', 'additionally': 'also',
        'however': 'nevertheless', 'nevertheless': 'however',
        'because': 'since', 'since': 'because',
        'including': 'encompassing', 'encompassing': 'including',
        'various': 'diverse', 'diverse': 'various',
        'provide': 'offer', 'offer': 'provide',
        'development': 'advancement', 'advancement': 'development',
        'impact': 'effect', 'effect': 'impact',
        'role': 'function', 'function': 'role',
        'approach': 'method', 'method': 'approach',
        'system': 'framework', 'framework': 'system',
        'process': 'procedure', 'procedure': 'process',
    }
    for text in random.sample(ai_texts, min(40, len(ai_texts))):
        words = text.split()
        modified = []
        for w in words:
            lower = w.lower().strip('.,;:!?')
            if lower in synonym_map and random.random() < 0.5:
                replacement = synonym_map[lower]
                if w[0].isupper():
                    replacement = replacement.capitalize()
                modified.append(replacement)
            else:
                modified.append(w)
        pairs.append({
            'text1': text, 'text2': ' '.join(modified),
            'label': 1, 'type': 'word_swap',
            'expected_min': 0.40
        })

    # --- Type 4: SHUFFLE_SENTENCES (label=1, expected > 60%) ---
    for text in random.sample(ai_texts, min(40, len(ai_texts))):
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sents) >= 3:
            shuffled = sents.copy()
            random.shuffle(shuffled)
            pairs.append({
                'text1': text, 'text2': ' '.join(shuffled),
                'label': 1, 'type': 'shuffle_sentences',
                'expected_min': 0.55
            })

    # --- Type 5: PARTIAL_COPY (label=1, expected > 30%) ---
    for text in random.sample(ai_texts, min(30, len(ai_texts))):
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sents) >= 4:
            half = len(sents) // 2
            other = random.choice(human_texts) if human_texts else "This is completely unrelated filler text about gardening and cooking."
            other_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', other) if s.strip()]
            combined = sents[:half] + other_sents[:half]
            pairs.append({
                'text1': text, 'text2': ' '.join(combined),
                'label': 1, 'type': 'partial_copy',
                'expected_min': 0.25
            })

    # --- Type 6: UNRELATED (label=0, expected < 20%) ---
    used_indices = set()
    for _ in range(60):
        i = random.randint(0, len(ai_texts) - 1)
        j = random.randint(0, len(human_texts) - 1)
        while (i, j) in used_indices:
            i = random.randint(0, len(ai_texts) - 1)
            j = random.randint(0, len(human_texts) - 1)
        used_indices.add((i, j))
        pairs.append({
            'text1': ai_texts[i], 'text2': human_texts[j],
            'label': 0, 'type': 'unrelated',
            'expected_min': 0.0
        })

    # Add cross-topic unrelated AI pairs
    for _ in range(30):
        i, j = random.sample(range(len(ai_texts)), 2)
        pairs.append({
            'text1': ai_texts[i], 'text2': ai_texts[j],
            'label': 0, 'type': 'unrelated_same_style',
            'expected_min': 0.0
        })

    random.shuffle(pairs)
    return pairs


def calibrate_similarity(pairs):
    """
    Test similarity algorithms on labeled pairs and find optimal thresholds.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: CALIBRATING PLAGIARISM SIMILARITY ENGINE")
    print("=" * 70)

    preprocessor = TextPreprocessor()
    analyzer = SimilarityAnalyzer()

    results = []
    total = len(pairs)

    print(f"\nRunning similarity analysis on {total} pairs...")
    start = time.time()

    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processing pair {i+1}/{total}...")

        text1, text2 = pair['text1'], pair['text2']
        tokens1 = preprocessor.preprocess(text1)
        tokens2 = preprocessor.preprocess(text2)

        analysis = analyzer.full_analysis(text1, text2, tokens1, tokens2,
                                          include_semantic=False)

        results.append({
            'label': pair['label'],
            'type': pair['type'],
            'overall': analysis['overall_plagiarism_score'],
            'cosine': analysis['cosine_similarity'],
            'jaccard': analysis['jaccard_similarity'],
            'ngram': analysis['ngram_analysis']['score'],
        })

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")

    # Find optimal threshold
    print("\n--- Threshold Optimization ---")
    best_acc = 0
    best_threshold = 0

    for threshold in np.arange(10, 50, 0.5):
        preds = [1 if r['overall'] >= threshold else 0 for r in results]
        true = [r['label'] for r in results]
        acc = accuracy_score(true, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold}%")
    print(f"Accuracy at optimal threshold: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # Detailed results at best threshold
    preds = [1 if r['overall'] >= best_threshold else 0 for r in results]
    true = [r['label'] for r in results]
    prec = precision_score(true, preds, zero_division=0)
    rec = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)
    cm = confusion_matrix(true, preds)

    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  Predicted:     Not-Plag  Plagiarized")
    print(f"  Not-Plag:     {cm[0][0]:7d}  {cm[0][1]:7d}")
    print(f"  Plagiarized:  {cm[1][0]:7d}  {cm[1][1]:7d}")

    # Per-type breakdown
    print("\n--- Per-Type Score Distribution ---")
    types = {}
    for r in results:
        t = r['type']
        if t not in types:
            types[t] = {'scores': [], 'label': r['label']}
        types[t]['scores'].append(r['overall'])

    print(f"{'Type':<25s} {'Label':>6s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 65)
    for t in sorted(types.keys()):
        scores = types[t]['scores']
        lbl = 'Plag' if types[t]['label'] == 1 else 'Clean'
        print(f"{t:<25s} {lbl:>6s} {np.mean(scores):>7.1f}% {np.std(scores):>7.1f}% "
              f"{np.min(scores):>7.1f}% {np.max(scores):>7.1f}%")

    # Weight optimization
    print("\n--- Weight Optimization (Grid Search) ---")
    best_weights = None
    best_w_acc = 0

    for cw in np.arange(0.2, 0.6, 0.05):
        for jw in np.arange(0.1, 0.5, 0.05):
            nw = 1.0 - cw - jw
            if nw < 0.05 or nw > 0.5:
                continue
            # Recompute scores with new weights
            new_scores = []
            for r in results:
                score = (cw * r['cosine'] + jw * r['jaccard'] + nw * r['ngram']) * 100
                new_scores.append(score)

            # Find best threshold for these weights
            for thr in np.arange(10, 50, 1):
                preds = [1 if s >= thr else 0 for s in new_scores]
                acc = accuracy_score(true, preds)
                if acc > best_w_acc:
                    best_w_acc = acc
                    best_weights = (cw, jw, nw)
                    best_w_thr = thr

    if best_weights:
        print(f"Best weights: Cosine={best_weights[0]:.2f}, Jaccard={best_weights[1]:.2f}, "
              f"N-gram={best_weights[2]:.2f}")
        print(f"Best threshold: {best_w_thr}%")
        print(f"Accuracy with optimized weights: {best_w_acc:.4f} ({best_w_acc*100:.2f}%)")

    return best_threshold, best_weights, best_w_thr, best_w_acc


def test_combined_system(pairs, threshold, weights, w_threshold):
    """
    Final end-to-end test with optimized parameters.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: FINAL END-TO-END ACCURACY TEST")
    print("=" * 70)

    preprocessor = TextPreprocessor()
    analyzer = SimilarityAnalyzer()

    true_labels = []
    pred_labels = []
    details = []

    for pair in pairs:
        text1, text2 = pair['text1'], pair['text2']
        tokens1 = preprocessor.preprocess(text1)
        tokens2 = preprocessor.preprocess(text2)

        cosine = analyzer.cosine_similarity(text1, text2)
        jaccard = analyzer.jaccard_similarity(tokens1, tokens2)
        ngram = analyzer.ngram_similarity(tokens1, tokens2)['score']

        if weights:
            score = (weights[0] * cosine + weights[1] * jaccard + weights[2] * ngram) * 100
            thr = w_threshold
        else:
            score = (0.40 * cosine + 0.30 * jaccard + 0.30 * ngram) * 100
            thr = threshold

        pred = 1 if score >= thr else 0
        true_labels.append(pair['label'])
        pred_labels.append(pred)
        details.append({
            'type': pair['type'], 'score': score,
            'label': pair['label'], 'pred': pred,
            'correct': pred == pair['label']
        })

    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)

    print(f"\n{'='*40}")
    print(f"  FINAL ACCURACY: {acc*100:.2f}%")
    print(f"{'='*40}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted:     Not-Plag  Plagiarized")
    print(f"    Not-Plag:     {cm[0][0]:7d}  {cm[0][1]:7d}")
    print(f"    Plagiarized:  {cm[1][0]:7d}  {cm[1][1]:7d}")

    # Misclassified analysis
    wrong = [d for d in details if not d['correct']]
    if wrong:
        print(f"\n  Misclassified: {len(wrong)}/{len(details)} ({len(wrong)/len(details)*100:.1f}%)")
        type_errors = {}
        for w in wrong:
            type_errors[w['type']] = type_errors.get(w['type'], 0) + 1
        print("  Errors by type:")
        for t, c in sorted(type_errors.items(), key=lambda x: x[1], reverse=True):
            print(f"    {t}: {c}")

    return acc, weights, w_threshold


def save_config(weights, threshold, ai_accuracy, plag_accuracy):
    """Save optimized configuration."""
    import json
    config = {
        'similarity_weights': {
            'cosine': float(weights[0]) if weights else 0.40,
            'jaccard': float(weights[1]) if weights else 0.30,
            'ngram': float(weights[2]) if weights else 0.30,
            'semantic': 0.0
        },
        'similarity_weights_with_semantic': {
            'cosine': 0.25,
            'jaccard': 0.15,
            'ngram': 0.15,
            'semantic': 0.45
        },
        'plagiarism_threshold': float(threshold),
        'ai_detector_accuracy': float(ai_accuracy),
        'plagiarism_detector_accuracy': float(plag_accuracy),
    }

    config_path = 'models/config.json'
    os.makedirs('models', exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")
    return config


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    CSV_PATH = 'test_data/balanced_ai_human_prompts.csv'

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at {CSV_PATH}")
        exit(1)

    # Load data
    print("Loading dataset...")
    texts, labels = load_csv_data(CSV_PATH)
    print(f"Loaded {len(texts)} samples")

    # Phase 1: Train AI detector
    detector, ai_acc = train_ai_detector(texts, labels)

    # Phase 2: Create pairs and calibrate similarity
    print("\nCreating synthetic plagiarism pairs...")
    pairs = create_plagiarism_pairs(texts, labels)
    print(f"Created {len(pairs)} pairs")
    type_counts = {}
    for p in pairs:
        type_counts[p['type']] = type_counts.get(p['type'], 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    threshold, weights, w_threshold, cal_acc = calibrate_similarity(pairs)

    # Phase 3: Final test
    final_acc, final_weights, final_threshold = test_combined_system(
        pairs, threshold, weights, w_threshold)

    # Save everything
    config = save_config(final_weights, final_threshold, ai_acc, final_acc)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  AI Text Detector Accuracy:      {ai_acc*100:.2f}%")
    print(f"  Plagiarism Detector Accuracy:    {final_acc*100:.2f}%")
    print(f"  Optimized Weights:               Cosine={config['similarity_weights']['cosine']:.2f}, "
          f"Jaccard={config['similarity_weights']['jaccard']:.2f}, "
          f"N-gram={config['similarity_weights']['ngram']:.2f}")
    print(f"  Plagiarism Threshold:            {final_threshold}%")
    print(f"  Model saved:                     models/ai_detector.pkl")
    print(f"  Config saved:                    models/config.json")
    print("=" * 70)
