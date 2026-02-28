"""
Plagiarism Detection Tool - Flask Web Application
==================================================
Main application entry point providing web interface for:
- Single document plagiarism checking
- Document-to-document comparison
- Batch processing of multiple documents
- PDF report generation and download
"""

import os
import uuid
import json
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify, session
)
from werkzeug.utils import secure_filename

from utils.document_parser import DocumentParser
from utils.preprocessor import TextPreprocessor
from utils.similarity import SimilarityAnalyzer
from utils.web_search import WebSearchChecker
from utils.report_generator import ReportGenerator
from utils.ai_detector import AIDetector

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'plagiarism-detection-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['REPORTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'reports')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Initialize components
preprocessor = TextPreprocessor()
analyzer = SimilarityAnalyzer()
web_checker = WebSearchChecker()
report_gen = ReportGenerator(app.config['REPORTS_FOLDER'])
parser = DocumentParser()
ai_detector = AIDetector()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file):
    """Save uploaded file with unique name and return path."""
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)
    return filepath, filename


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """Compare two documents for plagiarism."""
    if request.method == 'GET':
        return render_template('compare.html')

    # Validate uploads
    file1 = request.files.get('document1')
    file2 = request.files.get('document2')

    if not file1 or not file2:
        flash('Please upload both documents.', 'error')
        return redirect(url_for('compare'))

    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        flash('Unsupported file format. Use PDF, DOCX, or TXT.', 'error')
        return redirect(url_for('compare'))

    include_semantic = request.form.get('include_semantic') == 'on'

    try:
        # Save and extract text
        path1, name1 = save_upload(file1)
        path2, name2 = save_upload(file2)

        text1 = parser.extract_text(path1)
        text2 = parser.extract_text(path2)

        if not text1.strip() or not text2.strip():
            flash('One or both documents appear to be empty.', 'error')
            return redirect(url_for('compare'))

        # Preprocess
        tokens1 = preprocessor.preprocess(text1)
        tokens2 = preprocessor.preprocess(text2)

        # Run analysis
        results = analyzer.full_analysis(text1, text2, tokens1, tokens2,
                                         include_semantic=include_semantic)

        # Generate highlighted text
        matched_sentences = results.get('semantic_similarity', {}).get('matched_pairs', [])
        highlighted_html = report_gen.highlight_matches(text1, matched_sentences)

        # AI detection on primary document
        ai_result = ai_detector.predict(text1)
        results['ai_detection'] = ai_result

        # Web search check
        if request.form.get('web_check') == 'on':
            results['web_search'] = web_checker.check_text(text1)
        else:
            results['web_search'] = {'enabled': False, 'sources': []}

        # Generate PDF report
        pdf_path = report_gen.generate_pdf_report(results, name1, name2,
                                                   ai_result=ai_result)
        pdf_filename = os.path.basename(pdf_path)

        # Cleanup uploaded files
        os.remove(path1)
        os.remove(path2)

        return render_template('results.html',
                               results=results,
                               doc1_name=name1,
                               doc2_name=name2,
                               highlighted_html=highlighted_html,
                               pdf_filename=pdf_filename,
                               ai_result=ai_result)
    except Exception as e:
        flash(f'Error processing documents: {str(e)}', 'error')
        return redirect(url_for('compare'))


@app.route('/check', methods=['GET', 'POST'])
def check_single():
    """Check a single document for plagiarism (web search)."""
    if request.method == 'GET':
        return render_template('check.html')

    file = request.files.get('document')
    compare_text = request.form.get('compare_text', '').strip()

    if not file:
        flash('Please upload a document.', 'error')
        return redirect(url_for('check_single'))

    if not allowed_file(file.filename):
        flash('Unsupported file format. Use PDF, DOCX, or TXT.', 'error')
        return redirect(url_for('check_single'))

    include_semantic = request.form.get('include_semantic') == 'on'

    try:
        path, name = save_upload(file)
        text = parser.extract_text(path)

        if not text.strip():
            flash('Document appears to be empty.', 'error')
            return redirect(url_for('check_single'))

        results = {}

        # If comparison text is provided, compare against it
        if compare_text:
            tokens1 = preprocessor.preprocess(text)
            tokens2 = preprocessor.preprocess(compare_text)
            results = analyzer.full_analysis(text, compare_text, tokens1, tokens2,
                                             include_semantic=include_semantic)
            matched_sentences = results.get('semantic_similarity', {}).get('matched_pairs', [])
            highlighted_html = report_gen.highlight_matches(text, matched_sentences)
        else:
            highlighted_html = f'<p>{text}</p>'
            results = {
                'cosine_similarity': 0, 'jaccard_similarity': 0,
                'ngram_analysis': {'score': 0, 'matching_ngrams': [], 'common_count': 0,
                                   'total_ngrams_doc1': 0, 'total_ngrams_doc2': 0},
                'semantic_similarity': {'score': 0, 'matched_pairs': []},
                'overall_plagiarism_score': 0, 'classification': 'N/A (no comparison text)'
            }

        # AI detection
        ai_result = ai_detector.predict(text)
        results['ai_detection'] = ai_result

        # Web search
        results['web_search'] = web_checker.check_text(text)

        # Generate PDF
        pdf_path = report_gen.generate_pdf_report(results, name, ai_result=ai_result)
        pdf_filename = os.path.basename(pdf_path)

        os.remove(path)

        return render_template('results.html',
                               results=results,
                               doc1_name=name,
                               doc2_name='Pasted Text' if compare_text else None,
                               highlighted_html=highlighted_html,
                               pdf_filename=pdf_filename,
                               ai_result=ai_result)
    except Exception as e:
        flash(f'Error processing document: {str(e)}', 'error')
        return redirect(url_for('check_single'))


@app.route('/batch', methods=['GET', 'POST'])
def batch_process():
    """Batch compare multiple documents against each other."""
    if request.method == 'GET':
        return render_template('batch.html')

    files = request.files.getlist('documents')

    if len(files) < 2:
        flash('Please upload at least 2 documents for batch comparison.', 'error')
        return redirect(url_for('batch_process'))

    valid_files = [f for f in files if f.filename and allowed_file(f.filename)]
    if len(valid_files) < 2:
        flash('At least 2 valid documents required (PDF, DOCX, TXT).', 'error')
        return redirect(url_for('batch_process'))

    try:
        # Save and extract all documents
        documents = []
        for f in valid_files:
            path, name = save_upload(f)
            text = parser.extract_text(path)
            tokens = preprocessor.preprocess(text)
            documents.append({
                'name': name, 'path': path,
                'text': text, 'tokens': tokens
            })

        # Compare every pair
        pair_results = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                d1, d2 = documents[i], documents[j]
                result = analyzer.full_analysis(
                    d1['text'], d2['text'],
                    d1['tokens'], d2['tokens'],
                    include_semantic=False  # Skip semantic for speed in batch
                )
                pair_results.append({
                    'doc1': d1['name'],
                    'doc2': d2['name'],
                    'overall_score': result['overall_plagiarism_score'],
                    'cosine': result['cosine_similarity'],
                    'jaccard': result['jaccard_similarity'],
                    'ngram': result['ngram_analysis']['score'],
                    'classification': result['classification']
                })

        # Sort by similarity (highest first)
        pair_results.sort(key=lambda x: x['overall_score'], reverse=True)

        # Cleanup
        for d in documents:
            if os.path.exists(d['path']):
                os.remove(d['path'])

        return render_template('batch_results.html',
                               results=pair_results,
                               total_documents=len(documents))
    except Exception as e:
        flash(f'Error in batch processing: {str(e)}', 'error')
        return redirect(url_for('batch_process'))


@app.route('/download/<filename>')
def download_report(filename):
    """Download a generated PDF report."""
    filepath = os.path.join(app.config['REPORTS_FOLDER'], secure_filename(filename))
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    flash('Report file not found.', 'error')
    return redirect(url_for('index'))


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """REST API endpoint for document comparison."""
    file1 = request.files.get('document1')
    file2 = request.files.get('document2')

    if not file1 or not file2:
        return jsonify({'error': 'Both document1 and document2 are required'}), 400

    try:
        path1, name1 = save_upload(file1)
        path2, name2 = save_upload(file2)

        text1 = parser.extract_text(path1)
        text2 = parser.extract_text(path2)

        tokens1 = preprocessor.preprocess(text1)
        tokens2 = preprocessor.preprocess(text2)

        results = analyzer.full_analysis(text1, text2, tokens1, tokens2, include_semantic=False)

        os.remove(path1)
        os.remove(path2)

        return jsonify({
            'document1': name1,
            'document2': name2,
            'overall_score': results['overall_plagiarism_score'],
            'classification': results['classification'],
            'cosine_similarity': round(results['cosine_similarity'], 4),
            'jaccard_similarity': round(results['jaccard_similarity'], 4),
            'ngram_score': round(results['ngram_analysis']['score'], 4),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
