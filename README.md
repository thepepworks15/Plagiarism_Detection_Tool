# Plagiarism Detection Tool

An advanced plagiarism detection system built with Python and Flask that uses multiple similarity algorithms — including AI-powered semantic analysis — to detect copied content across documents.

## Features

- **Multi-Format Document Upload**: Supports PDF, DOCX, and TXT files
- **Text Preprocessing**: Tokenization, stopword removal, and Porter stemming
- **Four Detection Algorithms**:
  - Cosine Similarity (TF-IDF vectors)
  - Jaccard Similarity (set overlap)
  - N-gram Analysis (trigram matching)
  - Semantic Similarity (sentence-transformers AI)
- **Web Search Check**: Google Custom Search API integration
- **Highlighted Reports**: Color-coded matched sections
- **Percentage Score**: Weighted overall plagiarism percentage with classification
- **Batch Processing**: Compare multiple documents simultaneously
- **PDF Report Export**: Professional downloadable reports
- **REST API**: Programmatic access endpoint

## Project Structure

```
Plagiarism Detection Tool/
├── app.py                      # Flask application (main entry point)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── generate_ppt.py             # PowerPoint generator (18 slides)
├── generate_report.py          # Project report generator (50 pages)
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py         # Text preprocessing (tokenize, stem, clean)
│   ├── document_parser.py      # PDF/DOCX/TXT text extraction
│   ├── similarity.py           # 4 similarity algorithms
│   ├── web_search.py           # Google Custom Search integration
│   └── report_generator.py     # HTML highlighting & PDF report
├── templates/
│   ├── base.html               # Base template with navbar/footer
│   ├── index.html              # Landing page
│   ├── compare.html            # Two-document comparison form
│   ├── check.html              # Single document check form
│   ├── results.html            # Detailed results page
│   ├── batch.html              # Batch upload form
│   └── batch_results.html      # Batch comparison results
├── uploads/                    # Temporary upload storage
├── reports/                    # Generated PDF reports
└── static/
    ├── css/
    └── js/
```

## Setup Guide

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd "Plagiarism Detection Tool"

# Create virtual environment
python -m venv myenv

# Activate it
# Windows:
myenv\Scripts\activate
# macOS/Linux:
source myenv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **Flask** — Web framework
- **PyPDF2** — PDF text extraction
- **python-docx** — DOCX text extraction
- **nltk** — Tokenization, stopwords, stemming
- **scikit-learn** — TF-IDF and cosine similarity
- **sentence-transformers** — Semantic similarity (downloads ~90MB model on first run)
- **requests / beautifulsoup4** — Web search
- **reportlab** — PDF report generation
- **python-pptx** — PowerPoint generation

### Step 3: Download NLTK Data (Automatic)

NLTK data downloads automatically on first run. To manually download:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 4: Configure Web Search (Optional)

For Google Custom Search API integration:

1. Go to [Google Cloud Console](https://console.developers.google.com/)
2. Create a project and enable **Custom Search API**
3. Create an API key
4. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
5. Create a search engine (search the entire web)
6. Set environment variables:

```bash
# Windows
set GOOGLE_API_KEY=your_api_key_here
set GOOGLE_CSE_ID=your_search_engine_id_here

# macOS/Linux
export GOOGLE_API_KEY=your_api_key_here
export GOOGLE_CSE_ID=your_search_engine_id_here
```

### Step 5: Run the Application

```bash
python app.py
```

Open your browser to: **http://localhost:5000**

## Usage

### Compare Two Documents
1. Navigate to **Compare** page
2. Upload two documents (PDF, DOCX, or TXT)
3. Toggle semantic analysis and web check options
4. Click **Analyze for Plagiarism**
5. View highlighted results and download PDF report

### Check Single Document
1. Navigate to **Check** page
2. Upload a document
3. Optionally paste comparison text
4. Results include web source check

### Batch Processing
1. Navigate to **Batch** page
2. Upload 2+ documents
3. View pairwise comparison matrix

### REST API
```bash
curl -X POST http://localhost:5000/api/compare \
  -F "document1=@file1.pdf" \
  -F "document2=@file2.pdf"
```

Response:
```json
{
  "document1": "file1.pdf",
  "document2": "file2.pdf",
  "overall_score": 45.2,
  "classification": "Moderate Similarity",
  "cosine_similarity": 0.5123,
  "jaccard_similarity": 0.3456,
  "ngram_score": 0.2890
}
```

## Algorithm Details

### 1. Cosine Similarity (TF-IDF)
Converts documents into TF-IDF (Term Frequency–Inverse Document Frequency) vectors and computes the cosine of the angle between them. Score ranges from 0 (no similarity) to 1 (identical). Weight: 30%.

### 2. Jaccard Similarity
Measures the ratio of shared tokens to total unique tokens: `|A ∩ B| / |A ∪ B|`. Simple but effective for detecting direct copy-paste. Weight: 20%.

### 3. N-gram Analysis
Generates all trigrams (3-word sequences) from both documents and calculates their overlap ratio. Detects rearranged text and structural copying. Weight: 20%.

### 4. Semantic Similarity
Uses the `all-MiniLM-L6-v2` sentence transformer model to encode sentences into 384-dimensional embeddings. Computes pairwise cosine similarity between all sentence pairs. Catches paraphrased plagiarism that lexical methods miss. Weight: 30%.

### Scoring
The overall plagiarism score is a weighted combination:
- **0–15%**: Original
- **15–30%**: Low Similarity
- **30–50%**: Moderate Similarity
- **50–75%**: High Similarity
- **75–100%**: Very High Similarity (Likely Plagiarism)

## Generating Documentation

### PowerPoint Presentation (18 slides)
```bash
python generate_ppt.py
```
Generates `Plagiarism_Detection_Tool_Presentation.pptx`

### Project Report (50 pages)
```bash
python generate_report.py
```
Generates `Plagiarism_Detection_Tool_Project_Report.pdf`

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Backend | Python 3, Flask |
| NLP | NLTK, scikit-learn |
| AI/ML | Sentence Transformers (Hugging Face) |
| Document Parsing | PyPDF2, python-docx |
| PDF Reports | ReportLab |
| Frontend | Bootstrap 5, Font Awesome |
| Web Search | Google Custom Search API |

## License

This project is for educational purposes.
