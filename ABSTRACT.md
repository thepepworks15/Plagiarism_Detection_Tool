# Abstract

## Plagiarism Detection Tool: A Multi-Algorithm Approach with Semantic AI Analysis

### Background

Plagiarism — the act of presenting someone else's work or ideas as one's own — is a growing concern in academic institutions, publishing, and professional environments. With the vast amount of digital content available online, the temptation and ease of copying text have increased significantly. Traditional plagiarism detection methods rely on simple string matching or keyword comparison, which can be easily circumvented through paraphrasing, synonym substitution, or structural reorganization of content. There is a pressing need for intelligent plagiarism detection systems that can identify not only verbatim copying but also paraphrased and semantically similar content.

### Objective

This project presents the design and implementation of an advanced Plagiarism Detection Tool that employs multiple similarity detection algorithms — both lexical and semantic — to provide comprehensive plagiarism analysis. The system aims to detect various forms of textual similarity including direct copying, light paraphrasing, structural rearrangement, and intelligent paraphrasing that preserves meaning while changing surface-level text.

### Methodology

The tool implements four complementary plagiarism detection algorithms:

1. **Cosine Similarity with TF-IDF Vectorization**: Documents are converted into Term Frequency–Inverse Document Frequency (TF-IDF) vectors, and the cosine of the angle between these high-dimensional vectors is computed to measure overall document similarity. This method effectively captures the statistical distribution of important terms across documents.

2. **Jaccard Similarity Index**: The system computes the ratio of shared unique tokens (words) to the total unique tokens across both documents, providing a straightforward measure of vocabulary overlap that is particularly effective for detecting direct copy-paste plagiarism.

3. **N-gram Analysis**: The tool generates all possible trigrams (sequences of three consecutive words) from both documents and calculates their overlap ratio. This approach captures phrase-level similarities and is effective at detecting rearranged or partially modified text.

4. **Semantic Similarity using Sentence Transformers**: Leveraging the pre-trained `all-MiniLM-L6-v2` sentence transformer model from Hugging Face, the system encodes individual sentences into 384-dimensional dense vector embeddings that capture semantic meaning. Pairwise cosine similarity between all sentence embeddings from both documents is computed, enabling detection of paraphrased content that lexical methods cannot identify.

Additionally, the tool integrates with the Google Custom Search API to check document content against online sources, enabling detection of content copied from the web.

### Implementation

The system is implemented as a full-stack web application using:
- **Backend**: Python 3 with Flask web framework
- **NLP Pipeline**: NLTK for tokenization, stopword removal, and Porter stemming; scikit-learn for TF-IDF vectorization
- **AI/ML**: Sentence Transformers library for semantic embeddings
- **Document Processing**: PyPDF2 for PDF extraction, python-docx for Word documents
- **Reporting**: ReportLab for PDF report generation with color-coded plagiarism highlighting
- **Frontend**: Bootstrap 5 responsive web interface with drag-and-drop file upload

The application supports three modes of operation: (1) pairwise document comparison, (2) single document checking against web sources, and (3) batch processing for comparing multiple documents simultaneously.

### Results and Scoring

The system produces a weighted overall plagiarism score combining all four algorithms (Cosine: 30%, Jaccard: 20%, N-gram: 20%, Semantic: 30%) and classifies documents on a five-level scale from "Original" (0–15%) to "Very High Similarity — Likely Plagiarism" (75–100%). Detailed reports include highlighted matched sections, matched sentence pairs with individual similarity scores, matching trigrams, and identified online sources.

### Conclusion

The multi-algorithm approach provides significantly more robust plagiarism detection than any single method alone. Lexical methods (cosine, Jaccard, n-gram) effectively catch direct and lightly modified copying, while the semantic similarity analysis using sentence transformers successfully identifies intelligently paraphrased content. The combination of these approaches, presented through an intuitive web interface with detailed reporting, makes this tool a comprehensive solution for plagiarism detection in academic and professional settings.

### Keywords

Plagiarism Detection, Natural Language Processing, TF-IDF, Cosine Similarity, Jaccard Index, N-gram Analysis, Semantic Similarity, Sentence Transformers, Flask, Machine Learning
