"""
Web Search Plagiarism Checker
=============================
Uses Google Custom Search API to check text against online sources.

Setup:
1. Go to https://console.developers.google.com/
2. Create a project and enable "Custom Search API"
3. Create an API key
4. Go to https://programmablesearchengine.google.com/
5. Create a search engine (search the entire web)
6. Get the Search Engine ID (cx)
7. Set environment variables:
   - GOOGLE_API_KEY=your_api_key
   - GOOGLE_CSE_ID=your_search_engine_id

Algorithm:
    1. Extract key sentences/phrases from the document
    2. Search each phrase on Google
    3. Compare document text with search result snippets
    4. Flag matches above a similarity threshold
"""

import os
import requests
from urllib.parse import quote_plus


class WebSearchChecker:
    """Checks text against online sources using Google Custom Search API."""

    def __init__(self, api_key=None, cse_id=None):
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY', '')
        self.cse_id = cse_id or os.environ.get('GOOGLE_CSE_ID', '')
        self.base_url = 'https://www.googleapis.com/customsearch/v1'

    def is_configured(self):
        """Check if API credentials are set."""
        return bool(self.api_key and self.cse_id)

    def search(self, query, num_results=5):
        """
        Perform a Google Custom Search.

        Args:
            query: Search query string
            num_results: Number of results to return (max 10)

        Returns:
            list: Search results with title, link, snippet
        """
        if not self.is_configured():
            return []

        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'num': min(num_results, 10)
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            return results
        except requests.RequestException:
            return []

    def check_text(self, text, num_queries=5):
        """
        Check text against online sources.

        Process:
        1. Split text into sentences
        2. Select representative sentences as search queries
        3. Search each query on Google
        4. Collect and deduplicate matching sources

        Args:
            text: Document text to check
            num_queries: Number of search queries to make

        Returns:
            dict: Web plagiarism check results
        """
        from nltk.tokenize import sent_tokenize

        if not self.is_configured():
            return {
                'enabled': False,
                'message': 'Google Custom Search API not configured. Set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.',
                'sources': []
            }

        sentences = sent_tokenize(text)
        # Select longer sentences as they're more distinctive
        sentences = [s for s in sentences if len(s.split()) > 6]
        sentences.sort(key=len, reverse=True)

        # Pick representative queries spread across the document
        step = max(1, len(sentences) // num_queries)
        queries = sentences[::step][:num_queries]

        all_sources = []
        seen_links = set()

        for query in queries:
            # Truncate long queries (API limit)
            search_query = query[:150]
            results = self.search(f'"{search_query}"')

            for result in results:
                link = result['link']
                if link not in seen_links:
                    seen_links.add(link)
                    result['matched_query'] = query
                    all_sources.append(result)

        return {
            'enabled': True,
            'queries_made': len(queries),
            'sources_found': len(all_sources),
            'sources': all_sources[:20]
        }
