import requests
from duckduckgo_search import DDGS
import os
import sys
import io
from typing import List, Dict
import re

# Ensure UTF-8 output for Unicode characters
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Clean text to avoid encoding issues
def clean_text(text):
    return text.encode('ascii', errors='ignore').decode()

def duckduckgo_search(query: str, max_results: int = 5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for query: {query}"
        # Format results
        formatted_results = []
        for result in results:
            link = result.get('href') or result.get('link', 'N/A')
            title = clean_text(result.get('title', 'N/A'))
            body = clean_text(result.get('body', 'N/A'))
            # Extract and highlight key information
            if len(body) > 300:
                body = body[:300] + "..."
                
            formatted_results.append(
                f"Title: {title}\n"
                f"Description: {body}\n"
                f"URL: {link}")
        
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"❌ Error performing DuckDuckGo search: {str(e)}"

# print(duckduckgo_search("give biotech research papers", max_results=5))