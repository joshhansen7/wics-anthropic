"""
Fuzzy Cache Matching Module

Uses AI to determine if a search query matches an existing
cached article, enabling smart redirects instead of duplicate synthesis.
"""

import os
import json
import difflib
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

SIMILARITY_THRESHOLD = 0.95

# Function to get AI to evaluate cache matches
CACHE_MATCH_PROMPT = """
I need your help determining if a user's search query is similar enough to existing cached articles that we should redirect to one of them, rather than creating a new article.

User search query: "{query}"
Language: {language}

Existing cached articles in this language:
{cached_articles}

Please determine if any of these cached articles are similar enough to the user's query that we should redirect to it instead of creating a new article. Consider:
1. If the query is a slight misspelling of an existing article
2. If the query is a synonym or alternative form of an existing article
3. If the query is a more or less specific version of an existing article (e.g. "Albert Einstein" vs "Einstein")
4. If the query is a related concept that would be fully covered by an existing article

Format your response as a JSON object like this:
{{
  "redirect": true or false,
  "filename": "exact name of the file to redirect to, if any",
  "confidence": 0-1 score of how confident you are in this match,
  "rationale": "brief explanation of your decision"
}}

Be relatively conservative in your matching - only suggest a redirect if you're confident the existing article would satisfy the user's query. We prefer to create new articles when in doubt.
"""

def get_cached_articles(cache_dir: str, language: str) -> List[Dict]:
    """
    Get list of cached articles for a specific language.

    Args:
        cache_dir: Path to cache directory
        language: Language code to filter by

    Returns:
        List of cached article metadata
    """
    lang_cache_dir = os.path.join(cache_dir, language)

    if not os.path.exists(lang_cache_dir):
        return []

    return [
        {"filename": filename, "path": os.path.join(lang_cache_dir, filename)}
        for filename in os.listdir(lang_cache_dir)
    ]

def basic_similarity_check(query: str, cached_articles: List[Dict]) -> Optional[Dict]:
    """
    Quick similarity check using difflib before calling AI.

    Args:
        query: User's search query
        cached_articles: List of cached articles

    Returns:
        Matching article if high-confidence match found, else None
    """
    query_normalized = query.lower().strip()

    for article in cached_articles:
        article_name_normalized = article["filename"].lower().strip()

        # Check exact match
        if query_normalized == article_name_normalized:
            return article

        # Check high similarity
        similarity = difflib.SequenceMatcher(None, query_normalized, article_name_normalized).ratio()
        if similarity > SIMILARITY_THRESHOLD:
            return article

    return None

def ai_cache_match(client: OpenAI, query: str, language: str, cached_articles: List[Dict]) -> Tuple[bool, Optional[Dict], float, str]:
    """
    Use AI to determine if query should redirect to existing cached article.

    Args:
        client: OpenAI client
        query: User's search query
        language: Language code
        cached_articles: List of cached articles

    Returns:
        Tuple of (should_redirect, matching_article_dict, confidence, rationale)
    """
    if not cached_articles:
        return False, None, 0.0, "No cached articles available"

    cached_articles_text = "\n".join(f"{i+1}. {article['filename']}" for i, article in enumerate(cached_articles))

    prompt = CACHE_MATCH_PROMPT.format(
        query=query,
        language=language,
        cached_articles=cached_articles_text
    )

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=1000,
            temperature=0.2
        )

        response_text = response.output_text or ""

        import re
        json_match = re.search(r'({[\s\S]*})', response_text)

        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                should_redirect = json_data.get("redirect", False)
                article_name = json_data.get("filename", "")
                confidence = json_data.get("confidence", 0.0)
                rationale = json_data.get("rationale", "No rationale provided")

                matching_article = None
                if should_redirect and article_name:
                    for article in cached_articles:
                        if article["filename"].lower() == article_name.lower():
                            matching_article = article
                            break

                return should_redirect, matching_article, confidence, rationale

            except json.JSONDecodeError:
                print(f"Failed to parse JSON response: {response_text}")
        else:
            print(f"Failed to extract JSON from response: {response_text}")

    except Exception as e:
        print(f"Error in AI API call: {e}")

    return False, None, 0.0, "Error processing the request"

def find_fuzzy_cache_match(client: OpenAI, query: str, language: str, cache_dir: str) -> Tuple[bool, Optional[str], float, str]:
    """
    Determine if query should redirect to existing cached article.

    Args:
        client: OpenAI client
        query: User's search query
        language: Language code
        cache_dir: Path to cache directory

    Returns:
        Tuple of (should_redirect, redirect_path, confidence, rationale)
    """
    cached_articles = get_cached_articles(cache_dir, language)

    if not cached_articles:
        return False, None, 0.0, "No cached articles available"

    # Try fast similarity check first
    basic_match = basic_similarity_check(query, cached_articles)
    if basic_match:
        return True, basic_match["path"], 1.0, f"Exact or near-exact match: {basic_match['filename']}"

    # Use AI for sophisticated matching
    should_redirect, matching_article, confidence, rationale = ai_cache_match(
        client, query, language, cached_articles
    )

    if should_redirect and matching_article:
        return True, matching_article["path"], confidence, rationale

    return False, None, confidence, rationale
