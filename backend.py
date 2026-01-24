#!/usr/bin/env python3
"""
Wikipedia Synthesizer with Claude

Combines Wikipedia articles from multiple language editions using Claude AI.
- Finds article translations across different languages
- Uses Claude to select the most relevant language editions
- Translates all editions to a target language
- Synthesizes them into a comprehensive multilingual article
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
from multiprocessing.dummy import Pool as ThreadPool
from anthropic import Anthropic


def create_language_selection_prompt(title: str, max_translations: int, source_lang: str, lang_options_text: str) -> str:
    """Generate prompt for Claude to select relevant languages."""
    return f"""Select the {max_translations} most relevant Wikipedia language editions for "{title}".

Choose languages that would provide:
- Unique perspectives and complementary information
- Culturally significant details
- Comprehensive coverage from diverse viewpoints

Available language options (language code: article title):
{lang_options_text}

Select exactly {max_translations} languages (NOT including source language {source_lang}).
Include the source language in your final list.

Respond with JSON:
{{
  "selected_languages": ["xx", "yy", "zz"],
  "rationale": "brief explanation"
}}"""

def create_translation_prompt(source_lang: str, target_lang: str, text: str) -> str:
    """Generate prompt for translating Wikipedia content."""
    return f"""Translate from {source_lang} to {target_lang}.
Maintain the original structure and formatting.

TEXT TO TRANSLATE:
{text}

TRANSLATION:"""


def create_synthesis_prompt(original_title: str, articles: Dict[str, str], target_lang: str) -> str:
    """Generate prompt for synthesizing multiple Wikipedia versions."""
    context = f"""I have {len(articles)} versions of the Wikipedia article '{original_title}' from different language editions, all translated to {target_lang}.

Your task: synthesize these into a single comprehensive article. Do not worry about length constraints - include all important information."""

    article_sections = []
    for lang, content in articles.items():
        trimmed_content = content[:128000] if len(content) > 128000 else content
        article_sections.append(f"VERSION FROM {lang} WIKIPEDIA:\n{trimmed_content}\n\n---\n\n")

    combined = context + "\n\n" + "".join(article_sections)

    return f"""{combined}

Combine these Wikipedia versions into a single comprehensive article in {target_lang}.

Requirements:
1. Follow Wikipedia's neutral point of view
2. Maintain encyclopedic tone
3. Include all important facts from all language versions
4. Create well-structured sections and subsections
5. Resolve contradictions by noting different perspectives
6. Reference source languages when relevant
7. Include hyperlinks using format: [text](/article/{target_lang}/Article_Name)
   - Generate links liberally for an interconnected knowledge base
   - Use correct capitalization
   - Replace spaces with underscores
   - Handle parentheses carefully: [text](/article/en/Name_(disambiguation))

SYNTHESIZED ARTICLE:"""

# Claude API information
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = "claude-4-5-haiku-latest"

# Define cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Define the search_wikipedia tool
SEARCH_WIKIPEDIA_TOOL = {
    "name": "search_wikipedia",
    "description": "Search for a Wikipedia article by title and language",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the Wikipedia article"
            },
            "language": {
                "type": "string",
                "description": "The language code (e.g., 'en', 'es', 'fr')"
            },
            "page": {
                "type": "integer",
                "description": "Page number for paginated results (starting from 0)",
                "default": 0
            }
        },
        "required": ["title", "language"]
    }
}

# Define tools list for Claude
CLAUDE_TOOLS = [SEARCH_WIKIPEDIA_TOOL]

# Import the fuzzy search functionality
from wikipedia_fuzzy_search import (
    get_wikipedia_article_with_fuzzy_search,
    search_wikipedia
)

def get_wikipedia_article_with_tool(
    client: Anthropic,
    title: str,
    language: str,
    first_article: bool = False
) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    Retrieves a Wikipedia article using fuzzy search when needed.

    Args:
        client: Anthropic client
        title: Article title to search for
        language: Language code (e.g., 'en', 'es', 'fr')
        first_article: Enable fallback fuzzy search in other languages

    Returns:
        Tuple of (article_text, language_links) or (None, None) if not found
    """
    return get_wikipedia_article_with_fuzzy_search(client, title, language, first_article)

def select_relevant_languages(
    client: Anthropic,
    title: str,
    source_lang: str,
    all_lang_links: List[Dict],
    max_translations: int = 5
) -> List[str]:
    """
    Uses Claude to select the most relevant languages for a given topic.

    Args:
        client: Anthropic client
        title: Article title
        source_lang: Source language code
        all_lang_links: Available language editions
        max_translations: Maximum number of languages to select

    Returns:
        List of selected language codes
    """
    lang_options = [f"{link['language']}: {link['title']}" for link in all_lang_links]
    lang_options_text = "\n".join(lang_options)

    prompt = create_language_selection_prompt(title, max_translations, source_lang, lang_options_text)
    
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Extract JSON response
        response_text = response.content[0].text
        
        # Find JSON object in response
        import re
        json_match = re.search(r'({[\s\S]*})', response_text)
        
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                selected_languages = json_data.get("selected_languages", [])
                
                # Make sure we got exactly the right number
                if len(selected_languages) > max_translations:
                    selected_languages = selected_languages[:max_translations]+source_lang
                
                print(f"Selected languages: {selected_languages}")
                print(f"Rationale: {json_data.get('rationale', 'No rationale provided')}")
                
                return selected_languages
                
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response: {response_text}")
        else:
            print(f"Failed to extract JSON from response: {response_text}")
            
    except Exception as e:
        print(f"Error in Claude API call: {e}")
    
    # Fallback to selecting languages based on alphabetical order if Claude fails
    print("Falling back to alphabetical order selection")
    langs = [link["language"] for link in all_lang_links if link["language"] != source_lang]
    return langs[:max_translations]

def get_translation_content_with_tool(translations: List[Tuple[str, str]]) -> Dict[str, Optional[str]]:
    """
    Retrieves the content of each translated article.

    Args:
        translations: List of tuples containing (language_code, article_title)

    Returns:
        Dictionary mapping language codes to article content
    """
    translation_content = {}
    
    for lang, title in translations:
        print(f"  Retrieving {lang} article: {title}")
        
        full_text = ""
        current_page = 0
        
        while True:
            try:
                result = search_wikipedia(title, lang, current_page)
                
                if not result["found"]:
                    print(f"  Article not found: {result.get('error', 'Unknown error')}")
                    translation_content[lang] = None
                    break
                
                full_text += result["content"]
                
                # Check if we've reached the last page
                if current_page >= result["total_pages"] - 1:
                    translation_content[lang] = full_text
                    break
                    
                current_page += 1
                print(f"    Retrieved page {current_page} of {result['total_pages']}")
                
            except Exception as e:
                print(f"  Error retrieving Wikipedia article: {e}")
                if full_text:
                    # Return what we have so far
                    translation_content[lang] = full_text
                else:
                    translation_content[lang] = None
                break
            
    return translation_content

def translate_with_claude(client: Anthropic, text: str, source_lang: str, target_lang: str) -> str:
    """
    Translates text using Claude API with streaming.

    Args:
        client: Anthropic client
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Translated text
    """
    if source_lang == target_lang:
        return text

    # Limit text length for API constraints
    text = text[:128000] if len(text) > 128000 else text

    prompt = create_translation_prompt(source_lang, target_lang, text)
    
    try:
        # Use the Anthropic SDK with streaming
        print(f"  Starting translation stream from {source_lang} to {target_lang}...")
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=64000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        ) as stream:
            # Collect the streamed response
            translation = ""
            for text in stream.text_stream:
                translation += text
            
            if translation:
                return translation
            else:
                print(f"Warning: Empty translation result")
                return f"Translation failed: Empty result"
            
    except Exception as e:
        print(f"Error in Claude API call: {e}")
        return f"Translation failed: {str(e)}"

def synthesize_with_claude(client: Anthropic, articles: Dict[str, str], target_lang: str, original_title: str) -> str:
    """
    Synthesizes multiple translated articles into one comprehensive article.

    Args:
        client: Anthropic client
        articles: Language code to article content mapping
        target_lang: Target language code
        original_title: Original article title

    Returns:
        Synthesized article
    """
    prompt = create_synthesis_prompt(original_title, articles, target_lang)
    
    try:
        # Use the Anthropic SDK with streaming
        print(f"  Starting synthesis stream...")
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=64000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        ) as stream:
            # Collect the streamed response
            synthesis = ""
            for text in stream.text_stream:
                synthesis += text
            
            if synthesis:
                return synthesis
            else:
                print(f"Warning: Empty synthesis result")
                return f"Synthesis failed: Empty result"
            
    except Exception as e:
        print(f"Error in Claude API call: {e}")
        return f"Synthesis failed: {str(e)}"

def translate_article_worker(args):
    """
    Worker function for parallel translation.

    Args:
        args: Tuple containing (client, content, target_lang, lang)

    Returns:
        Tuple of (lang, translated_content)
    """
    client, content, target_lang, lang = args

    if lang == target_lang or content is None:
        return lang, content

    print(f"  Translating from {lang} to {target_lang}...")
    translated = translate_with_claude(client, content, lang, target_lang)

    if translated.startswith("Translation failed:"):
        print(f"  Warning: {translated}")
        return lang, None

    return lang, translated

def get_cache_key(title: str, language: str, _max_translations: int = 5) -> str:
    """
    Generate a unique cache key for an article request.

    Args:
        title: Article title
        language: Target language code
        _max_translations: Unused (kept for API compatibility)

    Returns:
        Cache key string
    """
    return f"{language}/{title.replace(' ', '_')}"

def get_cache_path(cache_key: str) -> str:
    """
    Get the filesystem path for a cached article.
    
    Args:
        cache_key: Cache key string
        
    Returns:
        Path to the cached HTML file
    """
    return os.path.join(CACHE_DIR, f"{cache_key}.html")

def check_cache(title: str, language: str, max_translations: int) -> Optional[str]:
    """
    Check if an article is in the cache.
    
    Args:
        title: Article title
        language: Target language code
        max_translations: Maximum number of translations
        
    Returns:
        Path to cached file if it exists, None otherwise
    """
    cache_key = get_cache_key(title, language, max_translations)
    cache_path = get_cache_path(cache_key)
    
    if os.path.exists(cache_path):
        print(f"Cache hit: {cache_path}")
        return cache_path
    
    return None

def save_to_cache(title: str, language: str, max_translations: int, html_content: str) -> str:
    """
    Save an article to the cache.
    
    Args:
        title: Article title
        language: Target language code
        max_translations: Maximum number of translations
        html_content: HTML content to cache
        
    Returns:
        Path to the cached file
    """
    cache_key = get_cache_key(title, language, max_translations)
    cache_path = get_cache_path(cache_key)
    
    # Create cache directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Write the HTML content to the cache file
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Saved to cache: {cache_path}")
    return cache_path

def main():
    """Main function to run the Wikipedia article synthesizer."""
    parser = argparse.ArgumentParser(description='Synthesize Wikipedia articles from different languages')
    parser.add_argument('title', help='Title of the Wikipedia article')
    parser.add_argument('language', help='Target language code (e.g., en, es, fr)')
    parser.add_argument('--max_translations', type=int, default=5, 
                        help='Maximum number of translations to process (default: 5)')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--api_key', help='Claude API key (optional, overrides default)')
    parser.add_argument('--threads', type=int, default=10,
                        help='Number of parallel threads for translation (default: 10)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (always generate fresh content)')
    
    args = parser.parse_args()
    
    # Update API key if provided
    global CLAUDE_API_KEY
    if args.api_key:
        CLAUDE_API_KEY = args.api_key
    
    # Check cache first unless no-cache is specified
    if not args.no_cache:
        cached_path = check_cache(args.title, args.language, args.max_translations)
        if cached_path:
            print(f"Using cached version from {cached_path}")
            
            if args.output:
                # Copy cache file to output if specified
                import shutil
                shutil.copy2(cached_path, args.output)
                print(f"Copied to {args.output}")
                
            else:
                # Print the cached content
                with open(cached_path, 'r', encoding='utf-8') as f:
                    print("\nCACHED ARTICLE:")
                    print("=" * 80)
                    print(f.read())
                    print("=" * 80)
                    
            return
    
    # Create Anthropic client
    client = Anthropic(api_key=CLAUDE_API_KEY)
    
    print(f"Step 1: Retrieving article '{args.title}' in {args.language}...")
    original_text, langlinks = get_wikipedia_article_with_tool(args.title, args.language)
    
    if not original_text or not langlinks:
        print(f"Error: Could not find article '{args.title}' in {args.language}")
        return
    
    print(f"Found article with {len(langlinks)} translations")
    
    # Select most relevant languages using Claude
    print(f"Step 2: Selecting the most relevant languages for this topic...")
    relevant_languages = select_relevant_languages(
        client, 
        args.title, 
        args.language, 
        langlinks, 
        max_translations=args.max_translations
    )
    
    # Create a list of (language, title) tuples for selected languages
    translations = []
    for lang_link in langlinks:
        if lang_link["language"] in relevant_languages:
            translations.append((lang_link["language"], lang_link["title"]))
    
    # Add the source language
    translations.append((args.language, args.title))
    
    print(f"Step 3: Retrieving content of translated articles...")
    translation_content = get_translation_content_with_tool(translations)
    
    print(f"Step 4: Translating articles in parallel (using {args.threads} threads)...")
    translated_articles = {}
    
    # Add the original language version first
    translated_articles[args.language] = original_text
    
    # Prepare arguments for parallel processing
    translation_args = []
    for lang, content in translation_content.items():
        if lang != args.language:  # Skip source language
            translation_args.append((client, content, args.language, lang))
    
    # Use ThreadPool for parallel translations
    from multiprocessing.dummy import Pool as ThreadPool
    
    with ThreadPool(args.threads) as pool:
        # Map worker function to arguments
        results = pool.map(translate_article_worker, translation_args)
        
        # Process results
        for lang, translated in results:
            if translated is not None:
                translated_articles[lang] = translated
    
    print(f"Step 5: Synthesizing {len(translated_articles)} articles...")
    synthesized_article = synthesize_with_claude(client, translated_articles, args.language, args.title)
    
    if synthesized_article.startswith("Synthesis failed:"):
        print(f"Error: {synthesized_article}")
        return
    
    # Save to cache
    cache_path = save_to_cache(args.title, args.language, args.max_translations, synthesized_article)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(synthesized_article)
        print(f"Synthesized article saved to {args.output}")
    else:
        print("\nSYNTHESIZED ARTICLE:")
        print("=" * 80)
        print(synthesized_article)
        print("=" * 80)

if __name__ == "__main__":
    main()