# Linguapedia

A multilingual Wikipedia synthesizer powered by AI that combines articles from multiple language editions into comprehensive, synthesized articles.

## Overview

Linguapedia takes Wikipedia articles across different languages, intelligently selects the most relevant editions for a topic, translates them all to a target language, and synthesizes them into a single comprehensive article. This creates richer, more complete articles by combining diverse perspectives and information from different language editions.

## Features

- **Multilingual Synthesis**: Combines Wikipedia articles from up to 5 different language editions
- **Intelligent Language Selection**: Uses AI to select the most relevant language editions for each topic
- **Fuzzy Search**: Smart search that handles misspellings and variations
- **Smart Caching**: Fuzzy cache matching prevents duplicate synthesis of similar articles
- **Web Interface**: Clean Flask-based UI supporting 10+ languages
- **Parallel Processing**: Concurrent translation for faster processing
- **Automatic Linking**: Generates interconnected article links for a wiki-like experience

## Supported Languages

- English (en)
- Japanese (ja)
- Russian (ru)
- German (de)
- Spanish (es)
- French (fr)
- Chinese (zh)
- Italian (it)
- Portuguese (pt)
- Polish (pl)

## Requirements

- Python 3.8+
- OpenAI API key (GPT-5-mini)
- Internet connection for Wikipedia access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd wics-anthropic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### Web Interface

Start the Flask application:
```bash
python frontend.py
```

Then open your browser to `http://localhost:5000`

### Command Line

Synthesize an article from the command line:
```bash
python backend.py "Article Title" en --max_translations 5
```

Options:
- `title`: Article title to search for
- `language`: Target language code (e.g., en, es, fr)
- `--max_translations`: Maximum number of language editions to combine (default: 5)
- `--output`: Output file path (optional)
- `--no-cache`: Bypass cache and regenerate article
- `--threads`: Number of parallel translation threads (default: 10)

### Examples

```bash
# Synthesize "Artificial Intelligence" in English
python backend.py "Artificial Intelligence" en

# Synthesize "Renaissance" in French with 3 language editions
python backend.py "Renaissance" fr --max_translations 3

# Save to file
python backend.py "Quantum Physics" en --output quantum.html

# Bypass cache
python backend.py "Climate Change" en --no-cache
```

## Architecture

### Core Components

1. **backend.py**: Core synthesis engine
   - Article retrieval with fuzzy search
   - Language selection using AI
   - Parallel translation pipeline
   - Article synthesis

2. **frontend.py**: Flask web application
   - Web interface and routing
   - Job management and status tracking
   - Caching system
   - Article browsing

3. **wikipedia_fuzzy_search.py**: Smart Wikipedia search
   - Fuzzy matching for article titles
   - Fallback to multiple languages
   - AI-powered result evaluation

4. **fuzzy_cache_match.py**: Intelligent cache matching
   - Prevents duplicate synthesis
   - AI-powered similarity detection
   - Smart redirects to existing articles

### How It Works

1. **Search**: User searches for an article by title
2. **Retrieve**: System finds the article using fuzzy search (handles typos)
3. **Select Languages**: AI selects most relevant language editions
4. **Fetch Content**: Retrieves articles from selected languages
5. **Translate**: Parallel translation of all articles to target language
6. **Synthesize**: AI combines all versions into comprehensive article
7. **Cache**: Saves result for future requests
8. **Display**: Shows synthesized article with automatic linking

## API Integration

Uses OpenAI's GPT-5-mini model for:
- Language selection decisions
- Translation of article content
- Article synthesis
- Fuzzy search evaluation
- Cache similarity matching

## Caching

Articles are cached in the `cache/` directory by language and title:
```
cache/
  en/
    Artificial_Intelligence.html
    Quantum_Physics.html
  es/
    Inteligencia_Artificial.html
```

The fuzzy cache matcher can redirect similar queries to existing cached articles, preventing unnecessary regeneration.

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Configuration

The application uses GPT-5-mini by default. To change the model, update `AI_MODEL` in:
- `backend.py`
- `wikipedia_fuzzy_search.py`
- `fuzzy_cache_match.py`

## Deployment

### Local Development

```bash
python frontend.py
```

### Production (Render, Railway, etc.)

The application includes a `render.yaml` for deployment to Render.com:

1. Set `OPENAI_API_KEY` in environment variables
2. Deploy using `gunicorn frontend:app`

## Performance

- **Fuzzy Search**: ~1-2 seconds
- **Language Selection**: ~2-3 seconds
- **Translation** (per language): ~10-20 seconds
- **Synthesis**: ~15-30 seconds
- **Total** (5 languages): ~2-3 minutes

Caching dramatically improves subsequent requests for the same or similar articles.

## Limitations

- Wikipedia API rate limits may apply
- AI API costs scale with article length and number of languages
- Very long articles may be truncated to fit token limits
- Some Wikipedia formatting may be lost in synthesis

## Contributing

Contributions are welcome! Areas for improvement:
- Additional language support
- Better error handling
- Enhanced caching strategies
- UI/UX improvements
- Support for other knowledge bases

## License

[Your chosen license]

## Acknowledgments

- Powered by [OpenAI GPT-5-mini](https://www.openai.com/)
- Wikipedia content via [Wikipedia API](https://www.mediawiki.org/wiki/API)
- Built with [Flask](https://flask.palletsprojects.com/)

## Support

For issues, questions, or contributions, please [open an issue](https://github.com/your-repo/issues) on GitHub.
