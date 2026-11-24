import feedparser
import requests
import time
import re
import json
import email.utils
import dateutil.parser
from dataclasses import dataclass, field as dataclass_field
from calendar import timegm
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from tqdm import tqdm
import logging
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, MofNCompleteColumn

from .models import Feed, Article
from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Feed entries published just before the previous complete read are re-examined
# on the next read. Duplicate suppression is by GUID anyway, and publisher
# timestamps are frequently a little off; the overlap closes that gap.
READ_WINDOW_OVERLAP = timedelta(hours=1)

USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
)



def call_openrouter_api(prompt: str, api_key: str, model: str) -> Optional[str]:
    """
    Call the OpenRouter API with a prompt.
    
    Args:
        prompt: Prompt to send to the API
        api_key: OpenRouter API key
        model: Model to use
        
    Returns:
        Response from the API or None if failed
    """
    if not api_key:
        logger.error("OpenRouter API key not configured")
        return None
    
    # Import modules here to avoid circular imports
    from .cost_tracker import track_api_call
    from .config import Config
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        response_data = response.json()
        
        # Track the cost of this API call if enabled
        config = Config()
        if config.cost_tracking_enabled:
            track_api_call(response_data, model)
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            logger.error(f"Invalid API response: {response_data}")
            return None
    
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        return None


def generate_summary(content: str, api_key: str, prompt_template: str, model: str) -> Optional[str]:
    """
    Generate a summary for an article.
    
    Args:
        content: Article content to summarize
        api_key: OpenRouter API key
        prompt_template: Prompt template for summarization
        model: Model to use
        
    Returns:
        Summary text or None if summarization failed
    """
    if not content:
        logger.warning("No content to summarize")
        return None
    
    prompt = prompt_template.format(content=content)
    return call_openrouter_api(prompt, api_key, model)


def analyze_value(content: str, api_key: str, prompt_template: str, model: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Analyze the value of an article.
    
    Args:
        content: Article content to analyze
        api_key: OpenRouter API key
        prompt_template: Prompt template for value analysis
        model: Model to use
        
    Returns:
        Tuple of (quality_tier, quality_score, labels)
    """
    if not content:
        return None, None, None
    
    prompt = prompt_template.format(content=content)
    response = call_openrouter_api(prompt, api_key, model)
    
    if not response:
        return None, None, None
    
    try:
        # The response is markdown front matter, not JSON
        # Parse the markdown front matter format
        quality_tier = None
        quality_score = None
        labels = None
        
        # Extract Rating (quality tier)
        rating_match = re.search(r'Rating:: ([SABCD])', response)
        if rating_match:
            quality_tier = rating_match.group(1)
        
        # Extract QualityScore
        score_match = re.search(r'QualityScore:: (\d+)', response)
        if score_match:
            try:
                quality_score = int(score_match.group(1))
            except ValueError:
                quality_score = None
        
        # Extract Labels
        labels_match = re.search(r'Labels:: (.*?)$', response, re.MULTILINE)
        if labels_match:
            labels = labels_match.group(1).strip()
        
        return quality_tier, quality_score, labels
    
    except Exception as e:
        logger.error(f"Failed to parse value analysis: {str(e)}")
        logger.debug(f"Response content: {response}")
        return None, None, None


def clean_html(content: str) -> str:
    """Clean HTML or markdown content for storage.
    
    Args:
        content: HTML or markdown content to clean
        
    Returns:
        Cleaned text content
    """
    # If content is empty, return empty string
    if not content:
        return ""
    
    # Check if content is likely markdown (from Jina.ai)
    markdown_indicators = [
        content.startswith('#'),  # Headers
        '**' in content,  # Bold
        '*' in content,  # Italic or list
        '```' in content,  # Code blocks
        '>' in content,  # Blockquotes
        '- ' in content,  # Lists
        '[' in content and '](' in content  # Links
    ]
    
    if any(markdown_indicators):
        # For markdown, preserve it but clean up excessive whitespace
        # Remove any HTML tags that might be mixed in
        content = re.sub(r'<[^>]+>', ' ', content)
        # Fix common Jina.ai artifacts
        content = re.sub(r'\[\s*\]\(\s*\)', '', content)  # Empty links
        content = re.sub(r'\[\s*\]\(http[^)]+\)', '', content)  # Links with empty text
        # Remove excessive newlines (more than 2 in a row)
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Remove excessive whitespace
        content = re.sub(r'\s{2,}', ' ', content)
        return content.strip()
    
    # For HTML content
    # More comprehensive HTML cleaning
    # Remove script and style elements completely
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    # Replace common HTML entities
    content = content.replace('&nbsp;', ' ')
    content = content.replace('&amp;', '&')
    content = content.replace('&lt;', '<')
    content = content.replace('&gt;', '>')
    content = content.replace('&quot;', '"')
    content = content.replace('&apos;', "'")
    # Remove all remaining HTML tags
    clean_text = re.sub(r'<[^>]+>', ' ', content)
    # Normalize whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


def fetch_jina_content(url: str) -> Optional[str]:
    """Fetch content from a URL using Jina.ai's content extraction service.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        Markdown content or None if fetching failed
    """
    start_time = time.time()
    logger.info(f"Fetching content from Jina.ai for URL: {url}")
    
    try:
        # Use direct URL to ensure proper content extraction
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        jina_start_time = time.time()
        logger.debug(f"Sending request to Jina.ai: {jina_url}")
        response = requests.get(jina_url, headers=headers, timeout=15)
        jina_duration = time.time() - jina_start_time
        
        if response.status_code == 200:
            # Jina.ai returns the content in markdown format
            content = response.text
            
            # Verify we actually got content and not just a tiny response
            if len(content) < 100:
                logger.warning(f"Jina.ai returned suspiciously short content ({len(content)} chars) for URL: {url} in {jina_duration:.2f}s")
                # Try direct URL as fallback
                direct_start_time = time.time()
                logger.info(f"Trying direct URL as fallback: {url}")
                direct_response = requests.get(url, headers=headers, timeout=15)
                direct_duration = time.time() - direct_start_time
                
                if direct_response.status_code == 200 and len(direct_response.text) > len(content):
                    logger.info(f"Using direct URL content instead of Jina.ai for URL: {url} ({len(direct_response.text)} chars, fetched in {direct_duration:.2f}s)")
                    content = direct_response.text
                else:
                    logger.info(f"Direct URL fallback didn't provide better content ({len(direct_response.text)} chars) in {direct_duration:.2f}s")
            
            total_duration = time.time() - start_time
            logger.info(f"Successfully fetched content from Jina.ai for URL: {url} ({len(content)} chars) in {total_duration:.2f}s")
            return content
        else:
            logger.warning(f"Failed to fetch content from Jina.ai for URL: {url}. Status code: {response.status_code} after {jina_duration:.2f}s")
            
            # Try direct URL as fallback
            try:
                direct_start_time = time.time()
                logger.info(f"Trying direct URL as fallback: {url}")
                direct_response = requests.get(url, headers=headers, timeout=15)
                direct_duration = time.time() - direct_start_time
                
                if direct_response.status_code == 200:
                    logger.info(f"Using direct URL content as fallback for URL: {url} ({len(direct_response.text)} chars, fetched in {direct_duration:.2f}s)")
                    return direct_response.text
                else:
                    logger.warning(f"Direct URL fallback failed with status code: {direct_response.status_code} after {direct_duration:.2f}s")
            except Exception as direct_err:
                logger.warning(f"Direct URL fallback also failed for URL: {url}. Error: {str(direct_err)}")
            
            total_duration = time.time() - start_time
            logger.warning(f"All content fetching attempts failed for URL: {url} after {total_duration:.2f}s")
            return None
    except Exception as e:
        total_duration = time.time() - start_time
        logger.warning(f"Error fetching content from Jina.ai for URL: {url}. Error: {str(e)} after {total_duration:.2f}s")
        return None


def extract_origin_url(description: str) -> Optional[str]:
    """Extract the origin article URL from a feed description.
    
    Args:
        description: The description text from the feed entry
        
    Returns:
        The origin URL if found, None otherwise
    """
    if not description:
        return None
    
    # Common patterns for finding URLs in descriptions
    # Pattern for <a href="URL">...
    href_pattern = re.compile(r'<a\s+(?:[^>]*?\s+)?href=(["\'])(https?://[^"\'>]+)\1', re.IGNORECASE)
    href_matches = href_pattern.findall(description)
    
    # If we found href links, return the first one (usually the main article link)
    if href_matches and len(href_matches) > 0:
        return href_matches[0][1]  # Return the URL part from the first match
    
    # Pattern for bare URLs
    url_pattern = re.compile(r'https?://[^\s<>"\')]+', re.IGNORECASE)
    url_matches = url_pattern.findall(description)
    
    # If we found bare URLs, return the first one
    if url_matches and len(url_matches) > 0:
        return url_matches[0]
    
    return None


def extract_content(entry: Dict[str, Any]) -> str:
    """Extract the main content from a feed entry."""
    content = ""
    
    # Try different content fields in order of preference
    # Some feeds put the full content in the 'content' field
    if 'content' in entry and entry['content']:
        for content_item in entry['content']:
            if 'value' in content_item:
                content = content_item['value']
                # If content is substantial, return it
                if len(content) > 500:  # Arbitrary threshold for "substantial" content
                    return content
    
    # Try other common fields
    if not content and 'summary_detail' in entry and 'value' in entry['summary_detail']:
        content = entry['summary_detail']['value']
    
    if not content and 'summary' in entry:
        content = entry['summary']
    
    if not content and 'description' in entry:
        content = entry['description']
    
    # Some feeds include content in non-standard fields
    if not content and 'content_encoded' in entry:
        content = entry['content_encoded']
    
    # Some feeds include content in the 'value' field directly
    if not content and 'value' in entry:
        content = entry['value']
        
    # Log if we couldn't find any content
    if not content:
        logger.warning(f"No content found in feed entry: {entry.get('title', 'Unknown title')}")
        # Print available fields for debugging
        logger.debug(f"Available fields: {list(entry.keys())}")
    
    return content


def _as_naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to naive UTC.

    Feed dates arrive from several parsers, some of which return aware
    datetimes. The database and the rest of the ingestion path use naive UTC,
    so aware values are converted rather than compared across timezones.
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def parse_published_date(entry: Dict[str, Any]) -> Optional[datetime]:
    """Parse the published date from a feed entry."""
    # First try the pre-parsed date fields from feedparser
    for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if date_field in entry and entry[date_field]:
            try:
                time_struct = entry[date_field]
                # Validate the time struct to avoid out of range errors
                if time_struct and len(time_struct) >= 9:
                    year = time_struct[0]
                    # Basic validation to avoid mktime errors
                    if 1970 <= year <= 2100:  # Reasonable year range
                        try:
                            return datetime.fromtimestamp(timegm(time_struct), timezone.utc).replace(tzinfo=None)
                        except (OverflowError, ValueError) as e:
                            logger.debug(f"Time struct conversion error: {str(e)} for {time_struct}")
            except (TypeError, ValueError) as e:
                logger.debug(f"Error parsing {date_field}: {str(e)}")
                continue

    # Try string date fields if parsed fields not available or failed
    for date_field in ['published', 'updated', 'created']:
        if date_field in entry and entry[date_field] and isinstance(entry[date_field], str):
            date_str = entry[date_field].strip()

            # Try multiple date parsing approaches
            parsers = [
                # ISO format with Z timezone
                lambda d: datetime.fromisoformat(d.replace('Z', '+00:00')),
                # RFC 2822 format using email.utils. Feed timestamps are UTC, so
                # interpret the parsed struct as UTC rather than local time; a
                # local-time reading would shift the read window by the tz offset.
                lambda d: datetime.fromtimestamp(email.utils.mktime_tz(email.utils.parsedate_tz(d)),
                                                 timezone.utc),
                # Use dateutil parser as a fallback
                lambda d: dateutil.parser.parse(d),
                # Try feedparser's own date parser
                lambda d: datetime.fromtimestamp(timegm(feedparser._parse_date(d)), timezone.utc)
            ]

            for parser in parsers:
                try:
                    parsed = parser(date_str)
                except (ValueError, TypeError, AttributeError, OverflowError):
                    continue
                normalized = _as_naive_utc(parsed)
                if normalized is not None:
                    return normalized

    # As a last resort, check if there's a date in the title or description
    for field in ['title', 'description', 'summary']:
        if field in entry and entry[field] and isinstance(entry[field], str):
            # Look for common date patterns in the text
            try:
                # Use dateutil's parser with fuzzy matching
                return _as_naive_utc(dateutil.parser.parse(entry[field], fuzzy=True))
            except (ValueError, TypeError):
                pass

    return None


@dataclass
class FetchResult:
    """Outcome of one attempt to read a feed's source document.

    ``status`` is one of:

    - ``ok``: HTTP 200 (or a local document) parsed with at least one entry.
    - ``empty``: the document was readable but contained no entries. Treated as
      a *temporary* condition: the source state is not advanced, so the next
      round still covers the gap.
    - ``not_modified``: HTTP 304. The cached validators describe the current
      document, so nothing needs to be ingested, but no state is cleared.
    - ``error``: transport, HTTP, or parse failure. The previous window and
      validators are kept so the next round can catch up.
    """

    status: str
    entries: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    http_status: Optional[int] = None
    fetched_at: datetime = dataclass_field(default_factory=datetime.utcnow)
    error: Optional[str] = None


# A bozo exception of this type only means the server's charset was overridden;
# the document itself is still usable.
_ACCEPTABLE_BOZO_TYPES: Tuple[type, ...] = (feedparser.CharacterEncodingOverride,)


def _request_headers(etag: Optional[str], last_modified: Optional[str]) -> Dict[str, str]:
    """Build request headers, adding conditional validators when known."""
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/rss+xml, application/atom+xml, application/xml;q=0.9, '
                  'text/xml;q=0.9, */*;q=0.8',
    }
    if etag:
        headers['If-None-Match'] = etag
    if last_modified:
        headers['If-Modified-Since'] = last_modified
    return headers


def _parse_feed_bytes(payload: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Parse feed bytes.

    Returns:
        Tuple of (entries, error). ``entries`` is None when the payload could
        not be parsed at all; an empty list means a valid but empty document.
    """
    try:
        feed_data = feedparser.parse(payload)
    except Exception as exc:  # feedparser is defensive, but be safe
        return None, f"feed parse failed: {exc}"

    bozo_exception = getattr(feed_data, 'bozo_exception', None)
    if getattr(feed_data, 'bozo', False) and bozo_exception is not None:
        if isinstance(bozo_exception, _ACCEPTABLE_BOZO_TYPES):
            logger.debug(f"Acceptable feed parsing warning: {bozo_exception}")
        elif feed_data.entries:
            logger.warning(f"Feed parsing warning: {bozo_exception}")
        else:
            return None, f"feed parse failed: {bozo_exception}"

    return list(feed_data.entries), None


def fetch_feed(
    feed_url: str,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 2,
) -> FetchResult:
    """Read a feed document, using HTTP conditional requests when possible.

    The function never raises for network/server problems; it reports them via
    ``FetchResult.status``. Callers decide what the outcome means for the
    source state.

    Args:
        feed_url: URL of the RSS/Atom feed
        etag: Previously stored ETag, sent as ``If-None-Match``
        last_modified: Previously stored Last-Modified value, sent as
            ``If-Modified-Since``
        max_retries: Maximum number of attempts for transient failures
        retry_delay: Delay between attempts in seconds

    Returns:
        FetchResult describing the outcome
    """
    # Local files and other non-HTTP sources have no conditional semantics.
    if not re.match(r'^https?://', feed_url, re.IGNORECASE):
        entries, parse_error = _parse_feed_bytes(feed_url)
        if entries is None:
            return FetchResult(status="error", error=parse_error or "feed parse failed")
        return FetchResult(status="ok" if entries else "empty", entries=entries)

    headers = _request_headers(etag, last_modified)
    last_error: Optional[str] = None

    for attempt in range(max_retries):
        try:
            response = requests.get(feed_url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            last_error = f"request failed: {exc}"
            if attempt < max_retries - 1:
                logger.warning(
                    f"{last_error} for {feed_url}, retrying in {retry_delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
                continue
            break

        fetched_at = datetime.utcnow()
        status = response.status_code

        if status == 304:
            if not (etag or last_modified):
                # A 304 without a conditional request means the server is
                # misbehaving and we have no cached document to fall back on.
                last_error = "HTTP 304 without cached validators"
                logger.warning(f"{last_error} for {feed_url}")
                return FetchResult(status="error", error=last_error, http_status=304,
                                   fetched_at=fetched_at)
            logger.info(f"Feed not modified since last complete read: {feed_url}")
            return FetchResult(
                status="not_modified",
                etag=response.headers.get('ETag') or etag,
                last_modified=response.headers.get('Last-Modified') or last_modified,
                http_status=304,
                fetched_at=fetched_at,
            )

        # Retry transient server-side conditions; other 4xx are permanent.
        if status in (408, 429) or status >= 500:
            last_error = f"HTTP {status}"
            if attempt < max_retries - 1:
                logger.warning(
                    f"{last_error} fetching {feed_url}, retrying in {retry_delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
                continue
            logger.error(f"Failed to fetch feed {feed_url}: {last_error} "
                         f"after {max_retries} attempts")
            return FetchResult(status="error", error=last_error, http_status=status,
                               fetched_at=fetched_at)

        if status >= 400:
            last_error = f"HTTP {status}"
            logger.error(f"Failed to fetch feed {feed_url}: {last_error}")
            return FetchResult(status="error", error=last_error, http_status=status,
                               fetched_at=fetched_at)

        entries, parse_error = _parse_feed_bytes(response.content)
        if entries is None:
            logger.warning(f"{parse_error} for {feed_url}")
            return FetchResult(status="error", error=parse_error, http_status=status,
                               fetched_at=fetched_at)

        return FetchResult(
            status="ok" if entries else "empty",
            entries=entries,
            etag=response.headers.get('ETag'),
            last_modified=response.headers.get('Last-Modified'),
            http_status=status,
            fetched_at=fetched_at,
        )

    return FetchResult(status="error", error=last_error or "unknown fetch error")


def compute_window_start(feed: Feed, lookback_days: int, now: datetime) -> datetime:
    """Compute the start of the window to read for a feed.

    The window always reaches back at least to the last complete read (minus a
    small overlap), so nothing that was pending during a failed or unchanged
    round is skipped. A larger configured lookback still widens the window, and
    the lookback alone applies when the feed has never been read completely.
    """
    lookback_start = now - timedelta(days=lookback_days)
    if not feed.last_complete_read_at:
        return lookback_start
    resumed = feed.last_complete_read_at - READ_WINDOW_OVERLAP
    if resumed > now:
        # Guard against a stored timestamp in the future (clock skew).
        resumed = now - READ_WINDOW_OVERLAP
    return min(lookback_start, resumed)


def filter_entries_by_window(entries: List[Dict[str, Any]], window_start: datetime) -> List[Dict[str, Any]]:
    """Keep entries published within the window.

    Entries without a parseable date are kept; duplicate suppression by GUID
    handles anything already ingested.
    """
    selected = []
    for entry in entries:
        published_date = parse_published_date(entry)
        if published_date is None or published_date >= window_start:
            selected.append(entry)
    return selected


def conditional_validators(feed: Feed) -> Tuple[Optional[str], Optional[str]]:
    """Return the HTTP validators to send for a feed, if any.

    Validators describe the representation seen at the last *complete* read and
    only resume from that same point while no work is outstanding. When a
    previous round fetched a document but did not finish handling it (the
    per-round cap left entries behind), the stored validators point at an older
    representation, so a 304 could hide the deferred entries. In that case no
    validators are sent and the document is read in full.
    """
    if feed.pending_backlog or feed.last_fetch_status == "capped":
        return None, None
    return feed.http_etag, feed.http_last_modified


def advance_source_state(feed: Feed, result: FetchResult, window_start: datetime) -> None:
    """Record a complete read: HTTP validators and the read window move together.

    Only called when the whole document was read and its entries were handled,
    so the next round can safely resume from here. Validators are taken from
    this read as-is: if the server stopped sending one, the stale value is
    cleared rather than kept for a future conditional request.
    """
    feed.http_etag = result.etag
    feed.http_last_modified = result.last_modified
    feed.last_complete_read_at = result.fetched_at
    feed.last_window_start = window_start
    feed.last_fetch_status = "ok"
    feed.pending_backlog = False
    feed.error_count = 0
    feed.last_error = None


def get_feed_entries(feed_url: str, lookback_days: int = 7, max_retries: int = 3, retry_delay: int = 2) -> List[Dict[str, Any]]:
    """
    Fetch and parse a feed, returning entries within lookback period.

    This is the stateless read path: it sends no conditional headers and keeps
    no source state. Ingest uses :func:`fetch_feed` directly so that HTTP
    validators and the read window are carried on the Feed row.

    Args:
        feed_url: URL of the RSS feed
        lookback_days: Number of days to look back for articles
        max_retries: Maximum number of retry attempts for transient errors
        retry_delay: Delay between retries in seconds

    Returns:
        List of feed entries within the lookback period
    """
    result = fetch_feed(feed_url, max_retries=max_retries, retry_delay=retry_delay)
    if result.status in ("error", "not_modified"):
        return []
    window_start = datetime.utcnow() - timedelta(days=lookback_days)
    return filter_entries_by_window(result.entries, window_start)


def process_feed(
    feed: Feed, 
    db_session: Session, 
    lookback_days: int = 7,
    max_articles: int = 25,
    progress_callback: Optional[Callable[[str], None]] = None,
    max_errors: int = 5,  # Maximum number of consecutive errors before marking feed as inactive
    retry_attempts: int = 3,  # Number of retry attempts for transient errors
    config: Optional[Config] = None,  # Configuration for article analysis
    analyze_content: bool = True  # Whether to analyze content during ingestion
) -> Tuple[int, int, int, int]:
    """
    Process a feed, fetch new articles and store them in the database.
    
    Args:
        feed: Feed object
        db_session: SQLAlchemy database session
        lookback_days: Number of days to look back for articles
        max_articles: Maximum number of articles to process per feed
        progress_callback: Optional callback function for progress updates
        max_errors: Maximum number of consecutive errors before marking feed as inactive
        retry_attempts: Number of retry attempts for transient errors
        config: Configuration for article analysis
        analyze_content: Whether to analyze content during ingestion
        
    Returns:
        Tuple of (new_articles, jina_enhanced, with_summary, with_value_analysis)
    """
    if feed.muted:
        if progress_callback:
            progress_callback(f"Skipping muted feed: {feed.title}")
        return 0, 0, 0, 0
    
    if progress_callback:
        progress_callback(f"Processing feed: {feed.title}")

    # Record that we looked, but not that we read anything: the read window and
    # validators only move forward once a read completes.
    feed.last_checked = datetime.utcnow()

    try:
        # Read the source document. HTTP validators are sent when the feed has
        # them; the read window starts from the last complete read so anything
        # pending from a failed or 304 round is picked up here.
        window_start = compute_window_start(feed, lookback_days, datetime.utcnow())
        etag, last_modified = conditional_validators(feed)
        result = fetch_feed(
            feed.url,
            etag=etag,
            last_modified=last_modified,
            max_retries=retry_attempts,
        )

        if result.status == "not_modified":
            # The cached document is still current. Keep the existing window and
            # validators; only note that the source was confirmed unchanged. No
            # stored data is cleared: the 304 re-affirms what was already read.
            # A 304 may return refreshed validators; keep them if so.
            if result.etag:
                feed.http_etag = result.etag
            if result.last_modified:
                feed.http_last_modified = result.last_modified
            feed.last_complete_read_at = result.fetched_at
            feed.last_window_start = window_start
            feed.last_fetch_status = "not_modified"
            feed.no_entries_count = 0
            feed.error_count = 0
            feed.last_error = None
            db_session.commit()
            return 0, 0, 0, 0

        if result.status == "error":
            # Leave the window and validators untouched so the next round reads
            # the same span again.
            feed.last_fetch_status = "error"
            feed.error_count = getattr(feed, 'error_count', 0) + 1
            feed.last_error = result.error or "feed fetch failed"
            logger.error(f"Error processing feed {feed.title}: {feed.last_error}")

            if feed.error_count >= max_errors:
                logger.warning(f"Feed {feed.title} has had {feed.error_count} consecutive errors. Marking as muted.")
                feed.muted = True
                feed.muted_reason = f"Auto-muted after {feed.error_count} consecutive errors. Last error: {feed.last_error}"

            db_session.commit()
            return 0, 0, 0, 0

        if result.status == "empty":
            # A readable but empty document is treated as temporary: do not
            # advance the window past content a later response may include.
            feed.last_fetch_status = "empty"
            feed.no_entries_count = getattr(feed, 'no_entries_count', 0) + 1
            if feed.no_entries_count > 5:  # After 5 consecutive checks with no entries
                logger.warning(f"Feed {feed.title} has had no entries for {feed.no_entries_count} consecutive checks")
            db_session.commit()
            return 0, 0, 0, 0

        # result.status == "ok": the document was read completely.
        entries = filter_entries_by_window(result.entries, window_start)

        if not entries:
            # The document is current and its entries all precede the window.
            # This is a complete read: the window moves forward.
            feed.no_entries_count = 0
            advance_source_state(feed, result, window_start)
            db_session.commit()
            return 0, 0, 0, 0

        # Reset no entries counter if we found entries
        feed.no_entries_count = 0

        # Sort entries by date if possible, newest first
        sorted_entries = []
        for entry in entries:
            published_date = parse_published_date(entry)
            sorted_entries.append((entry, published_date or datetime.min))

        sorted_entries.sort(key=lambda x: x[1], reverse=True)

        # Drop entries that are already stored before applying the per-round
        # cap. The cap then only limits new work; anything it leaves behind
        # keeps the source state short of a complete read so the next round
        # resumes the same span.
        candidate_ids = []
        candidate_urls = []
        for entry, _ in sorted_entries:
            guid = entry.get('id', entry.get('link', ''))
            if not guid:
                continue
            candidate_ids.append(guid)
            candidate_urls.append(entry.get('link', guid))

        known_ids: set = set()
        known_urls: set = set()
        if candidate_ids:
            for start in range(0, len(candidate_ids), 500):
                id_chunk = candidate_ids[start:start + 500]
                url_chunk = candidate_urls[start:start + 500]
                rows = db_session.query(Article.guid, Article.url).filter(
                    Article.guid.in_(id_chunk) | Article.url.in_(url_chunk)
                ).all()
                known_ids.update(guid for guid, _ in rows)
                known_urls.update(url for _, url in rows)

        pending_entries = []
        for entry, published_date in sorted_entries:
            guid = entry.get('id', entry.get('link', ''))
            if not guid:
                continue
            article_url = entry.get('link', guid)
            if guid in known_ids or article_url in known_urls:
                continue
            pending_entries.append((entry, published_date))

        capped_entries = pending_entries[:max_articles]
        skipped_entries = pending_entries[max_articles:]

        new_article_count = 0
        jina_enhanced_count = 0
        with_summary_count = 0
        with_value_count = 0

        for entry, published_date in capped_entries:
            guid = entry.get('id', entry.get('link', ''))
            article_url = entry.get('link', guid)

            # Extract content from feed entry
            content = extract_content(entry)
            
            # Try to fetch content from Jina.ai in these cases:
            # 1. No content was found in the feed
            # 2. Content is too short (likely truncated)
            # 3. Content appears to be just a summary or snippet
            jina_enhanced = False
            should_fetch_jina = (
                not content or 
                len(content) < 1000 or  # Content is suspiciously short
                "[...]".lower() in content.lower() or  # Content contains ellipsis indicating truncation
                "read more".lower() in content.lower() or  # Content has "read more" prompt
                "continue reading".lower() in content.lower()  # Content has "continue reading" prompt
            )
            
            # Check if this is an aggregator feed that needs to peer through to the origin article
            if feed.peer_through and entry.get('description'):
                origin_url = extract_origin_url(entry.get('description', ''))
                if origin_url and origin_url != article_url:
                    logger.info(f"Feed is marked for peer-through. Using origin URL: {origin_url} instead of {article_url}")
                    article_url = origin_url
            
            if should_fetch_jina and article_url.startswith('http'):
                logger.info(f"Content may be incomplete for {article_url}, attempting to fetch from Jina.ai")
                jina_content = fetch_jina_content(article_url)
                
                if jina_content and (not content or len(jina_content) > len(content)):
                    logger.info(f"Using Jina.ai content for {article_url} - {len(jina_content)} chars vs original {len(content)} chars")
                    content = jina_content
                    jina_enhanced = True
                    jina_enhanced_count += 1
            
            clean_content = clean_html(content)
            
            # Create new article
            new_article = Article(
                feed_id=feed.id,  # Ensure feed_id is set correctly
                title=entry.get('title', 'Untitled'),
                url=entry.get('link', ''),
                guid=guid,
                description=entry.get('summary', ''),
                content=clean_content,
                author=entry.get('author', ''),
                published_at=published_date,
                fetched_at=datetime.utcnow(),
                processed=False,  # Will be set to True if we analyze it now
                word_count=len(clean_content.split()),
                jina_enhanced=jina_enhanced  # Set flag if content was fetched from Jina.ai
            )
            
            # Log article information
            word_count = len(clean_content.split())
            logger.info(f"New article: '{new_article.title[:50]}' ({word_count} words, {len(clean_content)} chars)")
            if jina_enhanced:
                logger.info(f"  - Content enhanced with Jina.ai")
            if new_article.author:
                logger.info(f"  - Author: {new_article.author}")
            if published_date:
                logger.info(f"  - Published: {published_date.strftime('%Y-%m-%d %H:%M')}")
            
            # Double-check that feed_id is set correctly
            if not new_article.feed_id:
                logger.warning(f"feed_id not set for article {new_article.title}. Setting manually.")
                new_article.feed_id = feed.id
            
            # Analyze the article if requested and we have a configuration
            if analyze_content and config and clean_content:
                if progress_callback:
                    progress_callback(f"Analyzing article: {new_article.title[:30]}")
                
                analysis_start_time = time.time()
                
                # Generate summary
                if config.openrouter_api_key and config.openrouter_prompt:
                    logger.info(f"Generating summary for article: {new_article.title[:50]}")
                    summary_start_time = time.time()
                    summary = generate_summary(
                        clean_content,
                        config.openrouter_api_key,
                        config.openrouter_prompt,
                        config.openrouter_model
                    )
                    summary_duration = time.time() - summary_start_time
                    if summary:
                        new_article.summary = summary
                        with_summary_count += 1
                        logger.info(f"Summary generated successfully in {summary_duration:.2f}s ({len(summary)} chars)")
                    else:
                        logger.warning(f"Failed to generate summary after {summary_duration:.2f}s")
                
                # Analyze value
                if config.openrouter_api_key and config.value_prompt_enabled and config.value_prompt:
                    logger.info(f"Analyzing value for article: {new_article.title[:50]}")
                    value_start_time = time.time()
                    quality_tier, quality_score, labels = analyze_value(
                        clean_content,
                        config.openrouter_api_key,
                        config.value_prompt,
                        config.openrouter_processing_model or config.openrouter_model
                    )
                    value_duration = time.time() - value_start_time
                    
                    new_article.quality_tier = quality_tier
                    new_article.quality_score = quality_score
                    new_article.labels = labels
                    
                    if quality_tier:
                        with_value_count += 1
                        logger.info(f"Value analysis completed in {value_duration:.2f}s: Tier {quality_tier}, Score {quality_score}, Labels: {labels}")
                    else:
                        logger.warning(f"Value analysis failed after {value_duration:.2f}s")
                
                # Mark as processed
                new_article.processed = True
                new_article.processed_at = datetime.utcnow()
                
                total_analysis_duration = time.time() - analysis_start_time
                logger.info(f"Total analysis time for article '{new_article.title[:50]}': {total_analysis_duration:.2f}s")
            
            db_session.add(new_article)
            new_article_count += 1
        
        # Update feed last updated timestamp if new articles were found
        if new_article_count > 0:
            feed.last_updated = datetime.utcnow()

        if skipped_entries:
            # The per-round cap left work behind. The read did not cover the
            # whole window, so the source state is not advanced; the next round
            # resumes from the same point and picks these up.
            dated = [d for _, d in skipped_entries if d != datetime.min]
            oldest_note = (f", oldest {min(dated):%Y-%m-%d %H:%M}" if dated else "")
            feed.last_fetch_status = "capped"
            feed.pending_backlog = True
            feed.last_error = None
            feed.error_count = 0
            logger.info(
                f"Feed {feed.title}: {len(skipped_entries)} entries deferred by the "
                f"per-round cap{oldest_note}; window will be retried next round"
            )
        else:
            # The document was read completely: advance the HTTP validators and
            # the read window together so the next round resumes from here.
            advance_source_state(feed, result, window_start)

        db_session.commit()
        return new_article_count, jina_enhanced_count, with_summary_count, with_value_count

    except Exception as e:
        logger.error(f"Error processing feed {feed.title}: {str(e)}")
        # Update error information. The read window and validators stay as they
        # were, so a partial run is retried over the same span.
        feed.last_fetch_status = "error"
        feed.error_count = getattr(feed, 'error_count', 0) + 1
        feed.last_error = str(e)
        
        # Check if we should mark the feed as inactive due to too many errors
        if feed.error_count >= max_errors:
            logger.warning(f"Feed {feed.title} has had {feed.error_count} consecutive errors. Marking as muted.")
            feed.muted = True
            feed.muted_reason = f"Auto-muted after {feed.error_count} consecutive errors. Last error: {feed.last_error}"
        
        db_session.commit()
        return 0, 0, 0, 0


def process_all_feeds(
    db_session: Session, 
    lookback_days: int = 7,
    max_articles_per_feed: int = 25,
    max_errors: int = 5,
    retry_attempts: int = 3,
    config: Optional[Config] = None,
    analyze_content: bool = True
) -> Dict[str, int]:
    """
    Process all active feeds in the database.
    
    Args:
        db_session: SQLAlchemy database session
        lookback_days: Number of days to look back for articles
        max_articles_per_feed: Maximum number of articles to process per feed
        max_errors: Maximum number of consecutive errors before marking feed as inactive
        retry_attempts: Number of retry attempts for transient errors
        config: Configuration for article analysis
        analyze_content: Whether to analyze content during ingestion
        
    Returns:
        Dictionary with processing statistics
    """
    feeds = db_session.query(Feed).filter_by(muted=False).all()

    if not feeds:
        logger.warning("No active feeds found in database")
        return {
            "feeds_processed": 0,
            "new_articles": 0,
            "auto_muted": 0,
            "jina_enhanced": 0,
            "with_summary": 0,
            "with_value_analysis": 0,
            "not_modified": 0,
            "deferred": 0,
            "errors": 0,
        }
    
    total_new_articles = 0
    auto_muted_count = 0
    total_jina_enhanced = 0
    total_with_summary = 0
    total_with_value = 0
    not_modified_count = 0
    deferred_count = 0
    error_count = 0
    # Create a beautiful Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="green"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[bold]{task.fields[stats]}"),
        expand=True
    ) as progress:
        # Add the main task
        main_task = progress.add_task(f"[bold]Processing {len(feeds)} feeds", total=len(feeds), stats="")

        for feed in feeds:
            # Check if feed was muted before processing
            was_muted_before = feed.muted

            # Update the progress description with feed title
            progress.update(main_task, description=f"[bold blue]Processing: [cyan]{feed.title[:40]}")

            # Add timing information
            feed_start_time = time.time()

            # Create a progress callback that updates the task description
            def update_progress(msg):
                progress.update(main_task, description=f"[bold blue]{msg[:40]}")

            # Process the feed
            new_articles, jina_enhanced, with_summary, with_value = process_feed(
                feed,
                db_session,
                lookback_days=lookback_days,
                max_articles=max_articles_per_feed,
                progress_callback=update_progress,
                max_errors=max_errors,
                retry_attempts=retry_attempts,
                config=config,
                analyze_content=analyze_content
            )

            # Calculate processing time
            feed_duration = time.time() - feed_start_time

            # Update stats field with colorful information, including how the
            # source behaved this round (304 / deferred / error).
            status = feed.last_fetch_status or ""
            stats_text = (
                f"[green]new:[/green] {new_articles} [yellow]jina:[/yellow] {jina_enhanced} "
                f"[blue]summary:[/blue] {with_summary} [magenta]value:[/magenta] {with_value} "
                f"[cyan]time:[/cyan] {feed_duration:.1f}s"
            )
            if status:
                stats_text += f" [white]fetch:[/white] {status}"
            progress.update(main_task, stats=stats_text)

            total_new_articles += new_articles
            total_jina_enhanced += jina_enhanced
            total_with_summary += with_summary
            total_with_value += with_value

            if status == "not_modified":
                not_modified_count += 1
            elif status == "capped":
                deferred_count += 1
            elif status == "error":
                error_count += 1

            # Check if feed was auto-muted during processing
            if not was_muted_before and feed.muted:
                auto_muted_count += 1

            # Advance the progress
            progress.update(main_task, advance=1)

    # Log summary of auto-muted feeds
    if auto_muted_count > 0:
        logger.warning(f"Auto-muted {auto_muted_count} feeds due to persistent errors")

    return {
        "feeds_processed": len(feeds),
        "new_articles": total_new_articles,
        "auto_muted": auto_muted_count,
        "jina_enhanced": total_jina_enhanced,
        "with_summary": total_with_summary,
        "with_value_analysis": total_with_value,
        "not_modified": not_modified_count,
        "deferred": deferred_count,
        "errors": error_count,
    }