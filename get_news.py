import logging
import argparse
from typing import List, Dict, Tuple, Set
from datetime import datetime, timedelta

import yfinance as yf
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils import embedding_functions

# Optional: install feedparser for RSS backfill
try:
    import feedparser
except ImportError:
    feedparser = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Portfolio mapping: instrument -> list of tickers
PORTFOLIO: Dict[str, List[str]] = {
    'EquityTech': [
        'AAPL',  # Apple - hardware, services
        'MSFT',  # Microsoft - software, cloud
        'GOOGL', # Google - AI, cloud, ads
        'NVDA',  # Nvidia - chips, AI
        'AMZN',  # Amazon - e-commerce, cloud
        'CRM',   # Salesforce - SaaS
    ],
    'GreenEnergy': [
        'FSLR',  # First Solar
        'ENPH',  # Enphase Energy
        'NEE',   # NextEra Energy
        'ICLN',  # iShares Global Clean Energy ETF
        'SEDG',  # SolarEdge
        'PLUG',  # Plug Power - hydrogen
    ],
    'HealthBio': [
        'PFE',   # Pfizer
        'MRNA',  # Moderna
        'AMGN',  # Amgen
        'JNJ',   # Johnson & Johnson
        'XBI',   # Biotech ETF
        'REGN',  # Regeneron
    ],
    'GlobalBonds': [
        'BND',   # Total Bond Market
        'AGG',   # Core US Bond ETF
        'TIP',   # Treasury Inflation-Protected Securities
        'EMB',   # Emerging Markets Bonds
        'IBND',  # International Corp Bonds
    ],
    'CryptoIndex': [
        'BTC-USD',  # Bitcoin
        'ETH-USD',  # Ethereum
        'SOL-USD',  # Solana
        'AVAX-USD', # Avalanche
        'LINK-USD', # Chainlink
    ],
    'RealEstate': [
        'VNQ',  # Vanguard REIT ETF
        'PLD',  # Prologis - industrial REIT
        'SPG',  # Simon Property Group - retail REIT
        'AMT',  # American Tower - infrastructure
        'O',    # Realty Income - diversified REIT
    ],
    'EmergingMarkets': [
        'EEM',  # iShares Emerging Markets ETF
        'VWO',  # Vanguard Emerging Markets ETF
        'FXI',  # China Large-Cap
        'INDA', # India
        'EWZ',  # Brazil
    ],
    'AI_Robotics': [
        'BOTZ', # Global Robotics & AI ETF
        'ROBO', # Robotics & Automation Index ETF
        'SOXL', # Leveraged Semiconductor
        'AIQ',  # Global X AI & Technology ETF
        'PATH', # UiPath - automation
    ],
    'Commodities': [
        'GLD',  # Gold
        'USO',  # Oil
        'DBC',  # Commodity Index ETF
        'SLV',  # Silver
        'PALL', # Palladium
    ],
    'CashReserve': [
        'BIL',  # 1-3 Month Treasury Bill ETF
        'SGOV',# Short-term Treasury
        'SHV', # Short-term Treasury
    ],
}


# Initialize ChromaDB persistent client and collection
def init_chromadb(collection_name: str = "financial_news") -> Tuple[chromadb.PersistentClient, chromadb.api.models.Collection.Collection]:
    """
    Initialize PersistentClient for on-disk storage and get/create a collection with HuggingFace embeddings.
    Returns the client and collection for ingestion.
    """
    logger.info("Initializing ChromaDB persistent client...")
    print(DEFAULT_DATABASE,DEFAULT_TENANT)
    client = chromadb.PersistentClient(
        path="./chromadb_storage",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # Use all-mpnet-base-v2 model for embeddings
    hf_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Create or get the collection, providing the embedding function
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=hf_embedder,
    )
    logger.info(f"Collection '{collection_name}' initialized with HuggingFace embeddings.")
    return client, collection

# Fetch news articles for a ticker, with optional backfill via RSS
def fetch_news_for_ticker(
    ticker: str,
    count: int,
    backfill_days: int = 0
) -> List[Dict]:
    """
    Fetch news for a given ticker. If backfill_days>0 and feedparser is available,
    use Yahoo Finance RSS to retrieve articles up to backfill_days ago.
    Otherwise use yfinance's .news for the latest news.
    """
    logger.debug(f"Fetching {count} news items for ticker: {ticker} (backfill_days={backfill_days})")
    articles = []

    if backfill_days > 0 and feedparser:
        cutoff = datetime.utcnow() - timedelta(days=backfill_days)
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(rss_url)
        entries = []
        for entry in feed.entries:
            # parse published time
            published = entry.get('published_parsed')
            if not published:
                continue
            dt = datetime.fromtimestamp(
                datetime(*published[:6]).timestamp()
            )
            if dt >= cutoff:
                entries.append((dt, entry))
        # Sort by most recent first
        entries.sort(key=lambda x: x[0], reverse=True)
        raw_items = [e[1] for e in entries][:count]
        for i, item in enumerate(raw_items):
            doc_id = f"{ticker}-rss-{i}-{item.get('id', str(i))}"
            title = item.get('title', '')
            summary = item.get('summary', '')
            url = item.get('link', '')
            provider_time = int(datetime.fromtimestamp(item.published_parsed and datetime(*item.published_parsed[:6]).timestamp()).timestamp()) if item.get('published_parsed') else 0
            articles.append({
                'id': doc_id,
                'content': f"{title}. {summary}",
                'metadata': {
                    'ticker': ticker,
                    'providerPublishTime': provider_time,
                    'url': url,
                }
            })
    else:
        logger.info("yF")
        # Use yfinance for latest news
        search = yf.Ticker(ticker).news or []
        for i, item in enumerate(search[:count]):
            doc_id = f"{ticker}-{i}"
            title = item.get('title') or ''
            summary = item.get('summary') or ''
            provider_time = item.get('providerPublishTime') or 0
            url = item.get('link') or ''
            articles.append({
                'id': doc_id,
                'content': f"{title}. {summary}",
                'metadata': {
                    'ticker': ticker,
                    'providerPublishTime': provider_time,
                    'url': url,
                }
            })
    logger.info(f"Fetched {len(articles)} articles for {ticker}")
    return articles

# Ingest documents into ChromaDB, skipping duplicates if desired
def ingest_documents(
    collection: chromadb.api.models.Collection.Collection,
    docs: List[Dict],
    skip_ids: Set[str] = None
) -> None:
    """
    Insert documents into the given ChromaDB collection, skipping any ids in skip_ids.
    """
    skip_ids = skip_ids or set()
    docs_to_add = [d for d in docs if d['id'] not in skip_ids]
    if not docs_to_add:
        logger.info("No new documents to ingest.")
        return
    ids = [d['id'] for d in docs_to_add]
    contents = [d['content'] for d in docs_to_add]
    metadatas = [d['metadata'] for d in docs_to_add]
    logger.debug(f"Ingesting {len(ids)} new documents...")
    collection.add(
        documents=contents,
        metadatas=metadatas,
        ids=ids
    )
    logger.info(f"Successfully ingested {len(ids)} documents.")

# Main orchestration function
def main():
    """
    Orchestrates fetching and ingesting news, with options for dedupe and backfill.
    """
    parser = argparse.ArgumentParser(description="Fetch and ingest financial news into ChromaDB.")
    parser.add_argument(
        '--count', '-c', type=int, default=10,
        help='Number of news articles to fetch per ticker'
    )
    parser.add_argument(
        '--dedupe', '-d', action='store_true',
        help='Skip articles already in the database'
    )
    parser.add_argument(
        '--backfill-days', '-b', type=int, default=0,
        help='Fetch news items up to N days ago via RSS (requires feedparser)'
    )
    args = parser.parse_args()

    client, collection = init_chromadb()
    for tickers in PORTFOLIO.values():
        for ticker in tickers:
            try:
                # Determine existing IDs if dedupe enabled
                existing_ids: Set[str] = set()
                if args.dedupe:
                    try:
                        existing = collection.get(where={'ticker': ticker}, include=['ids'])
                        existing_ids = set(existing['ids'])
                    except Exception:
                        existing_ids = set()

                news = fetch_news_for_ticker(
                    ticker,
                    count=args.count,
                    backfill_days=args.backfill_days
                )
                ingest_documents(collection, news, skip_ids=existing_ids)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
    logger.info("ChromaDB ingestion complete.")

if __name__ == "__main__":
    main()
