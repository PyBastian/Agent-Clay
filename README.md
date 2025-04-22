# Financial News Ingestor Documentation

## Overview
This tool fetches financial news for portfolio assets and stores them in a ChromaDB vector database for AI-powered portfolio optimization. It supports multiple data sources including Yahoo Finance and RSS feeds.

## Prerequisites
- pip package manager

## Setup Instructions

### 1. Create Python Virtual Environment
```bash
python -m venv fintech-env
source fintech-env/bin/activate
pip install -r requirements.txt
```
## Now to get the news

``` bash
python3 news_ingestor.py --backfill-days 7 --count 20
```