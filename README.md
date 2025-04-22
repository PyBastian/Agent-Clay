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
python3 get_news.py --backfill-days 7 --count 20
```
## To run the agent.

First you will need to use Ollama 

```bash
ollama pull mistral && ollama serve
```

``` bash
python3 tech.py --save-graph
```
