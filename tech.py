#!/usr/bin/env python3
import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, TypedDict, Union

import networkx as nx
import matplotlib.pyplot as plt

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# === CLI Arguments ===
parser = argparse.ArgumentParser(
    description="Financial Portfolio Assistant with LangGraph + Ollama"
)
parser.add_argument(
    '--save-graph', action='store_true', help='Save workflow graph image'
)
parser.add_argument(
    '--model', default='llama2', help='Ollama model name'
)
parser.add_argument(
    '--embed-model', default='sentence-transformers/all-mpnet-base-v2',
    help='Embedding model name'
)
args = parser.parse_args()

# === Embedding & Vector Store Setup ===
embedding = HuggingFaceEmbeddings(model_name=args.embed_model)
vectorstore = Chroma(
    persist_directory="chroma_financial_news",
    embedding_function=embedding,
)
retriever = vectorstore.as_retriever()

# === Use Ollama LLM ===
llm = OllamaLLM(model=args.model)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

# === Initial Portfolio ===
INITIAL_PORTFOLIO: Dict[str, float] = {
    "EquityTech Fund": 2000,
    "GreenEnergy ETF": 1500,
    "HealthBio Stocks": 1000,
    "Global Bonds Fund": 1000,
    "CryptoIndex": 1000,
    "RealEstate REIT": 1000,
    "Emerging Markets Fund": 1000,
    "AI & Robotics ETF": 500,
    "Commodities Basket": 500,
    "Cash Reserve": 500
}

# === Graph State ===
class GraphState(TypedDict):
    question: str
    intent: str
    confirm: Union[str, None]
    portfolio: Dict[str, float]
    answer: Union[str, None]
    optimized_portfolio: Union[Dict[str, float], None]
    narrative: Union[str, None]
    show: Union[Dict[str, float], None]

# === Node Functions ===

def classify_intent_node(state: GraphState) -> GraphState:
    """Classify user intent: show_portfolio, optimize_portfolio, or answer_question."""
    prompt = (
        f"You are a query classifier. Given: '{state['question']}', respond with one of: "
        "show_portfolio, optimize_portfolio, answer_question."
    )
    intent = llm.invoke(prompt).strip()
    logger.info(f"Intent classified as: {intent}")
    return {**state, "intent": intent}


def confirm_node(state: GraphState) -> GraphState:
    """Prompt for confirmation before optimization."""
    resp = input("Do you want to proceed with portfolio optimization? (yes/no) > ")
    return {**state, "confirm": resp.strip().lower()}


def show_portfolio_node(state: GraphState) -> GraphState:
    """Display current portfolio holdings."""
    logger.info("Executing show_portfolio_node")
    return {**state, "show": state["portfolio"]}


def answer_question_node(state: GraphState) -> GraphState:
    """Answer arbitrary financial questions using the news vector store."""
    logger.info(f"Answering question: {state['question']}")
    answer = qa_chain.invoke(state["question"])
    return {**state, "answer": answer}


def optimize_portfolio_node(state: GraphState) -> GraphState:
    """Optimize portfolio and generate narrative."""
    logger.info("Optimizing portfolio based on news...")
    prompt = (
        "You are a financial analyst. Suggest an optimized allocation for $10,000 "
        f"based on news. Current: {json.dumps(state['portfolio'], indent=2)}. "
        "Return JSON: {'allocation': {...}, 'narrative': '...'}"
    )
    result = llm.invoke(prompt)
    try:
        data = json.loads(result)
        allocation = data.get('allocation', state['portfolio'])
        narrative = data.get('narrative', '')
    except json.JSONDecodeError:
        logger.warning("Invalid JSON, using existing allocation.")
        allocation = state['portfolio']
        narrative = ''
    # Persist change
    os.makedirs("portfolio", exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f"portfolio/portfolio_{ts}.json"
    with open(path, 'w') as f:
        json.dump({'allocation': allocation, 'narrative': narrative}, f, indent=2)
    logger.info(f"Saved new portfolio to {path}")
    return {**state, "optimized_portfolio": allocation, "narrative": narrative}

# === Build Workflow ===
workflow = StateGraph(GraphState)
workflow.add_node('classify_intent', classify_intent_node)
workflow.add_node('confirm', confirm_node)
workflow.add_node('show_portfolio', show_portfolio_node)
workflow.add_node('answer_question', answer_question_node)
workflow.add_node('optimize_portfolio', optimize_portfolio_node)
workflow.set_entry_point('classify_intent')
workflow.add_conditional_edges(
    'classify_intent', lambda s: s['intent'],
    {
        'show_portfolio': 'show_portfolio',
        'answer_question': 'answer_question',
        'optimize_portfolio': 'confirm'
    }
)
workflow.add_conditional_edges(
    'confirm', lambda s: s['confirm'],
    {'yes': 'optimize_portfolio', 'no': END}
)
workflow.add_edge('show_portfolio', END)
workflow.add_edge('answer_question', END)
workflow.add_edge('optimize_portfolio', END)
graph = workflow.compile()

# === Visualization ===
def visualize_graph(wf: StateGraph) -> str:
    G = nx.DiGraph()
    for name, node in wf.nodes.items():
        G.add_node(name)
        if hasattr(node, 'edges'):
            for nxt in node.edges:
                G.add_edge(name, nxt)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000,
            font_size=10, arrowsize=15)
    os.makedirs('graph_images', exist_ok=True)
    img = f"graph_images/langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(img, bbox_inches='tight')
    plt.close()
    logger.info(f"Graph image saved to {img}")
    return img

# === Main ===
if __name__ == '__main__':
    logger.info('Starting Financial Portfolio Assistant')
    question = input('Enter your question or command:\n> ')
    state: GraphState = {
        'question': question,
        'intent': '',
        'confirm': None,
        'portfolio': INITIAL_PORTFOLIO,
        'answer': None,
        'optimized_portfolio': None,
        'narrative': None,
        'show': None
    }
    if args.save_graph:
        img = visualize_graph(workflow)
        print(f'Graph saved to {img}')
    result = graph.invoke(state)

    # Output
    if result.get('show') is not None:
        print('\n📂 Current Portfolio:\n', json.dumps(result['show'], indent=2))
    if result.get('answer'):
        print('\n🧠 Financial Answer:\n', result['answer'])
    if result.get('optimized_portfolio'):
        print('\n💹 Optimized Portfolio:\n', json.dumps(result['optimized_portfolio'], indent=2))
    if result.get('narrative'):
        print('\n📑 Narrative:\n', result['narrative'])

    # === Sample Test Prompts ===
    sample_prompts = [
        'Show me my current portfolio',
        'What is the latest news on AI & Robotics ETF?',
        'Optimize my portfolio based on recent financial news',
        'Should I rebalance my portfolio now?',
        'Explain the rationale behind the investment decisions'
    ]
    print('\n🔎 Sample Prompts for Testing:')
    for sp in sample_prompts:
        print(f'- {sp}')
