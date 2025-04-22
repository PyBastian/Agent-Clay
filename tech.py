#!/usr/bin/env python3
import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, TypedDict, Any
import subprocess

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# === Configure Enhanced Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("portfolio_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === CLI Arguments ===
parser = argparse.ArgumentParser(
    description="AI Financial Portfolio Optimization Assistant"
)
parser.add_argument('--save-graph', action='store_true', help='Save workflow diagram')
parser.add_argument('--model', default='mistral', help='Ollama model name')
parser.add_argument('--embed-model', default='sentence-transformers/all-mpnet-base-v2',
                    help='Embedding model name')
args = parser.parse_args()

# === Financial Knowledge Base ===
embedding = HuggingFaceEmbeddings(model_name=args.embed_model)
vectorstore = Chroma(
    persist_directory="chroma_financial_news",
    embedding_function=embedding,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# === LLM Configuration ===
llm = OllamaLLM(model=args.model, temperature=0, format="json")

# === Portfolio Configuration ===
INITIAL_PORTFOLIO = {
    "EquityTech Fund": 2000.00,
    "GreenEnergy ETF": 1500.00,
    "HealthBio Stocks": 1000.00,
    "Global Bonds Fund": 1000.00,
    "CryptoIndex": 1000.00,
    "RealEstate REIT": 1000.00,
    "Emerging Markets Fund": 1000.00,
    "AI & Robotics ETF": 500.00,
    "Commodities Basket": 500.00,
    "Cash Reserve": 500.00
}

# === State Definition ===
class GraphState(TypedDict):
    question: str
    intent: str
    confirm: Any
    portfolio: Dict[str, float]
    answer: Any
    show: Any
    optimization_result: Any

# === Enhanced Validation System ===
def validate_portfolio(portfolio: Dict[str, float]) -> Dict[str, Any]:
    """Comprehensive portfolio validation with risk analysis"""
    total = sum(portfolio.values())
    validation = {
        "is_valid": True,
        "total": total,
        "violations": [],
        "risk_metrics": {
            "low_risk_assets": portfolio.get("Global Bonds Fund", 0.0) + portfolio.get("Cash Reserve", 0.0),
            "high_risk_assets": sum(v for k, v in portfolio.items() 
                               if k in {"CryptoIndex", "Emerging Markets Fund"})
        }
    }
    
    # Total validation
    if abs(total - 10000) > 0.01:
        validation["violations"].append(f"Total ${total:.2f} ≠ $10,000")
        validation["is_valid"] = False
    
    # Asset-level validation
    for asset, value in portfolio.items():
        if value < 0:
            validation["violations"].append(f"Negative allocation in {asset}")
            validation["is_valid"] = False
    
    return validation

# === Enhanced Intent Classification ===
intent_prompt = PromptTemplate.from_template("""
Analyze the user's financial query and classify it into exactly one of these intents:
<show_portfolio|answer_question|optimize_portfolio>

Guidelines:
1. show_portfolio: Contains "show", "display", "current", "holdings", "portfolio"
2. answer_question: News-related, market trends, explanations, "what", "why", "how", considering that recent information are more usefull
3. optimize_portfolio: Contains "optimize", "rebalance", "improve", "adjust allocation"

Examples:
- "Show my current investments" → <intent>show_portfolio</intent>
- "Explain recent tech stock trends" → <intent>answer_question</intent>
- "Rebalance based on market news" → <intent>optimize_portfolio</intent>

Query: "{question}"
Intent:""")

def classify_intent_node(state: GraphState) -> GraphState:
    """AI-powered intent classification with validation"""
    response = llm.invoke(intent_prompt.format(question=state["question"]))
    response_dict = json.loads(response)
    # Access the intent
    intent = response_dict.get("intent")
    valid_intents = {"show_portfolio", "answer_question", "optimize_portfolio"}
    print(intent)
    if intent not in valid_intents:
        logger.warning(f"Invalid intent '{intent}', defaulting to answer_question")
        intent = "answer_question"
    
    return {**state, "intent": intent}

@tool
def optimize_portfolio_tool(portfolio: Dict[str, float]) -> Dict[str, Any]:
    """Modern Portfolio Theory-based optimization with news analysis."""
    try:
        # Step 1: Fetch more diverse and recent financial news
        docs = retriever.get_relevant_documents("portfolio optimization financial news trends risk sectors inflation tech energy")
        unique_sources = list({d.metadata.get("source", ""): d for d in docs}.values())  # Dedup by source
        context = "\n".join(f"- {d.page_content}" for d in unique_sources[:5])

        # Step 2: Define clearer prompt structure
        portfolio_sample = json.dumps(portfolio, indent=2)

        prompt = f"""
        You are a Chief Investment Officer (CIO) responsible for optimizing client portfolios using Modern Portfolio Theory (MPT),
        incorporating current market trends and risk metrics.

        === Market Context ===
        The following news articles highlight current macroeconomic and sector-specific trends considerin as relevant news:
        {context}

        === Client Portfolio (Total: $10,000) ===
        The portfolio contains these assets and current allocations:
        {portfolio_sample}

        === Objective ===
        Optimize the portfolio using diversification principles and risk/return balance.

        === Constraints ===
        - Total allocation must exactly equal $10,000
        - Rebalance to include sectors showing growth or resilience
        - Explain reasoning with technical insight (Markdown format)

        === Required Response Format (JSON) ===
        {{
            "allocation": {{
                "EquityTech Fund": int,
                "GreenEnergy ETF": int,
                "HealthBio Stocks": int,
                "Global Bonds Fund": int,
                "CryptoIndex": int,
                "RealEstate REIT": int,
                "Emerging Markets Fund": int,
                "AI & Robotics ETF": int,
                "Commodities Basket": int,
                "Cash Reserve": int
            }},
            "narrative": "## Optimization Rational and Detailed reasoning here so its made like final inform of the decision taken mentioning the news that are use for that reasoning and allocation of the portofoio",
        }}
        """

        response = llm.invoke(prompt)
        result = json.loads(response)

        # Step 4: Fallback to original portfolio if invalid
        proposed = result.get("allocation", portfolio)
        validation = validate_portfolio(proposed)

        # Step 5: Audit log
        ts = datetime.now().isoformat()
        audit_data = {
            "timestamp": ts,
            "original": portfolio,
            "proposed": proposed,
            "validation": validation,
            "narrative": result.get("narrative", ""),
            "risk_metrics": result.get("risk_metrics", {})
        }

        os.makedirs("portfolio_audits", exist_ok=True)
        with open(f"portfolio_audits/{ts}.json", "w") as f:
            json.dump(audit_data, f, indent=2)

        return audit_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse optimization result: {str(e)}")
        logger.debug(f"Raw response: {response}")
        return {
            "error": "Invalid optimization response",
            "original": portfolio
        }

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return {
            "error": "Optimization process failed",
            "original": portfolio
        }

# === Workflow Nodes ===
def confirm_action_node(state: GraphState) -> GraphState:
    """User confirmation with input validation"""
    resp = input("Confirm portfolio optimization? (y/n) > ").strip().lower()
    return {**state, "confirm": "yes" if resp in {"y", "yes"} else "no"}

def show_portfolio_node(state: GraphState) -> GraphState:
    """Portfolio display with formatting"""
    return {**state, "show": state["portfolio"]}

def answer_question_node(state: GraphState) -> GraphState:
    """Enhanced financial Q&A with context"""
    qa_prompt = PromptTemplate.from_template("""
    Answer this financial question using only the provided context:
    
    Context:
    {context}
    
    Question: {question}
    
    Respond in Markdown format with sources.
    """)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return {**state, "answer": qa_chain.invoke(state["question"])}

# === Workflow Construction ===
workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("confirm_action", confirm_action_node)
workflow.add_node("show_portfolio", show_portfolio_node)
workflow.add_node("answer_question", answer_question_node)
workflow.add_node("optimize_portfolio", ToolNode([optimize_portfolio_tool]))

# Configure workflow transitions
workflow.set_entry_point("classify_intent")
workflow.add_edge(START, "classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    lambda s: s["intent"],
    {
        "show_portfolio": "show_portfolio",
        "answer_question": "answer_question",
        "optimize_portfolio": "optimize_portfolio"
    }
)

workflow.add_conditional_edges(
    "optimize_portfolio",
    lambda s: s["confirm"],
    {"yes": "confirm_action", "no": END}
)

workflow.add_edge("show_portfolio", END)
workflow.add_edge("answer_question", END)
workflow.add_edge("optimize_portfolio", "confirm_action")
workflow.add_edge("confirm_action", END)

def save_workflow_diagram(graph):
    from IPython.display import Image, display
    output_path = "graph.png"
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)

        # Optionally display it
        display(Image(filename=output_path))
    except Exception as e:
        print(f"Error rendering or saving graph: {e}")
# === Main Execution Loop ===
if __name__ == "__main__":
    # Compile and visualize workflow
    app = workflow.compile()
    
    if args.save_graph:
        diagram_path = save_workflow_diagram(app)
        logger.info(f"Workflow diagram saved to {diagram_path}")

    # Initialize portfolio
    current_portfolio = INITIAL_PORTFOLIO.copy()
    
    while True:
        try:
            query = input('\n💰 Financial Query (or "exit"):\n> ')
            if query.lower() in {"exit", "quit"}:
                break
                
            # Execute workflow
            result = app.invoke({
                "question": query,
                "portfolio": current_portfolio,
                "intent": "",
                "confirm": None,
                "answer": None,
                "show": None
            })
            
            # Process results
            if result.get("show"):
                print("\n📊 Current Portfolio:")
                for asset, val in result["show"].items():
                    print(f"- {asset}: ${val:,.2f}")
                print(f"Total: ${sum(result['show'].values()):,.2f}")
                
            if result.get("answer"):
                print("\n📈 Market Analysis:")
                print(result["answer"])
                
            if "optimization_result" in result:
                print('ciomoooo')
                opt = result["optimization_result"]
                print("\n🔄 Optimization Result:")
                print(f"Proposed Total: ${sum(opt['proposed'].values()):,.2f}")
                
                if opt['validation']['is_valid']:
                    print("✅ Valid Portfolio")
                    current_portfolio = opt["proposed"]
                    print("\nRisk Metrics:")
                else:
                    print("❌ Invalid Portfolio:")
                    print("\n".join(opt['validation']['violations']))
                
                print("\n📝 Investment Rationale:")
                print(opt.get("narrative", ""))
                
        except KeyboardInterrupt:
            print("\n🚨 Operation cancelled")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            continue

    print("\n💼 Final Portfolio:")
    for asset, val in current_portfolio.items():
        print(f"- {asset}: ${val:,.2f}")
    print(f"Total Value: ${sum(current_portfolio.values()):,.2f}")