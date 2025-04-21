import os
import json
from datetime import datetime
from typing import Dict, TypedDict, Union

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode  # FIX: correct import

# === Embedding & Vector Store Setup ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma(
    persist_directory="chroma_financial_news",
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever()

# === Use Ollama LLM ===
llm = OllamaLLM(model="llama2")  # Updated import + class name

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# === Initial Portfolio ===
INITIAL_PORTFOLIO = {
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

# === Tool: Answer Financial Question ===
@tool
def answer_financial_question(question: str) -> str:
    """Answer a financial question using the vectorstore with recent news."""
    return qa_chain.run(question)

# === Tool: Optimize Portfolio ===
@tool
def optimize_portfolio(portfolio: Dict[str, float]) -> Dict[str, float]:
    """Optimize the user's portfolio based on financial news and return a new allocation."""
    reasoning_prompt = f"""
You are a financial analyst AI. Based on recent financial news, suggest an optimized allocation for a $10,000 portfolio.
Here is the current portfolio:\n{json.dumps(portfolio, indent=2)}.
Respond ONLY with a JSON object of the new allocation.
"""
    result = qa_chain.run(reasoning_prompt)

    try:
        updated_portfolio = json.loads(result)
    except Exception:
        updated_portfolio = portfolio  # fallback to old portfolio if parsing fails

    # Save to a timestamped .txt
    os.makedirs("portfolio", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"portfolio/portfolio_{timestamp}.txt", "w") as f:
        f.write(json.dumps(updated_portfolio, indent=2))
    
    return updated_portfolio

# === LangGraph State ===
class GraphState(TypedDict):
    question: str
    portfolio: Dict[str, float]
    answer: Union[str, None]
    optimized_portfolio: Union[Dict[str, float], None]

# === Decision logic ===
def decide_next_action(state: GraphState) -> str:
    if "optimize" in state["question"].lower():
        return "optimize_portfolio"
    return "answer_question"

# === Build LangGraph ===
workflow = StateGraph(GraphState)

workflow.add_node("answer_question", ToolNode(answer_financial_question))  # FIX
workflow.add_node("optimize_portfolio", ToolNode(optimize_portfolio))      # FIX
workflow.add_node("decide", decide_next_action)

workflow.set_entry_point("decide")

workflow.add_conditional_edges("decide", decide_next_action, {
    "answer_question": "answer_question",
    "optimize_portfolio": "optimize_portfolio"
})

workflow.add_edge("answer_question", END)
workflow.add_edge("optimize_portfolio", END)

graph = workflow.compile()

# === Run ===
if __name__ == "__main__":
    print("📈 Welcome to the Financial Portfolio Assistant (Ollama-Powered)!")
    question = input("Ask a financial question or request portfolio optimization:\n> ")

    inputs = {
        "question": question,
            "portfolio": json.dumps(INITIAL_PORTFOLIO),  # now a string
        "answer": None,
        "optimized_portfolio": None
    }

    result = graph.invoke(inputs)

    if result.get("answer"):
        print("\n🧠 Financial Answer:")
        print(result["answer"])
    elif result.get("optimized_portfolio"):
        print("\n💹 Optimized Portfolio Allocation:")
        print(json.dumps(result["optimized_portfolio"], indent=2))
