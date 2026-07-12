from langchain_ollama import ChatOllama
from utils.agentic_rag_helper import AgenticRAGBase
from utils.load_dataset import build_index, load_faq_data
from utils.rag_helper import RAGBase


def main(run_rag: bool, run_agentic_rag: bool):
    """Main function to run the RAG example."""
    if not run_rag and not run_agentic_rag:
        print("Please set at least one of run_rag or run_agentic_rag to True.")
        raise ValueError("At least one of run_rag or run_agentic_rag must be True.")
    documents = load_faq_data()
    index = build_index(documents)
    # question = "I just discovered the course, can I still join?"
    question = "How can i run Olama locally?"
    llm_model = ChatOllama(model="llama3.2", temperature=0.2)

    if run_agentic_rag:
        agent_assistant = AgenticRAGBase(index, llm_model=llm_model)

        response = agent_assistant.chat(question)
        print("Agentic RAG Response:", response)
    if run_rag:
        assistant = RAGBase(index, llm_model=llm_model)

        response = assistant.rag(question)

        print("RAG Response:", response)


if __name__ == "__main__":
    main(run_rag=True, run_agentic_rag=True)
