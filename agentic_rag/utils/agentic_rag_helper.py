from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
import json


INSTRUCTIONS = """
You are a teaching assistant for the course.

Your purpose is to answer questions related ONLY to the course.

You can answer questions about:
- course logistics
- lectures
- assignments
- projects
- deadlines
- policies
- learning materials
- course concepts

You have access to a search tool that retrieves information from the course knowledge base.

Rules:

1. For every course-related question, ALWAYS use the search tool first.

2. When searching:
   - Use important keywords from the student's question.
   - If the first search is not enough, perform additional searches.
   - Try alternative keywords, synonyms, and related concepts or paraphrased questions to find relevant information.

3. Use only information returned by the search tool.
   Do not invent answers.

4. If the search results do not contain the answer, respond:
"I don't know based on the available course information."

5. If the question is unrelated to the course, respond:
"This question is not relevant to the purpose of this assistant. I am designed to answer questions about the course only."

6. Keep answers concise and accurate.
""".strip()


class AgenticRAGBase:
    def __init__(
        self,
        index,
        llm_model,
        course="llm-zoomcamp",
    ):

        self.index = index
        self.course = course

        @tool
        def search(query: str) -> str:
            """
            Search the course knowledge base.

            Use this tool for questions about:
            - course registration
            - assignments
            - deadlines
            - lectures
            - projects
            - course concepts
            - course policies

            Input should be a concise search query.
            """

            docs = self.index.search(
                query=query,
                num_results=5,
                boost_dict={
                    "question": 3.0,
                    "section": 0.5,
                },
                filter_dict={
                    "course": self.course,
                },
            )

            if not docs:
                return "No relevant course information found."

            results = []

            for doc in docs:
                results.append(
                    {
                        "section": doc["section"],
                        "question": doc["question"],
                        "answer": doc["answer"],
                    }
                )

            return json.dumps(results, indent=2)

        self.search_tool = search

        self.llm = llm_model.bind_tools([self.search_tool])

    def chat(self, question: str):

        messages = [
            SystemMessage(content=INSTRUCTIONS),
            HumanMessage(content=question),
        ]

        while True:
            ai_msg = self.llm.invoke(messages)

            messages.append(ai_msg)

            # LLM answered directly
            if not ai_msg.tool_calls:
                return ai_msg.content

            # Execute requested tools
            for tool_call in ai_msg.tool_calls:
                if tool_call["name"] == "search":
                    result = self.search_tool.invoke(tool_call["args"])

                    messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"],
                        )
                    )
