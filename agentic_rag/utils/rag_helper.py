from langchain_core.prompts import ChatPromptTemplate

INSTRUCTIONS = """
Your task is to only answer questions from the course participants
based on the provided context. 


Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION:
{question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    def __init__(
        self,
        index,
        llm_model,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
    ):
        self.index = index
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template

        # Load Ollama model
        self.llm = llm_model

        # Prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{instructions}"),
                ("human", "{prompt}"),
            ]
        )

        self.chain = self.chat_prompt | self.llm

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query=query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append(f"Q: {doc['question']}")
            lines.append(f"A: {doc['answer']}")
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context,
        )

    def llm_response(self, prompt):
        response = self.chain.invoke(
            {
                "instructions": self.instructions,
                "prompt": prompt,
            }
        )

        return response.content

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        return self.llm_response(prompt)
