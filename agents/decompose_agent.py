import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except Exception:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

try:
    from langchain.agents import AgentType, initialize_agent
except Exception:
    from langchain_classic.agents import AgentType, initialize_agent


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
try:
    from langchain.chains import LLMChain
except Exception:
    from langchain_classic.chains import LLMChain

try:
    from langchain.prompts import PromptTemplate
except Exception:
    from langchain_core.prompts import PromptTemplate

from langchain_community.llms import Ollama  # Import the Ollama class

try:
    from langchain.tools import Tool
except Exception:
    from langchain_core.tools import Tool



class DecomposeAgent:
    def __init__(self, config):
        self.config = config
        # Use Ollama to connect to the locally deployed Qwen2.5-7B model
        self.llm = Ollama(
            model="qwen2.5:32b",  # Ensure the model name in Ollama is correct
            temperature=0.35
        )
        self.tools = [
            Tool(
                name="Decompose",
                func=self.decompose,
                description="Decompose the query into sub-queries.",
            )
        ]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def count_intents(self, query: str) -> int:
        """
        Determine the number of intents in the input query.
        Use LLM to analyze the number of intents contained in the input text.
        Args:
            query (str): The input query text.
        Returns:
            int: The number of intents.
        """
        # Clearly specify the prompt format
        prompt = PromptTemplate.from_template(
            "Please calculate how many independent intents are contained in the following query. Return only an integer:\n{query}\nNumber of intents: "
        )
        max_attempts = 3
        for attempt in range(max_attempts):
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.run(query=query)
            try:
                return int(response.strip())
            except ValueError:
                if attempt == max_attempts - 1:
                    return 1  # If parsing fails after multiple attempts, default to 1 intent
        return 1

    def decompose(self, query: str) -> List[str]:
        """
        Decompose the query. If the number of intents is greater than 1, perform intent decomposition.
        Args:
            query (str): The input query text.
        Returns:
            List[str]: A list of decomposed sub-queries.
        """
        intent_count = self.count_intents(query)
        intent_count = min(intent_count, 3)  # Limit the number of intents to a maximum of 5
        if intent_count > 1:
            return self._split_query(query)
        # return [query]
        return query

    def _split_query(self, query: str) -> List[str]:
        """
        The method that actually performs query decomposition.
        Args:
            query (str): The input query text.
        Returns:
            List[str]: A list of decomposed sub-queries.
        """
        prompt = PromptTemplate.from_template(
            "Split the following query into multiple independent sub-queries, separated by '||', without additional explanations:\n{query}\nList of sub-queries: "
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(query=query)
        return [q.strip() for q in response.split("||") if q.strip()]


# def run_decomposition(agent: DecomposeAgent, query: str) -> List[str]:
#     return agent.decompose(query)


# # ---------------------- Run tests ----------------------
# if __name__ == "__main__":
#     class MockConfig:
#         pass
#     config = MockConfig()
#     agent = DecomposeAgent(config)
#     query = "Check today's weather in Shanghai, then summarize the latest scientific research news from Fudan University, and finally compare the advantages and disadvantages of Python and Java."
#     subqueries = run_decomposition(agent, query)
#     print("Decomposed sub-queries:")
#     for i, subq in enumerate(subqueries, 1):
#         print(f"{i}. {subq}")