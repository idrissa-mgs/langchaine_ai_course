import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


load_dotenv()

tools = [TavilySearch()]

llm = ChatOllama( model="gemma3:270m")

react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools=tools, prompt=react_prompt)

chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def main():
    result = chain.invoke(
        input={
            "input": "What is the latest news about Elon Musk? Summarize it in a few sentences."
        }
    )
    print(result)



if __name__ == "__main__":
    main()

