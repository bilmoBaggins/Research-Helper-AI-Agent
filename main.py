from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from my_tools import search_tool, wiki_tool, save_tool, save_to_txt

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a comprehensive research report on a given topic.
            Answer the user's query and use the tools to gather information.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    
    formatted_output = (
        f"Topic: {structured_response.topic}\n\n"
        f"Summary:\n{structured_response.summary}\n\n"
        f"Sources:\n- " + "\n- ".join(structured_response.sources) + "\n\n"
        f"Tools Used:\n- " + "\n- ".join(structured_response.tools_used)
    )

    save_to_txt(formatted_output)
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "\nRaw response:", raw_response)