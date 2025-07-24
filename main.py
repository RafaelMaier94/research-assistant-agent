from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from IPython.display import Markdown
import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langsmith import trace


from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph
from pathlib import Path
from langchain.prompts import PromptTemplate
from langgraph.graph import MessagesState
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string

from runnables.writer_graph import build_writer_graph
from langchain_openai import ChatOpenAI



### Tavily web search
tavily_search = TavilySearchResults(max_results=3)

### LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)





def invoke_graph(graph, topic):
    max_analysts = 3
    
    thread = {"configurable": {"thread_id": 1}}
    for event in graph.invoke({"topic": topic, "max_analysts": max_analysts}, thread, stream_mode="values"):
        final_state = graph.get_state(thread)
        report = final_state.values.get('final_report')
        Markdown(report)
        print(Markdown(report))


if __name__ == "__main__":
    topic = "A produção de soja no cerrado"
    writer_graph = build_writer_graph(model, tavily_search)
    with trace("My LangGraph Agent Run"):
        invoke_graph(writer_graph, topic)
