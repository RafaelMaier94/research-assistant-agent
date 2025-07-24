import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display


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

from models.analyst import Analyst


class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")


def build_analysts_graph(model):
    def create_analysts(state: GenerateAnalystsState):
        """ Create analysts """
        topic=state["topic"]
        state["max_analysts"] = 1
        max_analysts=state["max_analysts"]
        human_analyst_feedback=state.get("human_analyst_feedback", "")

        structured_llm = model.with_structured_output(Perspectives)


        markdown_prompt = Path("prompts/analysts.md").read_text(encoding="utf-8")

        analysts_prompt = PromptTemplate.from_template(markdown_prompt)

        
        system_message = analysts_prompt.format(topic=topic, human_analyst_feedback=human_analyst_feedback, max_analysts=max_analysts)

        analysts = structured_llm.invoke([SystemMessage(content=system_message), HumanMessage(content="Generate the set of analysts")])
        return {"analysts": analysts.analysts}

    def human_feedback(state: GenerateAnalystsState):
        """ Get human feedback """
        human_analyst_feedback = input("Please provide feedback on the generated analysts (or press Enter to continue): ")
        return {"human_analyst_feedback": human_analyst_feedback}

    def should_continue(state: GenerateAnalystsState):
        """ Return the next node to execute """

        # Check if human feedback
        human_analyst_feedback = state.get("human_analyst_feedback", None)
        if human_analyst_feedback:
            return "create_analysts"
        
        return END

    builder = StateGraph(GenerateAnalystsState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

    # Compile
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
    # display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
    return graph
