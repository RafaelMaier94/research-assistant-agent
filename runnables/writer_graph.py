import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.constants import Send
import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display, Markdown


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
from runnables.analysts_graph import GenerateAnalystsState, Perspectives, build_analysts_graph
from runnables.interview_graph import build_interview_graph

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report




def build_writer_graph(model, search_engine):
    def initiate_all_interviews(state: ResearchGraphState):
        """ This is the "map" step where we run each interview sub-graph using Send API """    
        # Check if human feedback
        human_analyst_feedback=state.get('human_analyst_feedback')
        if human_analyst_feedback:
            # Return to create_analysts
            return "create_analysts"

        # Otherwise kick off interviews in parallel via Send() API
        else:
            topic = state["topic"]
            return [Send("conduct_interview", {"analyst": analyst,
                                            "messages": [HumanMessage(
                                                content=f"So you said you were writing an article on {topic}?"
                                            )
                                                        ]}) for analyst in state["analysts"]]

    def write_report(state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        markdown_prompt = Path("prompts/write_report.md").read_text(encoding="utf-8")

        write_report_prompt = PromptTemplate.from_template(markdown_prompt)
        # Summarize the sections into a final report
        system_message = write_report_prompt.format(topic=topic, context=formatted_str_sections)    
        report = model.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
        return {"content": report.content}


    def write_introduction(state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        markdown_prompt = Path("prompts/write_intro_conclusion.md").read_text(encoding="utf-8")

        intro_conclusion_prompt = PromptTemplate.from_template(markdown_prompt)
        # Summarize the sections into a final report
        
        instructions = intro_conclusion_prompt.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        intro = model.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
        return {"introduction": intro.content}

    def write_conclusion(state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        markdown_prompt = Path("prompts/write_intro_conclusion.md").read_text(encoding="utf-8")

        intro_conclusion_prompt = PromptTemplate.from_template(markdown_prompt)
        # Summarize the sections into a final report
        
        instructions = intro_conclusion_prompt.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        conclusion = model.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
        return {"conclusion": conclusion.content}

    def finalize_report(state: ResearchGraphState):
        """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
        # Save full final report
        content = state["content"]
        if content.startswith("## Insights"):
            content = content.strip("## Insights")
        if "## Sources" in content:
            try:
                content, sources = content.split("\n## Sources\n")
            except:
                sources = None
        else:
            sources = None

        final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
        if sources is not None:
            final_report += "\n\n## Sources\n" + sources
        with open("final_report.md", "w", encoding="utf-8") as f:
            f.write(final_report)
        return {"final_report": final_report}

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
    # Add nodes and edges 
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", build_interview_graph(model, search_engine))
    builder.add_node("write_report",write_report)
    builder.add_node("write_introduction",write_introduction)
    builder.add_node("write_conclusion",write_conclusion)
    builder.add_node("finalize_report",finalize_report)

    # Logic
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    # Compile
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph(xray=1).draw_mermaid_png())
    return graph