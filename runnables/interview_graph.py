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

from models.interview import InterviewState



class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")



def build_interview_graph(model, search_engine):
    def generate_question(state: InterviewState):
        """ Node to generate a question """

        #Get state
        analyst = state["analyst"]
        messages = state["messages"]

        # Generate question
        markdown_prompt = Path("prompts/questions.md").read_text(encoding="utf-8")

        analysts_prompt = PromptTemplate.from_template(markdown_prompt)

        
        system_message = analysts_prompt.format(goals=analyst.persona)
        question = model.invoke([SystemMessage(content=system_message)]+messages)
        return {"messages": [question]}

    def search_web(state: InterviewState):
        """ Retrieve docs from web search """

        markdown_prompt = Path("prompts/search.md").read_text(encoding="utf-8")

        # Search query
        structured_llm = model.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([markdown_prompt] + state["messages"])

        # Search
        search_docs = search_engine.invoke(search_query.search_query)
        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )
        return {"context": [formatted_search_docs]}

    def search_wikipedia(state: InterviewState):
        """ Retrieve docs from wikipedia """

        search_prompt = Path("prompts/search.md").read_text(encoding="utf-8")


        # Search query
        structured_llm = model.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_prompt] + state["messages"])

        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )

        return {"context": [formatted_search_docs]}

    def answer_question(state: InterviewState):
        """ Node to answer a question """
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]
        answer_prompt = Path("prompts/answers.md").read_text(encoding="utf-8")
        formatted_answer_prompt = answer_prompt.format(goals=analyst.persona, context=context)
        answer = model.invoke([SystemMessage(content=formatted_answer_prompt)] + messages)

        answer.name = "expert"

        return {"messages": [answer]}

    def save_interview(state: InterviewState):
        """Save interviews"""

        messages = state["messages"]

        interview = get_buffer_string(messages)

        return {"interview": interview}

    def route_messages(state: InterviewState, name: str = "expert"):
        """ Route messages to the appropriate node """
        messages = state["messages"]
        max_num_turns = state.get("max_num_turns", 2)

        num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

        # End if expert has answered more than the max turns
        if num_responses >= max_num_turns:
            return 'save_interview'

        # This router is run after each question - answer pair 
        # Get the last question asked to check if it signals the end of discussion
        last_question = messages[-2]

        if "Thank you so much for your help" in last_question.content:
            return 'save_interview'
        return "ask_question"

    def write_section(state: InterviewState):

        """ Node to answer a question """

        # Get state
        interview = state["interview"]
        context = state["context"]
        analyst = state["analyst"]
    
        answer_prompt = Path("prompts/write_section.md").read_text(encoding="utf-8")
        formatted_answer_prompt = answer_prompt.format(focus=analyst.description)
        # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
        section = model.invoke([SystemMessage(content=formatted_answer_prompt)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                    
        # Append it to state
        return {"sections": [section.content]}
    #Add nodes and edges
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("write_section", write_section)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("answer_question", answer_question)

    #Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    memory = MemorySaver()
    interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")
    display(Image(interview_graph.get_graph().draw_mermaid_png()))
    return interview_graph
