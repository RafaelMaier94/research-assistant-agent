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



### Tavily web search
tavily_search = TavilySearchResults(max_results=3)

### LLM
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"
    
class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions

class InterviewState(MessagesState):
    max_num_turns: int # Max number of turns of conversation
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")
def build_interview_graph():
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
        search_docs = tavily_search.invoke(search_query.search_query)
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
        answer_prompt = Path("prompts/answer.md").read_text(encoding="utf-8")
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

def build_analysts_graph():
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

def invoke_graph(graph, topic):
    max_analysts = 3
    
    thread = {"configurable": {"thread_id": 1}}
    for event in graph.invoke({"topic": topic, "max_analysts": max_analysts}, thread, stream_mode="values"):
        analysts = event.get("analysts", "")
        return analysts