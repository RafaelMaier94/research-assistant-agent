from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from IPython.display import Markdown

from runnable.research_assistant import build_analysts_graph, build_interview_graph, invoke_graph




if __name__ == "__main__":
    topic = "The benefits of adopting LangGraph as an agent framework"
    analysts_graph = build_analysts_graph()
    analysts = invoke_graph(analysts_graph, topic)
    import pdb
    pdb.set_trace()
    messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
    thread = {"configurable": {"thread_id": "1"}}
    interview_graph = build_interview_graph()
    interview = interview_graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns": 2}, thread)
    print(Markdown(interview['sections'][0]))
