import uuid
from typing import Annotated, TypedDict

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import WebBaseLoader

from settings import BaseModelSettings
from helpers import search_serpapi


@tool
def save_content(content: str, file_name: str) -> str:
    """Save the content to a Markdown file.

    Args:
        content: content to save
        file_name: name of the file to save. It must include the Markdown extension.
    """
    try:
        with open(f"./articles/{file_name}", "x") as f:
            f.write(content)
    except FileExistsError:
        return f"File {file_name} already exists!"
    return f"File {file_name} saved !"

@tool
def search_web(query: str) -> str:
    """Search the web for current information.

    Args:
        query: term to search for
    """
    res = search_serpapi(query, 1)
    return res[0]

@tool
def fetch_web_page(url: str) -> str:
    """Fetch and extract the readable text from a webpage.

    Args:
        url: url to fetch content
    """
    loader = WebBaseLoader(
        web_paths=(url,)
    )

    documents = loader.load()
    return "\n\n".join(
        document.page_content
        for document in documents
    )


tools_registry: dict = {
    "search_web": search_web,
    "fetch_web_page": fetch_web_page,
    "save_content": save_content
}

MAX_ITERATIONS = 20

class State(TypedDict):
    messages: Annotated[list, add_messages]
    # input
    user_topic: str
    # output
    article: str
    research: str

class Input(TypedDict):
    user_topic: str

class Output(TypedDict):
    research: str
    article: str

class MultiNodeGraphPoweredChat:
    """A langgraph powered chat that implement multi node steps to research a topic and produce a well-structured article"""
    def __init__(self):
        self.settings = BaseModelSettings()
        self.content_generator_model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature="0.1",
        ).bind_tools([search_web, fetch_web_page])
        self.content_saver_model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature="0.8",
        ).bind_tools([save_content])
        self.model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature="0.7",
        )
        # , then transform the findings into a well-structured, article-style piece of content
        self.content_generator_prompt = SystemMessage(content="""
        # Goal:
        - Research the user's topic using reliable web sources.
        
        # Available Tools
        - **search_web**: Search the web for relevant, current, and reliable information.
        - **fetch_web_page**: Fetch and extract the readable text from a webpage.
        
        # Instructions
        - Search for the user's topic using both **search_web**.
        - Retrieve web pages content using both **fetch_web_page**.
        - Use multiple searches when necessary to obtain sufficient context and coverage.
        - Prioritize authoritative, relevant, and trustworthy sources.
        - When information differs between sources, critically evaluate the discrepancy and prefer the most reliable source.
        - Do not fabricate facts or fill gaps with assumptions.
        - Focus the research on information that is directly relevant to the user's topic.
        
        # Input
        - The topic provided by the user.
        
        # Output
        Content must be returned in Plain text
        
        Do not return:
        - Markdown syntax
        - HTML
        - JSON
        - XML
        - Source/tool metadata
        - Search-result lists
        - Explanations of the research process
        """)
        self.content_formatter_prompt = SystemMessage(content="""
        # Goal:
        - You are a Content Structure and Formatting Agent.
        - Your responsibility is to transform a research package into a clear, engaging, publication-quality article using Markdown.
        - You receive research produced by another AI model. Your job is to organize, synthesize, and present that research, not to perform independent research.
        
        # Objective
        Transform the provided research into a well-structured article that is:
        - Accurate
        - Coherent
        - Engaging
        - Easy to read
        - Logically organized
        - Properly formatted in Markdown
        
        # Instructions
        - Use the Research package as the Source of Truth.
        - Do not invent facts, statistics, examples, quotations, or sources.
        - Preserve important qualifications and uncertainties from the research.
        - Create a logical article structure appropriate for the topic.
        - Write in a polished magazine/article style.
        
        # Input
        - User Topic.
        - Research Package produced by the Research Agent
        
        # Output
        Return only the final Markdown article.

        Do not include:
        - Commentary about your process
        - Research-agent instructions
        - Internal reasoning
        - Statements such as "Based on the research..."
        - Search-result metadata
        - Unnecessary preambles
        - JSON
        - XML

        The output must be ready to render directly as a Markdown article.
        """)
        self.content_saver_prompt = SystemMessage(content="""
        # Goal:
        - You are a Markdown File Writer Agent.
        - Your responsibility is to take a finalized Markdown article, determine an appropriate filename, and save the article to the local filesystem using the **save_content** tool.
        - You are the final stage of the content-generation pipeline.
                
        # Objective
        Given a finalized Markdown article:

        - Determine an appropriate filename.
        - Generate a filesystem-safe .md filename.
        - Save the complete article using the **save_content** tool.
        - Confirm that the file was successfully saved.

        # Available Tools
        - **save_content**: Save the article to the local filesystem.
        
        The tool should receive:
        - content: The complete Markdown article.
        - file_name: The filename to create.
            
        Always use this tool to save the article.
                
        # Filename Instructions
        Generate the filename from the article's topic or title.
        
        The filename must:
        - End with .md
        - Use lowercase characters
        - Use hyphens (-) between words
        - Contain only filesystem-safe characters
        - Avoid special characters
        - Avoid unnecessary words
        - Be descriptive enough to identify the article
        - Prefer the article's Markdown # title when available
        
        Examples
        Artificial Intelligence in Healthcare
        → artificial-intelligence-in-healthcare.md
        
        The History of Deep Learning
        → history-of-deep-learning.md
        
        What Is Retrieval-Augmented Generation?
        → retrieval-augmented-generation.md
        
        10 Benefits of Cloud Computing
        → benefits-of-cloud-computing.md        
        
        # Content Preservation
        The Markdown article must be saved exactly as provided.
        
        Do not:
        - Rewrite the article
        - Summarize the article
        - Remove sections
        - Change Markdown formatting
        - Add commentary
        - Add metadata
        - Add a new title
        - Modify links
        - Correct grammar
        - Change citations
        
        # Input
        - The finalized Markdown article produced by the Content Agent
        
        # Output
        After successfully saving the file, return a concise confirmation containing:

        - The generated filename.
        - The fact that the article was successfully saved.
        
        Example:

        Article successfully saved.
        
        File: artificial-intelligence-in-healthcare.md
        
        Do not return the full article in the response.
        """)
        self.builder = StateGraph(state_schema=State, input_schema=Input)
        self.graph = self.__build()

    @staticmethod
    def __execute_tools(tool_calls) -> list:
        output = []
        for tool_call in tool_calls:
            selected_tool = tools_registry[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])
            output.append(
                ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call["id"]
                )
            )
        return output

    def __generate_content(self, state: State):
        user_message = HumanMessage(content=state["user_topic"])
        messages = [
            self.content_generator_prompt,
            *state["messages"],
            user_message
        ]
        # Tools call execution support
        for _ in range(MAX_ITERATIONS):
            answer_msg = self.content_generator_model.invoke(messages)
            messages.append(answer_msg)
            if not answer_msg.tool_calls:
                break

            messages.extend(MultiNodeGraphPoweredChat.__execute_tools(answer_msg.tool_calls))
        else:
            raise RuntimeError("Chat exceeded maximum tool interactions")

        return {
            "research": answer_msg.content,
            "messages": [user_message, messages[-1]],
        }

    def __format_content(self, state: State):
        messages = [
            self.content_formatter_prompt,
            # Contains user's topic, tool_calls, search responses and research content from prev steps
            *state["messages"]
        ]
        answer_msg = self.model.invoke(messages)

        return {
            "article": answer_msg.content,
            "messages": answer_msg,
        }

    def __save_content(self, state: State):
        # Contains user's topic, tool_calls, search responses, research content and article content from prev steps
        # *state["messages"]
        messages = [
            self.content_saver_prompt,
            # Contains article content from prev steps
            state["messages"][-1]
        ]
        # Tools call execution support
        for _ in range(MAX_ITERATIONS):
            answer_msg = self.content_saver_model.invoke(messages)
            messages.append(answer_msg)
            if not answer_msg.tool_calls:
                break

            messages.extend(MultiNodeGraphPoweredChat.__execute_tools(answer_msg.tool_calls))
        else:
            raise RuntimeError("Chat exceeded maximum tool interactions")

        return {
            "messages": messages[-1],
        }

    def __build(self):
        self.builder.add_node("content_generator", self.__generate_content)
        self.builder.add_node("content_formatter", self.__format_content)
        self.builder.add_node("content_saver", self.__save_content)

        self.builder.add_edge(START, "content_generator")
        self.builder.add_edge("content_generator", "content_formatter")
        self.builder.add_edge("content_formatter", "content_saver")
        self.builder.add_edge("content_saver", END)

        return self.builder.compile()

    def draw_graph(self):
        self.graph.get_graph().draw_mermaid_png(output_file_path="multi_node_graph.png")

    def initialize(self, config):
        print("Welcome to ChatPrompt!")
        print("Start typing ('c' for exit) >> ")
        while True:
            question = input()
            if question == "c":
                break
            elif question.strip() == "":
                continue
            state = self.graph.invoke(input={
                "user_topic": question
            }, config=config)
            print("ARTICLE:")
            print(state["article"])


class MemorySaverSingleNodeGraphPoweredChat:
    """A langgraph powered chat with memory that implement a single node to perform as helpful assistant"""
    def __init__(self):
        self.settings = BaseModelSettings()
        self.model = init_chat_model(
            model=self.settings.model_name,
            model_provider=self.settings.provider,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        self.builder = StateGraph(MessagesState)
        self.graph = self.__build()

    def __chatbot(self):
        def func(state: MessagesState):
            #print(state["messages"])
            answer_msg = self.model.invoke(state["messages"])

            return {
                "messages": [answer_msg],
            }

        return func

    def __build(self):
        self.builder.add_node("chatbot", self.__chatbot())

        self.builder.add_edge(START, "chatbot")
        self.builder.add_edge("chatbot", END)

        return self.builder.compile(checkpointer=InMemorySaver())

    def draw_graph(self):
        self.graph.get_graph().draw_mermaid_png(output_file_path="single_node_graph.png")

    def initialize(self, config):
        print("Welcome to ChatPrompt!")
        print("Start typing ('c' for exit) >> ")
        while True:
            question = input()
            if question == "c":
                break
            elif question.strip() == "":
                continue
            state = self.graph.invoke(input={
                "messages": [HumanMessage(content=question)]
            }, config=config)
            print(state["messages"][-1].content)


#main_chat = MemorySaverSingleNodeGraphPoweredChat()
main_chat = MultiNodeGraphPoweredChat()

if __name__ == '__main__':
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    main_chat.initialize(thread_config)
    #main_chat.draw_graph()