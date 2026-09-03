from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command


class ToolMonitoringMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # 1. (Optional) Run logic BEFORE the tool is called
        print(f"Executing tool: {request.tool_call['name']}")
        print(f"Arguments: {request.tool_call['args']}")
        try:
            # 2. Execute the tool by invoking the next handler
            result = handler(request)
            # 3. Run logic AFTER the tool completes
            # Note: 'result' typically contains a ToolMessage with the content
            tool_output = result.content
            print(f"Tool finished executing. Output: {tool_output}")

            return result
        except Exception as e:
            print(f"Tool failed: {e}")
            raise