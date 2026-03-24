import asyncio
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()  # load environment variables from .env
logging.basicConfig(level=logging.DEBUG)

# Groq model constant
GROQ_MODEL = "llama-3.3-70b-versatile"


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self._groq: OpenAI | None = None

    @property
    def groq(self) -> OpenAI:
        """Lazy-initialize Groq client when needed"""
        if self._groq is None:
            self._groq = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
        return self._groq

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        if is_python:
            path = Path(server_script_path).resolve()
            server_params = StdioServerParameters(
                command="uv",
                args=["--directory", str(path.parent), "run", path.name],
                env=None,
            )
        else:
            server_params = StdioServerParameters(
                command="node", args=[server_script_path], env=None
            )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Groq (OpenAI-compatible) and available tools"""
        import json

        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
        # Format tools for OpenAI-compatible API
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]

        # Initial API call with tools
        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        # Process response and handle tool calls
        final_text = []
        message = response.choices[0].message

        # Handle text response
        if message.content:
            final_text.append(message.content)

        # Handle tool calls if they exist
        while message.tool_calls:
            # Add assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            # Execute tool calls and collect results
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                # Debug log the raw tool call for inspection
                logging.debug("Tool call requested: %s %s", tool_name, tool_args)

                # Small validation / normalization for get_alerts: ensure 2-letter state code
                if tool_name == "get_alerts" and isinstance(tool_args, dict):
                    state_val = tool_args.get("state")
                    if isinstance(state_val, str):
                        state_norm = state_val.strip().upper()
                        # If user provided full state name, take first two letters as fallback
                        if len(state_norm) != 2:
                            state_norm = state_norm[:2]
                        tool_args["state"] = state_norm

                # Convert string numbers to actual numbers based on schema
                tool_schema = next(
                    (
                        t["function"]["parameters"]
                        for t in available_tools
                        if t["function"]["name"] == tool_name
                    ),
                    None,
                )
                if tool_schema and "properties" in tool_schema:
                    for key, value in tool_args.items():
                        if key in tool_schema["properties"]:
                            prop_type = tool_schema["properties"][key].get("type")
                            if prop_type == "number" and isinstance(value, str):
                                try:
                                    tool_args[key] = float(value)
                                except (ValueError, TypeError):
                                    pass
                            elif prop_type == "integer" and isinstance(value, str):
                                try:
                                    tool_args[key] = int(value)
                                except (ValueError, TypeError):
                                    pass

                try:
                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(
                        f"[Calling tool {tool_name} with args {tool_args}]"
                    )

                    # Extract tool result content
                    tool_result_text = ""
                    if hasattr(result, "content") and result.content:
                        if isinstance(result.content, list):
                            tool_result_text = (
                                result.content[0].text
                                if hasattr(result.content[0], "text")
                                else str(result.content[0])
                            )
                        else:
                            tool_result_text = (
                                result.content.text
                                if hasattr(result.content, "text")
                                else str(result.content)
                            )

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result_text,
                        }
                    )

                except Exception as e:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error executing tool: {str(e)}",
                        }
                    )

            # Get next response from API
            response = self.groq.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=1000,
                messages=messages,
                tools=available_tools,
            )

            message = response.choices[0].message
            if message.content:
                final_text.append(message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])

        # Check if we have a valid API key to continue
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(
                "\nNo GROQ_API_KEY found. To query these tools with Groq, set your API key:"
            )
            print("  export GROQ_API_KEY=your-api-key-here")
            return

        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
