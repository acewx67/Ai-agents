import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---- TOOL: Python calculator ----
def calculator(expression: str):
    try:
        # ⚠️ eval is dangerous in real systems (we'll fix later)
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# ---- TOOL REGISTRY ----
TOOLS = {
    "calculator": calculator
}


# ---- SYSTEM PROMPT ----
# SYSTEM_PROMPT = """
# You are an AI agent.

# If a math calculation is required, DO NOT solve it yourself.
# Instead, call the calculator tool.

# Respond ONLY in JSON format:

# {
#   "action": "tool_name" OR "final",
#   "input": "input string"
# }

# Examples:

# User: 2 + 2
# Response:
# {"action": "calculator", "input": "2+2"}

# User: hello
# Response:
# {"action": "final", "input": "Hello!"}
# """



# ---- AGENT LOOP ----
def run_agent(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # fast on Groq
            messages=messages,
            temperature=0
        )

        content = response.choices[0].message.content

        print("LLM:", content)

        try:
            data = json.loads(content)
        except:
            return "Invalid JSON from LLM"

        action = data["action"]
        tool_input = data["input"]

        if action == "final":
            return tool_input

        elif action in TOOLS:
            result = TOOLS[action](tool_input)

            # Feed result back to LLM
            messages.append({
                "role": "assistant",
                "content": content
            })

            messages.append({
                "role": "user",
                "content": f"Tool result: {result}"
            })

        else:
            return "Unknown tool"


# ---- RUN ----
if __name__ == "__main__":
    while True:
        query = input(">> ")
        answer = run_agent(query)
        print("Final:", answer)