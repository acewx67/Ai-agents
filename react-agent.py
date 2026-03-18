import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an AI agent that uses reasoning and tools.

You MUST follow this format:

Thought: what you think
Action: tool name OR "finish"
Action Input: input to the tool

Available tools:
- calculator: evaluates math expressions

When you receive a tool result, continue reasoning.

When done:
Action: finish
Action Input: final answer

Example:

User: What is (2 + 3) * 4?

Thought: I need to calculate (2+3)*4
Action: calculator
Action Input: (2+3)*4

Observation: 20

Thought: I now know the answer
Action: finish
Action Input: The result is 20
"""

# ---- TOOL ----
def calculator(expression: str):
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {str(e)}"

TOOLS = {
    "calculator": calculator
}


def parse_response(text):
    lines = text.strip().split("\n")

    thought = None
    action = None
    action_input = None

    for line in lines:
        if line.startswith("Thought:"):
            thought = line.replace("Thought:", "").strip()
        elif line.startswith("Action:"):
            action = line.replace("Action:", "").strip()
        elif line.startswith("Action Input:"):
            action_input = line.replace("Action Input:", "").strip()

    return thought, action, action_input


def run_agent(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    while True:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0
        )

        content = response.choices[0].message.content
        print("\nLLM OUTPUT:\n", content)

        thought, action, action_input = parse_response(content)

        if not action:
            return "Failed to parse LLM response"

        if action == "finish":
            return action_input

        if action not in TOOLS:
            return f"Unknown tool: {action}"

        # ---- EXECUTE TOOL ----
        result = TOOLS[action](action_input)

        print("TOOL RESULT:", result)

        # ---- FEED BACK ----
        messages.append({
            "role": "assistant",
            "content": content
        })

        messages.append({
            "role": "user",
            "content": f"Observation: {result}"
        })

# ---- RUN ----
if __name__ == "__main__":
    while True:
        query = input(">> ")
        answer = run_agent(query)
        print("Final:", answer)