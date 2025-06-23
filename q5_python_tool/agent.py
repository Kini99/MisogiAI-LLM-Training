import sys
import io
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load TinyLlama (CPU, small model)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
def load_model():
    print("Loading TinyLlama model (this may take a while if not cached)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    return tokenizer, model

tokenizer, model = load_model()

# Tool: python.exec
def python_exec(code):
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        result = eval(code)
        if result is not None:
            print(result)
    except Exception:
        try:
            exec(code)
        except Exception as e:
            print(f"Error: {e}")
    sys.stdout = old_stdout
    return mystdout.getvalue().strip()

# Tool: noop
def noop():
    return ""

# Rule-based agent: decide which tool to use and what code to run
def agent_decide(query):
    # Counting queries
    if "how many" in query.lower() and "'" in query:
        # e.g., How many 'r' in 'strawberry'?
        import re
        m = re.search(r"how many '(.+)' in '(.+)'", query.lower())
        if m:
            char = m.group(1)
            string = m.group(2)
            code = f"len([c for c in '{string}' if c == '{char}'])"
            return "python.exec", code
    if "count" in query.lower() and "in" in query.lower():
        # e.g., Count the number of 'a' in 'banana'
        import re
        m = re.search(r"count (?:the number of )?'(.+)' in '(.+)'", query.lower())
        if m:
            char = m.group(1)
            string = m.group(2)
            code = f"len([c for c in '{string}' if c == '{char}'])"
            return "python.exec", code
    if "count" in query.lower() and "substring" in query.lower():
        # e.g., Count substring 'ana' in 'banana'
        import re
        m = re.search(r"count substring '(.+)' in '(.+)'", query.lower())
        if m:
            substr = m.group(1)
            string = m.group(2)
            code = f"'{string}'.count('{substr}')"
            return "python.exec", code
    # Arithmetic queries
    if any(op in query for op in ["+", "-", "*", "/", "^"]):
        # e.g., What is 3 + 5 * 2?
        import re
        m = re.search(r"what is (.+)[?]", query.lower())
        if m:
            expr = m.group(1).replace("^", "**")
            code = expr
            return "python.exec", code
    if "sum of" in query.lower():
        # e.g., What is the sum of 10 and 15?
        import re
        m = re.search(r"sum of (\d+) and (\d+)", query.lower())
        if m:
            a, b = m.group(1), m.group(2)
            code = f"{a} + {b}"
            return "python.exec", code
    # Otherwise, noop
    return "noop", None

# Simulate LLM response (using TinyLlama for final answer)
def llm_respond(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_new_tokens=32, do_sample=False)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt from the response
    return response[len(prompt):].strip()

# Example transcripts
def run_examples():
    examples = [
        # Counting queries
        ("How many 'r' in 'strawberry'?"),
        ("Count the number of 'a' in 'banana'"),
        ("Count substring 'ana' in 'banana'"),
        # Arithmetic queries
        ("What is 3 + 5 * 2?"),
        ("What is the sum of 10 and 15?"),
    ]
    for i, user_query in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"User: {user_query}")
        tool, code = agent_decide(user_query)
        if tool == "python.exec":
            print(f"Tool call: python.exec('{code}')")
            tool_output = python_exec(code)
            print(f"Tool output: {tool_output}")
            # LLM final response
            if "count" in user_query.lower() or "how many" in user_query.lower():
                final_prompt = f"User asked: {user_query}\nThe answer is {tool_output}. Respond in a helpful sentence."
            else:
                final_prompt = f"User asked: {user_query}\nThe answer is {tool_output}. Respond in a helpful sentence."
            llm_response = llm_respond(final_prompt)
            print(f"LLM: {llm_response}")
        else:
            print("Tool call: noop()")
            print("Tool output: (nothing)")
            llm_response = llm_respond(f"User asked: {user_query}\nRespond in a helpful sentence.")
            print(f"LLM: {llm_response}")
        print("-" * 40)

if __name__ == "__main__":
    run_examples() 