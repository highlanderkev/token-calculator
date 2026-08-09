# token-calculator

Simple LLM/Agentic Token Calculator — a Python package that counts tokens and
estimates costs for popular LLMs, designed to be used as a **plugin** in
agentic AI systems (OpenAI Assistants, LangChain, CrewAI, AutoGen, …).

---

## Installation

```bash
pip install token-calculator          # tiktoken only (OpenAI models)
pip install "token-calculator[all]"   # + transformers + langchain-core + streamlit
pip install "token-calculator[ui]"    # + Streamlit UI
```

## Quick start

```python
from token_calculator import count_tokens, estimate_cost

n = count_tokens("Hello, world!", model="gpt-4o")
print(n)                              # e.g. 4

cost = estimate_cost(n, model="gpt-4o", direction="input")
print(f"${cost:.6f}")                 # e.g. $0.000020
```

## Supported models

```python
from token_calculator import list_models
print(list_models())
# ['claude-3-5-sonnet', 'claude-3-haiku', 'claude-3-opus', 'claude-3-sonnet',
#  'gemini-1.5-flash', 'gemini-1.5-pro', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo',
#  'gpt-4o', 'gpt-4o-mini', 'llama-3-70b', 'llama-3-8b', 'mistral-7b']
```

---

## Plugin integrations

### 1. OpenAI Assistants API / Chat Completions

```python
from openai import OpenAI
from token_calculator import OPENAI_TOOL_SPEC

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "How many tokens is 'Hello world'?"}],
    tools=[OPENAI_TOOL_SPEC],
    tool_choice="auto",
)
```

When the model calls the tool, invoke it and return the result:

```python
import json
from token_calculator import token_calculator_callable

tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
result = token_calculator_callable(**args)
```

### 2. LangChain agent

```python
# pip install "token-calculator[langchain]"
from token_calculator import TokenCalculatorTool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
tools = [TokenCalculatorTool()]

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)
agent.run("How many tokens does 'The quick brown fox' use on GPT-4o?")
```

### 3. CrewAI / AutoGen / plain callable

```python
from token_calculator import token_calculator_callable

# Any framework that accepts a Python callable works:
result = token_calculator_callable(
    text="The quick brown fox jumps over the lazy dog.",
    model="gpt-4o-mini",
    direction="output",
)
print(result)
# {"model": "gpt-4o-mini", "token_count": 10, "direction": "output",
#  "estimated_cost_usd": 6e-06}
```

### 4. Streamlit UI

```bash
pip install -e ".[ui]"
streamlit run /absolute/path/to/token-calculator/streamlit_app.py
```

The UI provides a text area, model selector, direction toggle, and live token
and cost summary for quick testing and evaluation.

---

## Development

```bash
pip install -e ".[dev]"
pytest --cov=token_calculator
```

## License

MIT
