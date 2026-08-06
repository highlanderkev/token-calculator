# Token Calculator

A simple LLM/Agentic Token Calculator built with React + TypeScript + Vite.

## Features

- **Simple Mode** – Paste input and output text to count tokens and estimate API cost
- **Agentic Loop Mode** – Estimate costs for multi-turn agentic workflows, accounting for growing context across turns
- Supports models from OpenAI, Anthropic, Google, and Meta
- Accurate token counting via [tiktoken](https://github.com/openai/tiktoken)
- Context window usage indicator

## Getting Started

```bash
npm install
npm run dev
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm test` | Run tests |
| `npm run lint` | Run linter |

## How it works

Token counts use OpenAI's `cl100k_base` encoding (tiktoken), which is also a close approximation for Anthropic and Google models.

**Agentic cost model:** Each turn includes the accumulated context (system prompt + all prior messages + responses), which is how most LLM APIs are billed. The estimated cost grows super-linearly with the number of turns.
