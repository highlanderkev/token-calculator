export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  encoding: string;
  contextWindow: number;
  inputPricePerMToken: number;
  outputPricePerMToken: number;
}

export const MODELS: ModelInfo[] = [
  // OpenAI
  {
    id: "gpt-4o",
    name: "GPT-4o",
    provider: "OpenAI",
    encoding: "cl100k_base",
    contextWindow: 128000,
    inputPricePerMToken: 2.5,
    outputPricePerMToken: 10.0,
  },
  {
    id: "gpt-4o-mini",
    name: "GPT-4o mini",
    provider: "OpenAI",
    encoding: "cl100k_base",
    contextWindow: 128000,
    inputPricePerMToken: 0.15,
    outputPricePerMToken: 0.6,
  },
  {
    id: "gpt-4-turbo",
    name: "GPT-4 Turbo",
    provider: "OpenAI",
    encoding: "cl100k_base",
    contextWindow: 128000,
    inputPricePerMToken: 10.0,
    outputPricePerMToken: 30.0,
  },
  {
    id: "gpt-3.5-turbo",
    name: "GPT-3.5 Turbo",
    provider: "OpenAI",
    encoding: "cl100k_base",
    contextWindow: 16385,
    inputPricePerMToken: 0.5,
    outputPricePerMToken: 1.5,
  },
  {
    id: "o1",
    name: "o1",
    provider: "OpenAI",
    encoding: "cl100k_base",
    contextWindow: 200000,
    inputPricePerMToken: 15.0,
    outputPricePerMToken: 60.0,
  },
  {
    id: "o1-mini",
    name: "o1-mini",
    provider: "OpenAI",
    encoding: "cl100k_base",
    contextWindow: 128000,
    inputPricePerMToken: 3.0,
    outputPricePerMToken: 12.0,
  },
  // Anthropic Claude
  {
    id: "claude-opus-4-5",
    name: "Claude Opus 4.5",
    provider: "Anthropic",
    encoding: "cl100k_base",
    contextWindow: 200000,
    inputPricePerMToken: 15.0,
    outputPricePerMToken: 75.0,
  },
  {
    id: "claude-sonnet-4-5",
    name: "Claude Sonnet 4.5",
    provider: "Anthropic",
    encoding: "cl100k_base",
    contextWindow: 200000,
    inputPricePerMToken: 3.0,
    outputPricePerMToken: 15.0,
  },
  {
    id: "claude-3-5-sonnet-20241022",
    name: "Claude 3.5 Sonnet",
    provider: "Anthropic",
    encoding: "cl100k_base",
    contextWindow: 200000,
    inputPricePerMToken: 3.0,
    outputPricePerMToken: 15.0,
  },
  {
    id: "claude-3-5-haiku-20241022",
    name: "Claude 3.5 Haiku",
    provider: "Anthropic",
    encoding: "cl100k_base",
    contextWindow: 200000,
    inputPricePerMToken: 0.8,
    outputPricePerMToken: 4.0,
  },
  // Google
  {
    id: "gemini-2.0-flash",
    name: "Gemini 2.0 Flash",
    provider: "Google",
    encoding: "cl100k_base",
    contextWindow: 1048576,
    inputPricePerMToken: 0.1,
    outputPricePerMToken: 0.4,
  },
  {
    id: "gemini-1.5-pro",
    name: "Gemini 1.5 Pro",
    provider: "Google",
    encoding: "cl100k_base",
    contextWindow: 2000000,
    inputPricePerMToken: 1.25,
    outputPricePerMToken: 5.0,
  },
  // Meta
  {
    id: "llama-3.1-405b",
    name: "Llama 3.1 405B",
    provider: "Meta (via Groq)",
    encoding: "cl100k_base",
    contextWindow: 131072,
    inputPricePerMToken: 0.59,
    outputPricePerMToken: 0.79,
  },
];

export function getModelById(id: string): ModelInfo | undefined {
  return MODELS.find((m) => m.id === id);
}

export function calculateCost(
  tokens: number,
  pricePerMToken: number
): number {
  return (tokens / 1_000_000) * pricePerMToken;
}
