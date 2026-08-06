import { get_encoding, type TiktokenEncoding } from "tiktoken";

const encoderCache = new Map<string, ReturnType<typeof get_encoding>>();

function getEncoder(encodingName: string): ReturnType<typeof get_encoding> {
  if (!encoderCache.has(encodingName)) {
    encoderCache.set(
      encodingName,
      get_encoding(encodingName as TiktokenEncoding)
    );
  }
  return encoderCache.get(encodingName)!;
}

export function countTokens(text: string, encodingName: string): number {
  if (!text) return 0;
  try {
    const encoder = getEncoder(encodingName);
    return encoder.encode(text).length;
  } catch {
    // Fallback: rough approximation (~4 chars per token)
    return Math.ceil(text.length / 4);
  }
}

export function estimateAgenticCost(
  systemPromptTokens: number,
  userMessageTokens: number,
  outputTokensPerTurn: number,
  numTurns: number,
  inputPricePerMToken: number,
  outputPricePerMToken: number
): {
  totalInputTokens: number;
  totalOutputTokens: number;
  inputCost: number;
  outputCost: number;
  totalCost: number;
} {
  // In an agentic loop, the system prompt + growing context is sent each turn
  // Simplified model: each turn includes system prompt + accumulated messages
  let totalInputTokens = 0;
  let accumulatedTokens = systemPromptTokens;

  for (let i = 0; i < numTurns; i++) {
    accumulatedTokens += userMessageTokens;
    totalInputTokens += accumulatedTokens;
    accumulatedTokens += outputTokensPerTurn;
  }

  const totalOutputTokens = outputTokensPerTurn * numTurns;
  const inputCost = (totalInputTokens / 1_000_000) * inputPricePerMToken;
  const outputCost = (totalOutputTokens / 1_000_000) * outputPricePerMToken;

  return {
    totalInputTokens,
    totalOutputTokens,
    inputCost,
    outputCost,
    totalCost: inputCost + outputCost,
  };
}
