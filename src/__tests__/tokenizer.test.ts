import { describe, it, expect } from "vitest";
import { countTokens } from "../tokenizer";
import { estimateAgenticCost } from "../tokenizer";

describe("countTokens", () => {
  it("returns 0 for empty string", () => {
    expect(countTokens("", "cl100k_base")).toBe(0);
  });

  it("counts tokens for a simple sentence", () => {
    const tokens = countTokens("Hello, world!", "cl100k_base");
    expect(tokens).toBeGreaterThan(0);
    expect(tokens).toBeLessThan(10);
  });

  it("returns more tokens for longer text", () => {
    const short = countTokens("hi", "cl100k_base");
    const long = countTokens(
      "This is a much longer sentence with many more words in it.",
      "cl100k_base"
    );
    expect(long).toBeGreaterThan(short);
  });
});

describe("estimateAgenticCost", () => {
  it("calculates zero cost when all inputs are zero", () => {
    const result = estimateAgenticCost(0, 0, 0, 0, 1.0, 2.0);
    expect(result.totalInputTokens).toBe(0);
    expect(result.totalOutputTokens).toBe(0);
    expect(result.totalCost).toBe(0);
  });

  it("calculates cost for a single turn with no system prompt", () => {
    // 1 turn, 100 user tokens/turn, 50 output tokens/turn
    const result = estimateAgenticCost(0, 100, 50, 1, 1.0, 2.0);
    expect(result.totalInputTokens).toBe(100);
    expect(result.totalOutputTokens).toBe(50);
    // inputCost = 100/1e6 * 1 = 0.0001
    expect(result.inputCost).toBeCloseTo(0.0001, 7);
    // outputCost = 50/1e6 * 2 = 0.0001
    expect(result.outputCost).toBeCloseTo(0.0001, 7);
    expect(result.totalCost).toBeCloseTo(0.0002, 7);
  });

  it("accumulates context across multiple turns", () => {
    // 2 turns: system=0, user=10 tokens each, output=10 tokens/turn
    // Turn 1: accumulated=0+10=10, input+=10, accumulated=10+10=20
    // Turn 2: accumulated=20+10=30, input+=30, accumulated=30+10=40
    // totalInputTokens = 10 + 30 = 40
    const result = estimateAgenticCost(0, 10, 10, 2, 1.0, 1.0);
    expect(result.totalInputTokens).toBe(40);
    expect(result.totalOutputTokens).toBe(20);
  });

  it("includes system prompt tokens in every turn", () => {
    const result = estimateAgenticCost(100, 0, 0, 1, 0, 0);
    expect(result.totalInputTokens).toBe(100);
  });
});
