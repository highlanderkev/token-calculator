import { describe, it, expect } from "vitest";
import { MODELS, calculateCost, getModelById } from "../models";

describe("MODELS", () => {
  it("has at least one model", () => {
    expect(MODELS.length).toBeGreaterThan(0);
  });

  it("each model has required fields", () => {
    for (const m of MODELS) {
      expect(m.id).toBeTruthy();
      expect(m.name).toBeTruthy();
      expect(m.provider).toBeTruthy();
      expect(m.contextWindow).toBeGreaterThan(0);
      expect(m.inputPricePerMToken).toBeGreaterThanOrEqual(0);
      expect(m.outputPricePerMToken).toBeGreaterThanOrEqual(0);
    }
  });

  it("model IDs are unique", () => {
    const ids = MODELS.map((m) => m.id);
    const unique = new Set(ids);
    expect(unique.size).toBe(ids.length);
  });
});

describe("getModelById", () => {
  it("returns the correct model", () => {
    const m = getModelById("gpt-4o");
    expect(m).toBeDefined();
    expect(m!.name).toBe("GPT-4o");
  });

  it("returns undefined for unknown model", () => {
    expect(getModelById("nonexistent")).toBeUndefined();
  });
});

describe("calculateCost", () => {
  it("returns 0 for 0 tokens", () => {
    expect(calculateCost(0, 10)).toBe(0);
  });

  it("correctly calculates cost for 1M tokens", () => {
    expect(calculateCost(1_000_000, 10)).toBe(10);
  });

  it("correctly calculates cost for 500k tokens", () => {
    expect(calculateCost(500_000, 2)).toBe(1);
  });
});
