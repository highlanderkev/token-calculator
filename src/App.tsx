import { useState } from "react";
import { MODELS, calculateCost, type ModelInfo } from "./models";
import { countTokens, estimateAgenticCost } from "./tokenizer";
import "./App.css";

type Tab = "simple" | "agentic";

function formatCost(cost: number): string {
  if (cost === 0) return "$0.00";
  if (cost < 0.000001) return `$${cost.toExponential(2)}`;
  if (cost < 0.01) return `$${cost.toFixed(6)}`;
  return `$${cost.toFixed(4)}`;
}

function formatNumber(n: number): string {
  return n.toLocaleString();
}

function groupedModels(): Record<string, ModelInfo[]> {
  const groups: Record<string, ModelInfo[]> = {};
  for (const m of MODELS) {
    if (!groups[m.provider]) groups[m.provider] = [];
    groups[m.provider].push(m);
  }
  return groups;
}

function ModelSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (id: string) => void;
}) {
  const groups = groupedModels();
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="model-select"
    >
      {Object.entries(groups).map(([provider, models]) => (
        <optgroup key={provider} label={provider}>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

function SimpleCalculator() {
  const [modelId, setModelId] = useState(MODELS[0].id);
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");

  const model = MODELS.find((m) => m.id === modelId)!;
  const inputTokens = countTokens(inputText, model.encoding);
  const outputTokens = countTokens(outputText, model.encoding);
  const totalTokens = inputTokens + outputTokens;
  const inputCost = calculateCost(inputTokens, model.inputPricePerMToken);
  const outputCost = calculateCost(outputTokens, model.outputPricePerMToken);
  const totalCost = inputCost + outputCost;

  const contextUsedPct = Math.min(
    100,
    Math.round((totalTokens / model.contextWindow) * 100)
  );

  return (
    <div className="calculator-panel">
      <div className="field-group">
        <label htmlFor="simple-model">Model</label>
        <ModelSelect value={modelId} onChange={setModelId} />
        <p className="hint">
          Context window: {formatNumber(model.contextWindow)} tokens &nbsp;·&nbsp;
          Input: ${model.inputPricePerMToken}/M &nbsp;·&nbsp; Output: $
          {model.outputPricePerMToken}/M
        </p>
      </div>

      <div className="two-col">
        <div className="field-group">
          <label htmlFor="input-text">Input / Prompt</label>
          <textarea
            id="input-text"
            rows={8}
            placeholder="Paste your prompt or system message here…"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          />
          <p className="token-count">
            <span className="badge">{formatNumber(inputTokens)} tokens</span>
            <span className="cost">{formatCost(inputCost)}</span>
          </p>
        </div>

        <div className="field-group">
          <label htmlFor="output-text">Output / Response</label>
          <textarea
            id="output-text"
            rows={8}
            placeholder="Paste the model response here, or type expected output length…"
            value={outputText}
            onChange={(e) => setOutputText(e.target.value)}
          />
          <p className="token-count">
            <span className="badge">{formatNumber(outputTokens)} tokens</span>
            <span className="cost">{formatCost(outputCost)}</span>
          </p>
        </div>
      </div>

      <div className="results-card">
        <div className="result-row">
          <span>Total tokens</span>
          <strong>{formatNumber(totalTokens)}</strong>
        </div>
        <div className="result-row">
          <span>Context used</span>
          <div className="progress-wrap">
            <div
              className="progress-bar"
              role="progressbar"
              aria-valuenow={contextUsedPct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`Context window usage: ${contextUsedPct}%`}
            >
              <div
                className="progress-fill"
                style={{
                  width: `${contextUsedPct}%`,
                  backgroundColor:
                    contextUsedPct > 90
                      ? "#ef4444"
                      : contextUsedPct > 70
                      ? "#f59e0b"
                      : "#22c55e",
                }}
              />
            </div>
            <span aria-hidden="true">{contextUsedPct}%</span>
          </div>
        </div>
        <div className="result-row total-cost">
          <span>Estimated cost</span>
          <strong className="cost-big">{formatCost(totalCost)}</strong>
        </div>
      </div>
    </div>
  );
}

function AgenticCalculator() {
  const [modelId, setModelId] = useState(MODELS[0].id);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [userMessage, setUserMessage] = useState("");
  const [outputTokensPerTurn, setOutputTokensPerTurn] = useState(500);
  const [numTurns, setNumTurns] = useState(10);

  const model = MODELS.find((m) => m.id === modelId)!;
  const systemTokens = countTokens(systemPrompt, model.encoding);
  const userTokens = countTokens(userMessage, model.encoding);

  const estimate = estimateAgenticCost(
    systemTokens,
    userTokens,
    outputTokensPerTurn,
    numTurns,
    model.inputPricePerMToken,
    model.outputPricePerMToken
  );

  return (
    <div className="calculator-panel">
      <div className="field-group">
        <label>Model</label>
        <ModelSelect value={modelId} onChange={setModelId} />
        <p className="hint">
          Context window: {formatNumber(model.contextWindow)} tokens &nbsp;·&nbsp;
          Input: ${model.inputPricePerMToken}/M &nbsp;·&nbsp; Output: $
          {model.outputPricePerMToken}/M
        </p>
      </div>

      <div className="two-col">
        <div className="field-group">
          <label>System Prompt</label>
          <textarea
            rows={6}
            placeholder="Paste your system prompt here…"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
          />
          <p className="token-count">
            <span className="badge">{formatNumber(systemTokens)} tokens</span>
          </p>
        </div>

        <div className="field-group">
          <label>Average User Message per Turn</label>
          <textarea
            rows={6}
            placeholder="Paste a representative user message here…"
            value={userMessage}
            onChange={(e) => setUserMessage(e.target.value)}
          />
          <p className="token-count">
            <span className="badge">{formatNumber(userTokens)} tokens</span>
          </p>
        </div>
      </div>

      <div className="two-col">
        <div className="field-group">
          <label>
            Output tokens per turn
            <span className="value-badge">{formatNumber(outputTokensPerTurn)}</span>
          </label>
          <input
            type="range"
            min={50}
            max={8000}
            step={50}
            value={outputTokensPerTurn}
            onChange={(e) => setOutputTokensPerTurn(Number(e.target.value))}
          />
        </div>
        <div className="field-group">
          <label>
            Number of turns
            <span className="value-badge">{numTurns}</span>
          </label>
          <input
            type="range"
            min={1}
            max={100}
            step={1}
            value={numTurns}
            onChange={(e) => setNumTurns(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="results-card">
        <div className="result-row">
          <span>Total input tokens</span>
          <strong>{formatNumber(estimate.totalInputTokens)}</strong>
        </div>
        <div className="result-row">
          <span>Total output tokens</span>
          <strong>{formatNumber(estimate.totalOutputTokens)}</strong>
        </div>
        <div className="result-row">
          <span>Input cost</span>
          <strong>{formatCost(estimate.inputCost)}</strong>
        </div>
        <div className="result-row">
          <span>Output cost</span>
          <strong>{formatCost(estimate.outputCost)}</strong>
        </div>
        <div className="result-row total-cost">
          <span>Estimated total cost ({numTurns} turns)</span>
          <strong className="cost-big">{formatCost(estimate.totalCost)}</strong>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("simple");

  return (
    <div className="app">
      <header className="app-header">
        <h1>🪙 Token Calculator</h1>
        <p className="subtitle">
          Estimate LLM token counts and API costs — including agentic workflows
        </p>
      </header>

      <nav className="tabs">
        <button
          className={tab === "simple" ? "tab active" : "tab"}
          onClick={() => setTab("simple")}
        >
          Simple
        </button>
        <button
          className={tab === "agentic" ? "tab active" : "tab"}
          onClick={() => setTab("agentic")}
        >
          Agentic Loop
        </button>
      </nav>

      <main>
        {tab === "simple" ? <SimpleCalculator /> : <AgenticCalculator />}
      </main>

      <footer className="app-footer">
        <p>
          Prices are approximate and may vary. Token counts use{" "}
          <a
            href="https://github.com/openai/tiktoken"
            target="_blank"
            rel="noreferrer"
          >
            tiktoken
          </a>{" "}
          (cl100k_base).
        </p>
      </footer>
    </div>
  );
}
