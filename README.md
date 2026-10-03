# llm-engineering

Three portfolio projects for an AI engineering interview prep. Built end-to-end against production patterns — retrieval quality, agent architecture, enterprise safety — with measurements, not vibes.

Each project lives in its own folder with its own README, numbers, and known limitations.

## Projects

### 1. Contract Intelligence RAG — [`project1-rag/`](./project1-rag/)

Retrieval-augmented generation over Berkshire Hathaway's 2023 shareholder letter. Hybrid search (dense + BM25), contextual retrieval, cross-encoder reranking, and a three-path abstention prompt that distinguishes answerable, unanswerable, and table-dependent questions.

**Headline numbers on a 50-question hand-reviewed golden set:**
- 42/42 answerable attempted
- 8/8 unanswerable correctly abstained
- 0 hallucinations, 0 over-refusals

### 2. Enterprise Ops Agent — [`project2-agent/`](./project2-agent/)

Tool-using agent over a mock CRM/ERP (customers, invoices, tickets, refunds). Built twice — hand-rolled loop against Groq, and on the Claude Agent SDK — to compare the abstraction against the thing it abstracts. Human-in-the-loop approval gate enforced in code rather than in the prompt. Isolated research subagent for deep-dive questions. 21-task eval harness.

**Headline numbers:**
- 20/21 eval pass rate (95%)
- Subagent experiment: main context 9.3K → 1.1K tokens on the same question
- Caught a silent HITL bypass in the SDK's `can_use_tool` path; switched to a `PreToolUse` hook

### 3. Guardrail & Eval Layer — `project3-guardrails/` *(in progress)*

Multi-tenant isolation enforced at the database layer (Postgres RLS), prompt-injection defence with a measured before/after, PII redaction at ingest, audit logging, and CI-run regression evals across Projects 1 and 2.

## Repo layout

```
llm-engineering/
├── project1-rag/          # RAG system + eval harness
├── project2-agent/        # Agent + subagent + eval harness
├── project3-guardrails/   # (coming soon)
├── LOG.md                 # daily build log — predictions, numbers, surprises
├── .env.example
└── README.md
```

## About the build log

[`LOG.md`](./LOG.md) is the daily record of what I changed, what the numbers did, and what surprised me. The format is three parts per entry — change, numbers, surprise — because the gap between prediction and reality is where I actually learned something. If you're interviewing me and want to see how I actually work, the LOG is more honest than the per-project READMEs.

## Stack

Python 3.12, `uv` for packaging, Supabase Postgres with `pgvector`, Groq (openai/gpt-oss-120b) as the primary model, Gemini embeddings + Flash as fallback, Claude via the Anthropic SDK and Agent SDK where the project calls for it. `sentence-transformers` for the reranker.