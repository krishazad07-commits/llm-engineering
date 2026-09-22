# Contract Intelligence RAG

Retrieval-Augmented Generation over Berkshire Hathaway's 2023 shareholder letter. Ask a question, get an answer grounded in the document with chunk-level citations, or a clean refusal when the answer is not there.

First of three projects for LLM engineering interview prep. Built to run end-to-end and be measured honestly.

## Results

On a 50-question hand-reviewed golden set:

| Metric                   | Result       |
|--------------------------|--------------|
| Answerable attempted     | 42/42 (100%) |
| Unanswerable abstained   | 8/8 (100%)   |
| Hallucinations           | 0            |
| Over-refusals            | 0            |

Retrieval quality on the 37 scoreable questions (excluding 8 unanswerable + 5 table-dependent), TOP_K = 10:

| Retriever  | Recall@1 | Recall@5 | MRR   | NDCG@10 | Coverage@5 |
|------------|---------:|---------:|------:|--------:|-----------:|
| Vector     |    0.757 |    1.000 | 0.855 |   0.891 |      0.950 |
| Reranked   |    0.784 |    1.000 | 0.879 |   0.910 |      0.973 |
| Contextual |    0.784 |    1.000 | 0.863 |   0.897 |      0.977 |

Recall@10 hits 1.000 across all three retrievers. Aggregates hide what the per-category breakdown shows: reranker gains most of its Recall@1 lift from partial questions (0.667 → 0.833); contextual retrieval helps multi-hop (0.750 → 0.875) but hurts inferential (0.833 → 0.667), because prepending narrative context adds topical noise to reasoning-shaped queries that lack keyword hooks.

## Architecture

```
Ingest (once)
┌─────────────────────────────────────────────────────────────┐
│  PDF (Berkshire 2023, 17 pages)                             │
│    │                                                        │
│    ▼                                                        │
│  pymupdf extract  →  strip page numbers / headers / dots    │
│    │                                                        │
│    ▼                                                        │
│  chunker: recursive split                                   │
│  (1000 chars max, 150 overlap)  →  55 chunks                │
│    │                                                        │
│    ▼                                                        │
│  Gemini embedding (768 dim, MRL-truncated from 3072)        │
│    │                                                        │
│    ▼                                                        │
│  Supabase Postgres + pgvector                               │
│  (content, source_doc, page, chunk_id, embedding)           │
└─────────────────────────────────────────────────────────────┘

Query (per question)
┌─────────────────────────────────────────────────────────────┐
│  question  →  Gemini embed  →  pgvector cosine search       │
│                                (top 10, TOP_K = 10)         │
│                                       │                     │
│                                       ▼                     │
│  build_prompt: system rules + 3 few-shot examples           │
│  (full / partial / abstain) + retrieved chunks in XML tags  │
│                                       │                     │
│                                       ▼                     │
│  Groq openai/gpt-oss-120b generates answer                  │
│                                       │                     │
│                                       ▼                     │
│  answer with [chunk_id] citations, or INSUFFICIENT_CONTEXT  │
└─────────────────────────────────────────────────────────────┘

Eval
┌─────────────────────────────────────────────────────────────┐
│  50 hand-reviewed golden questions                          │
│  (42 answerable + 8 unanswerable, chunk-ID annotated)       │
│    │                                                        │
│    ▼                                                        │
│  retrieval metrics: Recall@k, MRR, NDCG@10, Coverage@k      │
│  generation metrics: attempts, abstentions, hallucinations  │
└─────────────────────────────────────────────────────────────┘
```

## Stack

- Embedding: Google Gemini `gemini-embedding-001`, 768 dim after MRL truncation from 3072.
- Vector store: Supabase Postgres with the pgvector extension. Cosine distance via the `<=>` operator.
- Generation: Groq `openai/gpt-oss-120b`.
- Language: Python 3.12, managed with uv.

Written from scratch. No LangChain, no LlamaIndex. Every piece by hand so I could understand and debug it.

## Decisions and why

**TOP_K = 10.** Tested 5, 15, and 10. TOP_K=5 misses q_022, a multi-hop arithmetic question where the critical chunk sits at rank 6. TOP_K=15 catches it but costs 3× the retrieval tokens per query. TOP_K=10 catches the same chunk with a 4-rank margin against the observed worst-case retrieval depth. Full 42/42 attempts held at both 10 and 15.

**Chunk size 1000 chars, 150 overlap.** Chosen from the Berkshire letter's paragraph structure. No formal chunk-size sweep yet; on the list before the multi-doc extension.

**Cross-encoder reranking on top of vector search.** Fetch top-20 with vector, rerank with `bge-reranker-v2-m3`, keep the top 10. Aggregate Recall@1 goes up by 0.027; the per-category story matters more (partial questions jump from 0.667 to 0.833).

**Contextual retrieval (Anthropic's pattern).** Before embedding, prepend a one-sentence LLM-generated summary situating each chunk in its document, so a chunk that says "the Party shall not be liable" also carries context about which section and contract it came from. Helps multi-hop, hurts inferential, net-flat on the aggregate. Kept because the multi-hop gain matters more than the inferential loss on this corpus.

**Three-path generation prompt.** System prompt allows a full answer, a partial answer, or an `INSUFFICIENT_CONTEXT` abstention. Three worked examples in the prompt, one for each path. Before this change the model was binary (full answer or refuse); after, over-refusals dropped from 8 to 1 in a single commit.

**Groq for generation, Gemini for embedding.** Gemini's per-day generation cap on the free tier killed two full eval runs before I moved generation to Groq. The retry decorator is provider-agnostic so the swap was a 15-line change. Trade-off: `openai/gpt-oss-120b` insists on CJK fullwidth brackets `【】` for citations regardless of the prompt. Citation parsing accepts both.

## Failure modes and what they taught me

**The labeler was the bottleneck, not retrieval.** Day 12 baseline showed Recall@10 = 0.811. Adding hybrid vector+BM25 retrieval on Day 13 seemed to make things dramatically worse: Recall@1 collapsed from 0.568 to 0.054. Two days of debugging showed the retrievers were correctly finding the right chunks; the labeler was doing fragile substring matching on hint text with ellipses in it. Fixed by manually annotating relevant chunk IDs on all 37 scoreable questions. Vector Recall@10 jumped from 0.811 to 1.000. Every retrieval change from Day 15 onward was measured against the hand-annotated labels, not the generated hint substrings.

**A single character in the system prompt broke q_016.** Day 21: a stray `- -` instead of `--` in the prompt was enough to make q_016 refuse a question it should have answered. No retrieval change, no model change. Fixed the typo and the answer came back correct. Prompts are code.

**Groq generates fullwidth brackets for citations.** Even with explicit instructions to use `[]`, `openai/gpt-oss-120b` consistently outputs `【】`. The behaviour is below the prompt layer. Any downstream citation parser has to accept both.

**Free-tier rate limits set the pace.** Two full eval runs died to Gemini's per-day generation cap. Groq then hit its own per-day cap (200K tokens) after two eval runs on Day 23. This is why the pipeline batches embedding calls, uses a provider-agnostic retry with exponential backoff, and writes per-run results into files named by their config (`eval_generation_results_topk10.jsonl`) instead of overwriting a single file.

**Supabase free tier pauses inactive projects.** Silently, after about a week. The pooler stops recognising the tenant and connections fail with "tenant not found". Cost an hour of "why is nothing working" on Day 23 after a 14-day break.

## Known limitations

**Single-document corpus.** The whole system indexes one 17-page PDF. Multi-document ingest is the planned extension after the interview prep sprint.

**No hybrid retrieval in the shipped system.** Tested on Day 13, shelved once the labeler was fixed and vector alone hit Recall@10 = 1.000. On a prose corpus with semantic queries, BM25 has no job. It comes back as a first-class citizen in the multi-doc extension, where exact-token queries (document names, section references, rare proper nouns) will need it.

**No prompt caching yet.** Every request re-sends the full system prompt and examples. Fine at this scale, expensive once the corpus grows. Adding it when generation swaps to Anthropic's API.

**No CI for evals.** The eval harness runs manually. Every prompt or retrieval change should trigger it automatically.

**Tables are degraded.** pymupdf handles prose well and tables badly. On this corpus that costs roughly 3 questions worth of context. A table-heavy corpus would need pdfplumber or a vision-based OCR pass.

**Golden set is small.** 50 questions is enough to distinguish real changes from noise on a 55-chunk index. Not enough for confident category-level claims where the per-category n is under 8.

## How it was built

Day-by-day working log lives in `LOG.md` at the repo root. Every experiment is recorded with the change, the numbers, and what surprised me. The log was written to be readable in six months when I've forgotten why any given decision was made.

Rough phases:
- Days 1-6: Python and API refresher
- Days 7-9: PDF ingest, chunking, first end-to-end RAG
- Days 10-15: golden set, eval harness, labeler audit
- Days 16-17: reranking, contextual retrieval
- Days 18-21: generation, three-path prompt, retrieval observability
- Days 22-24: TOP_K tuning, retrieval consolidation, this README

## Running it

```bash
# Setup
uv sync
cp .env.example .env
# fill in GOOGLE_API_KEY, GROQ_API_KEY, DATABASE_URL

# Ingest (one time per document)
uv run scratchpad1/pdf_ingest.py
uv run scratchpad1/embed_and_insert.py

# Ask
uv run scratchpad1/rag.py "What did Buffett say about Munger?"

# Full generation eval
uv run scratchpad1/eval_generation.py

# Retrieval-only eval (per retriever)
uv run scratchpad1/eval_retrieval.py --retriever vector
uv run scratchpad1/eval_retrieval.py --retriever reranked
uv run scratchpad1/eval_retrieval.py --retriever contextual

# Consolidate the three retrieval-eval outputs into one table
uv run scratchpad1/consolidate_retrieval_evals.py
```

Golden set is checked in at `scratchpad1/golden_qa.jsonl`.