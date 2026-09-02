# LOG.md — Experiment Journal
# Krish — LLM Engineering Prep Log

**Started:** Wed 5 Aug 2026
**Target 1 (external, hard):** CV-ready by Mon 22 Sep 2026 — projects on GitHub, READMEs polished, numbers in place. CVs go out end of Sept.
**Target 2 (interview):** ~mid-Oct 2026 (date not yet fixed)

**Plan:**
- Core roadmap (Weeks 2-6): Aug 18 – Sep 16 (natural pace, ~31 sessions)
- Project polish + depth extensions: Sep 17 – Sep 21 (~5 days buffer for READMEs, demo videos, any incomplete work)
- 🎯 CV send: Mon 22 Sep
- Interview prep depth: Sep 23 – ~Oct 14 (mocks, whiteboarding, Anthropic reading, question bank reps)

**Trigger to re-plan:** if projects slip past Sep 19, delay CV send by ≤ 5 days rather than shipping unpolished READMEs. Interview date still assumed mid-Oct.


Every change I make to my LLM systems gets three lines here.
This file becomes my interview answers in Week 6.

Format:
---
### YYYY-MM-DD — [Project] — Short title
**What I changed:** 
**What happened (numbers):** 
**What surprised me:** 
---

## Entries

### 2026-08-05 — Week 0 — Repo setup
**What I changed:** Created llm-engineering repo with three project folders.
**What happened (numbers):** N/A — setup only.
**What surprised me:** Nothing yet. This is the boring day.

### 2026-08-06 — Week 0 — Complete
**What I changed:** Set up full workspace: uv, GitHub repo, 3 API keys, Supabase + pgvector, Claude Code.
**What happened (numbers):** N/A — infrastructure only.
**What surprised me:** How much of "LLM engineering" is really just careful project setup. The actual model call is one line — the surrounding scaffolding is where the discipline lives.

### 2026-08-06 — Week 1 Wed — Pydantic exploration
**What I changed:** Explored Pydantic's default behaviors — negative ints, string→int coercion, unknown fields.
**What happened (numbers):** N/A — behavior study.
**What surprised me:** Pydantic separates *type validation* from *business validation* — `int` accepts negative numbers because that's a domain rule, not a type rule. Also, extra fields silently disappear by default — could hide typos in production.

### 2026-08-06 — Week 1 Wed — Pydantic Booking model
**What I changed:** Built first real Pydantic model (Booking) with Field constraints, Literal for status, extra="forbid" config. Wrote four test cases including three intentional failure modes.
**What happened (numbers):** All four tests pass — valid booking parses, invalid ones raise structured ValidationErrors reporting exactly which field and why.
**What surprised me:** Pydantic collects ALL errors at once — passing both 0 guests AND negative price returned 2 validation errors in one exception. In production this means you can show the user everything wrong in one shot instead of playing whack-a-mole.

### 2026-08-07 — Week 1 Day 2 — async/await
**What I changed:** Learned async/await syntax (translated from JS). Built sequential vs concurrent versions of a fake "customer order" task. Tested asyncio.gather with and without return_exceptions=True.
**What happened (numbers):** Sequential: 7.53s (sum of all delays). Concurrent: 3.00s (bounded by slowest task, not the sum). With return_exceptions=True and one failing task: still ~3s, all 3 outcomes reported (2 success, 1 error). Without it: gather() raises immediately and discards the other two results entirely.
**What surprised me:** Calling an async function doesn't run it — it returns a coroutine object that only executes once awaited. Also: without return_exceptions=True, a failing task doesn't just fail itself — it silently destroys the results of tasks that already succeeded. That's a real production risk if you're not careful.

## Week 1 Day 3 — First Gemini API calls (sync, streaming, errors, tokens) + async + structured output

- **What changed:** Built four small scripts in `project1-rag/scratchpad1/` — `gemini_basics.py` (sync generate_content + inspecting the raw response object), streamed responses via `generate_content_stream`, `gemini_errors.py` (try/except around `errors.ClientError` / `ServerError` / `APIError` with correct ordering), `gemini_tokens.py` (pre-flight `count_tokens`), and `gemini_structured.py` — an async function `recommend_movie(topic)` calling `client.aio.models.generate_content` with `response_schema=MovieRecommendation` (Pydantic), three topics fired concurrently via `asyncio.gather(..., return_exceptions=True)`.
- **What happened numerically:** `"hi"` = 2 tokens (not 1 — structural framing overhead). Longer prompts came in around **5.5–6.2 chars/token**, not the "chars/4" rule of thumb. Three concurrent movie recommendations returned validated Pydantic objects (Heat 1995 / Stalker 1979 / 12 Angry Men 1957) in **<CONCURRENT>s wall-clock vs **<SEQUENTIAL>s** sequential — concurrent bounded by the slowest call, same lesson as Day 2, now on real network I/O.
- **What surprised me:** `except errors.APIError` placed above `except errors.ClientError` silently swallows the 404 into the generic handler — Python matches the first `except` clause that fits, not the most specific one. `ClientError` is a subclass of `APIError`, so ordering matters the same way it does with any inheritance-based dispatch. Also: `response.parsed` fills in *only* when you pass `response_schema` in the config — the field was `None` in every earlier call and I hadn't clocked why until today.

## Week 1 Day 4 — Embeddings & cosine similarity (RAG retrieval, in-memory)

- **What changed:** Built `gemini_embeddings.py` in `project1-rag/scratchpad1/` — generated 3072-dim embeddings via `client.models.embed_content` (`gemini-embedding-001`), compared `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` task_type on identical text, wrote `cosine_similarity()` by hand (no numpy), then built a 5-document in-memory semantic search ranked by similarity against a query.
- **What happened numerically:** Same sentence, doc vs query task_type → 0.87 similarity (not 1.0, confirming asymmetric embeddings). Two completely unrelated sentences (heist vs photosynthesis) → 0.70, not near 0. Full ranked search for "crime and robbery movie" correctly put the heist doc first (0.7158) and, notably, ranked a courtroom-drama doc second (0.6080) above astronauts/chef/photosynthesis — despite zero shared keywords.
- **What surprised me:** Unrelated-sentence similarity landing at 0.70 instead of near 0 — embedding spaces are compressed into a narrow band (anisotropy), so absolute cosine scores aren't a universal "related/unrelated" threshold. Only relative ranking within a query is trustworthy, which is why real RAG systems retrieve top-k neighbors instead of thresholding on a fixed similarity cutoff.

## Week 1 Day 5 — Supabase pgvector retrieval

**What changed:** Set up `documents` table with `vector(768)` column (MRL-truncated from Gemini's 3072-dim output) + RLS enabled. Built `scratchpad1/insert_documents.py` — embeds 5 docs with `task_type=RETRIEVAL_DOCUMENT`, MRL-slices to 768, batched insert via `executemany` with `ON CONFLICT (content) DO NOTHING` for idempotency. Added `UNIQUE(content)` constraint. Built `scratchpad1/search_documents.py` — embeds query with `task_type=RETRIEVAL_QUERY`, runs cosine distance retrieval via pgvector's `<=>` operator with `::vector` cast, returns top-K. Connection via psycopg 3 + `pgvector.psycopg.register_vector`, over Supabase session pooler (aws-0-ap-northeast-1, port 5432).

**Numbers:** 5 docs × 768 dims stored. Query `"crime and robbery movie"` returned heist (sim=0.6774, dist=0.3226), courtroom (0.6063), Mars astronauts (0.5500). Ranking order preserved from Day 4's in-memory search; absolute scores drifted ~5% due to 3072→768 MRL truncation. All 5 rows shared identical `created_at` timestamp — confirmed `executemany` batched into a single transaction.

**What surprised me:** How many invisible layers had to line up for one query to work. The direct-connect DNS failing because Supabase free tier is IPv6-only and my ISP is IPv4 — I didn't know networks had "sides" until today. The `conn.commit()` trap where forgetting one line would silently throw away all my inserts with zero errors. Duplicate rows appearing because I ran the script twice and the database had no reason to complain — it did exactly what I told it to. And `executemany` batching being visible in the identical timestamps across all 5 rows, proving one transaction not five. Most of today's real learning wasn't the pgvector operator — it was the layers underneath: connections, transactions, constraints, network topology. The vector search itself was almost the easy part.

## Week 1 Day 6 — Hand-rolled agent loop (Gemini function calling)

**What changed:** Built `scratchpad1/agent_loop.py` — manual ReAct loop with 3 fake tools (get_weather, get_time, add). Tool schemas as `types.FunctionDeclaration`, dispatch via TOOL_REGISTRY dict + `**args` unpacking, errors returned as tool_result dicts instead of raised, MAX_STEPS safety cap, parallel function calling handled via iterating `model_content.parts`. Then ran a break-it experiment: commented out `history.append(model_content)` to test the "function_response must be preceded by matching function_call" invariant. Added ruff to dev dependencies, ran `check --fix` + `format`, cleaned up 14 issues across scratchpad1/ files (import ordering, timezone-naive datetime in gemini_structured.py, kept a documented `# noqa: BLE001` at the tool-dispatch boundary in agent_loop.py).

**Numbers:** 2 loop trips for a 3-tool parallel query ("weather in Ahmedabad + Mumbai + add 47+89"). Model called all 3 tools in one turn (1 Content with 3 function_call parts), returned final answer in trip 2. Break-it experiment: `history` went from 3 entries (user, model, user) to 2 entries (user, user) — no model turn — and Gemini answered coherently anyway. Ruff: 12 auto-fixed, 2 required judgment calls.

**What surprised me:** Gemini is more forgiving than I expected about conversation structure. I predicted a 400 error when I removed the model's tool-call turn from history, leaving only the tool results with no matching call. Gemini just answered anyway. Anthropic strictly enforces this pairing (tool_use_id must match); Gemini apparently doesn't. Takeaway for portable code: always follow the strict shape even when the lenient API lets you cheat, because production may swap providers. Also — my BLE001 blind-except was flagged, and realizing it was one of the few *correct* uses (tool dispatch trust boundary) taught me that linter warnings need judgment, not blind compliance.

## Week 1 Day 7 — Real PDF ingestion (parse + chunk)

**What changed:** Started Project 1 real corpus. Added `pymupdf` dep. Downloaded Berkshire Hathaway 2023 shareholder letter (real 17-page PDF, prose-heavy with financial tables). Built `scratchpad1/pdf_ingest.py` with per-page extraction + regex cleanup (strips dot leaders `re.sub(r"(\s*\.\s*){3,}", ...)`, strips trailing page-number footers, normalizes whitespace). Built `scratchpad1/chunker.py` — hand-rolled recursive character splitter with separator hierarchy (`\n\n` > `\n` > `. ` > ` ` > `""`), greedy merging with overlap, hard invariant `overlap < max_size` enforced with ValueError.

**Numbers:** Naive extraction: 50,779 chars. After cleanup: 41,698 chars (~18% noise removed). 16 non-empty pages, avg 2,606 chars/page. Chunked at max_size=1000, overlap=150 → **55 chunks total**, avg 852 chars, min 166, max 1000 (invariant held). Table pages still degraded — columns collapse to separate lines, documented as known limitation, deferred to pdfplumber layer later.

**What surprised me:** Two things. (1) I proposed adding pdfplumber today to "fix the 10% table problem" — got pushed back on scope creep. The reasoning stuck: measure before optimizing, ship MVP end-to-end before perfecting one layer. That's a real engineering discipline, not just theory. (2) The recursive splitter silently drops the `.` when splitting on `". "` because `str.split()` consumes its delimiter — subtle correctness bug that only shows up when you actually look at the chunks. "Read the output like a diagnostician" caught it; a shipped-and-forgotten implementation wouldn't have.

## Week 1 Day 8 — Real RAG end-to-end (embed → insert → retrieve on Berkshire corpus)

**What changed:** Schema migration: added nullable `source_doc`/`page`/`chunk_id` columns, dropped legacy `documents_content_unique` constraint, added `documents_chunk_unique unique nulls not distinct (source_doc, page, chunk_id)`, deleted 5 Day 5 seed rows to clear NULL-duplicate collision. Built `scratchpad1/embed_and_insert.py` — async batched embedding with `asyncio.Semaphore(3)` capping in-flight batches, `BATCH_SIZE=10` for API efficiency, `return_exceptions=True` on `asyncio.gather` for fault tolerance. `executemany` insert with `ON CONFLICT DO NOTHING` for idempotency on re-runs. Updated `search_documents.py` to select and display `source_doc`/`page`/`chunk_id` for citations.

**Numbers:** 55 chunks × 768 dims embedded in **3.98s** wall clock (72ms/chunk) — vs estimated ~55s sequential = **~14× speedup** (batching + concurrency combined). Insert of 55 rows in 4.49s. Semaphore visibly working — batches 3-5 waited for slots to free before starting. Retrieval on "What did Warren Buffett say about Charlie Munger?" returned top-3 all from page 1 (the Munger tribute), similarity scores 0.7418 / 0.7353 / 0.7251 — semantically correct clustering, though rank 1 and 2 flipped from ideal (chunk 1 with direct quote beat chunk 0 with the title by 0.006 — anisotropy-noise territory).

**What surprised me:** Two things landed hard. (1) Async concurrency's speedup was even bigger than I intuited — 14× rather than the ~10× I would have guessed. Every layer (batching within the API call + concurrent batches under semaphore) multiplies. (2) The rank flip on the Munger query — chunk 1 beat chunk 0 by a rounding error, even though chunk 0 is the canonical "about Munger" content. This is exactly why production RAG has a reranking layer: vector retrieval nails the *cluster*, but within the top cluster the ordering is anisotropy-noise-limited. First time I've *seen* the case for cross-encoder reranking rather than read about it.

## Week 2 Day 9 — Full RAG pipeline glued end-to-end (retrieval + generation)

**What changed:** Built `scratchpad1/generate_answer.py` — the generation half of RAG. Defined `RetrievedChunk` dataclass, `SYSTEM_PROMPT` enforcing "answer only from context / cite chunk IDs / INSUFFICIENT_CONTEXT fallback" per §7.3, `build_prompt` wrapping chunks in `<chunk id="..." source="..." page="...">` XML tags with worst-to-best ordering (so best chunk sits nearest the question — "lost in the middle" mitigation), and `generate_answer` calling Gemini with temperature=0. Built `scratchpad1/rag.py` — the pipeline glue: `rows_to_chunks` converting `search()` tuples → `RetrievedChunk` dataclass instances (with composite `chunk_id` string `f"{source}_p{page}_c{idx}"` for machine-checkable citations), and `ask()` orchestrating the four stages (embed → retrieve → convert → generate). Hit and diagnosed three real errors along the way: (1) `NameError` from missed import, (2) `TypeError` from `api_Key` capital-K typo, (3) `404 NOT_FOUND` because `gemini-2.5-flash` was deprecated mid-session — updated to `gemini-3.6-flash` per the API's error message.

**Numbers:** ~110 lines total across both new files. Pipeline verified end-to-end via the traceback — embed call succeeded (embedding endpoint still working), Supabase retrieval returned rows, tuple→dataclass conversion completed, `generate_answer` was reached and made the API call. Final Gemini generation call blocked by upstream 503 (Google server overload, not code failure) — SDK's built-in `tenacity` retry already exhausted before giving up, confirming this was capacity-side, not transient.

**What surprised me:** Two things. (1) Model deprecation is a real, live production concern — Gemini retired 2.5-flash mid-session and the API told me the successor name directly in the error message. If I'd hardcoded that model string in 5 places instead of one config constant, this would have been a 5-file grep instead of a 1-line fix. Real motivation for "model IDs live in exactly one config constant per role." (2) The 4xx vs 5xx split showed up organically in the same session — the 404 was my bug (well, the API's schedule), fix once; the 503 was theirs, wait and retry. Different error classes, different responses. The Gemini SDK's built-in tenacity retry already handled the backoff for me — if I still see the 5xx after that, it's a real outage, not something more retries will fix.

**Time check:** Day 13 of 42 (calendar) / Day 9 of ~28 sessions (worked). Currently at end of Week 2 Day 4 equivalent on the roadmap. Owed for Week 2: golden dataset (Tue) + eval harness (Wed). Slipping 2 calendar days into Week 3's slot; absorbed by buffer, not compressed. Target unchanged.

## Week 2 Day 10 (part 2) — Golden set generator built, blocked by API 503

**What changed:** Full session on the golden dataset infrastructure. Ran 5 diagnostic questions through naive RAG to surface failure taxonomy (extrinsic-via-literal-grounding on Q1, over-abstention on Q2, correct-refusal on Q3, working multi-hop on Q4, parsing-failure-disguised-as-retrieval on Q5). Added debug print to `ask()` in `rag.py` so retrieved chunks + similarity scores are always visible before generation — this was what turned Q5 from "working correctly" into "silently broken at parse time." Designed golden set schema (7 fields: id, question, expected_answer, relevant_chunk_ids, category, difficulty, answerable) with `answerable` as a behavioral flag (does correct behavior include a real answer?) not a linguistic feel. Locked category mix at 15 extractive / 8 inferential / 8 multi_hop / 6 partial / 5 table_dependent / 8 unanswerable = 50 questions. Built `scratchpad1/build_golden_set.py`: 6 category-specific prompts (each with concrete question shapes, not just properties, in Bullet 1 — the shape-vs-property distinction was a real prompt-writing lesson), `generate_candidates_for_category()` using Gemini structured outputs (response_mime_type="application/json") with temperature 0.7 for question variety, and `main()` that loads the PDF once, loops through 6 categories, assigns zero-padded IDs (q_001 not q_1 to avoid alphabetical-vs-numeric sort bugs), writes to JSONL.

**Numbers:** 6 prompts written, all reviewed and locked. Total ~250 lines in `build_golden_set.py`. Ran the generator — got a 503 on the very first category (extractive) before any candidates were generated. This is the 3rd 503 in 2 days from Gemini free tier. Warmup queries earlier in session all worked, so it's peak-demand deprioritization, not systemic. Decision logged: if 503 recurs tomorrow, activate Groq fallback (~30 min to add, unblocks all remaining eval work).

**What surprised me:** Two things. (1) Prompt-writing has a specific failure mode I didn't expect — pattern-matching from memory instead of literal copy-paste of a template introduces silent bugs (`"questions"` vs `"question"`, `"Multi_hop"` vs `"multi_hop"`, missing quotes around JSON string values). Copy-paste is discipline, not laziness. MULTI_HOP took 3 review cycles because I was retyping instead of copy-pasting from INFERENTIAL. (2) The shape-vs-property distinction in category definitions matters more than I realized — my first INFERENTIAL bullet described *what an inferential question isn't* (not directly stated); the improved version described *concrete shapes* (calculation, comparison, synthesis). Same category, night-and-day difference in what Gemini would generate. PARTIAL and TABLE_DEPENDENT (written after this landed) had zero issues on first pass.

## Week 2 Day 11 — Groq fallback deferred, golden set generated + reviewed (with a caveat)

**What changed:** Started by retrying the Gemini generator from yesterday's 503 block — hit a 4th consecutive 503 on `gemini-3.6-flash`, so before building the planned Groq fallback, tried swapping the generator model to `gemini-3.6-flash-lite` (one-line change to the `GEN_MODEL` constant). That unblocked the run immediately — all 6 categories generated cleanly, 50/50 candidates written to `golden_qa_draft.jsonl`. Decided this doesn't retire the Groq fallback plan; free-tier 503s are a serving-capacity problem, not a flash-vs-flash-lite problem, and it'll resurface once eval runs put real load on the API in Week 3.

Built `scratchpad1/review_golden_set.py` — a terminal hand-review harness with accept/edit/reject/skip/quit, streaming (append-per-decision) writes to `golden_qa.jsonl` and `rejections.jsonl` for crash safety, and resume support via `load_reviewed_ids()` reading both output files at startup so a crash mid-review doesn't lose completed work. Wrote every function myself from scaffolds — `load_reviewed_ids`, `display_candidate`, `get_decision`, `edit_candidate`, `main()`'s dispatch loop — except `get_rejection_reason` and `load_drafts`, which were handed to me directly under real time pressure near the end of the session.

Ran the full review against the Berkshire PDF side-by-side. Result: 50/50 accepted, 0 edits, 0 rejects.

**Numbers:** 50 candidates reviewed, 50 accepted, 0 edited, 0 rejected — a 0% reject rate against my own pre-review prediction of ~25% reject / ~20% edit. Review was done under real time pressure (last ~30 min before I had to leave), and the terminal transcript shows every single one of the 50 decisions as a straight `a`, including two questions (`q_018`, `q_020`) that were flagged as likely category mislabels *before* review started — both are phrased as `inferential` but the source states the answer directly (no reasoning step), so they should almost certainly be `extractive`. Neither got edited. `q_027`'s Greg Abel "born in Canada" claim was also flagged for a fact-check and wasn't independently reconfirmed in the transcript.

**What surprised me:** A 0% reject/edit rate landing exactly on the categories I'd pre-flagged as risky is itself a signal, not a clean pass. The most honest read: my category-label check dropped out under time pressure in favor of just checking "is the stated fact correct" — those are two different checks, and I was only reliably doing one of them by the back half of the review. `golden_qa.jsonl` is populated and unblocks tomorrow's eval harness work, but it isn't fully trustworthy yet on category correctness. Concrete follow-up before running evals: recheck `q_018` and `q_020` categories (extractive, not inferential), confirm `q_027`'s Greg Abel fact, and do a quick spot-check pass on 5-10 more questions picked at random rather than trusting the sequential review was uniformly rigorous all the way to q_050.

## Week 2 Day 12 — Retrieval eval harness built, baseline captured

**What changed:** Refactored `search_documents.py` to expose a clean `retrieve(client, conn, query, k)` wrapper with caller-owned client + connection lifecycle (dependency injection, ~15 min). Built `eval_retrieval.py`: `load_scoreable_questions()` (filters out 8 unanswerable + 5 table_dependent = 37 scoreable), `is_relevant()` (regex-strips `(Page N)` markers, splits on `;` for multi-segment hints, substring-matches any segment), `label_ranking()`, and 4 metric functions (`recall_at_k`, `precision_at_k`, `reciprocal_rank`, `ndcg_at_k`) each unit-tested by prediction against hand-crafted inputs before wiring. Built `report()` — aggregate + per-category breakdown. Also built `inspect_failure.py` — throwaway diagnostic tool that dumps top-10 retrieval + labeling for any single question ID, used to inspect the 7 recall@10 failures.

**Numbers:** Baseline retrieval eval over 37 scoreable questions from `golden_qa.jsonl`. Overall: **recall@1 = 0.568, recall@5 = 0.811, recall@10 = 0.811, precision@5 = 0.173, MRR = 0.676, NDCG@10 = 0.710**. Per-category recall@10: extractive 0.941 (n=17), inferential 0.500 (n=6), multi_hop 0.875 (n=8), partial 0.667 (n=6). recall@5 == recall@10 across every category — when retrieval finds the right chunk, it's always within top-5; never rank 6-10. Inspected 2 of 7 failures via `inspect_failure.py`: q_026 (multi_hop, BNSF+BHE) and q_022 (inferential, Ajit Jain) — both were **labeler false-negatives, not retrieval failures**. Chunks contained the right content but hints had ellipses (`...`) and cross-page concatenations that no single chunk can substring-match. True Recall@10 is likely 0.85-0.90 with cleaner hints; the 0.811 number is a floor.

**What surprised me:** [YOUR HONEST LINE — options: "The recall@5 == recall@10 identity was more informative than the absolute numbers — it told me the ranking is either right or wrong, never almost-right, which means Week 3 needs better ranking (rerank, hybrid) not deeper retrieval." OR "The inferential category scoring 0.500 while extractive scored 0.941 was surprising until inspection showed the gap was mostly hint-format issues in Gemini's generated hints, not real retrieval failure." OR write your own.]

## Week 3 Day 13 — Hybrid search built; discovered eval labeler is the bottleneck, not retrieval

**What changed:** Added `rank_bm25` dependency via `uv add`. Built `scratchpad1/hybrid_search.py` with four functions: `tokenize()` (lowercase whitespace-split — deliberately naive, MVP-first), `build_bm25_index()` (loads all 55 chunks from Supabase once, tokenizes, constructs `BM25Okapi`, returns `(index, chunks)` with parallel-list mapping), `search_bm25()` (tokenize query → `get_scores` → sort desc → top-k in same 6-tuple shape as vector search), `rrf_fuse()` (dict-based score accumulation with `k=60` dampening, `chunk[0]` DB primary key as the join identity), and `retrieve_hybrid()` orchestrating the whole chain. Refactored `eval_retrieval.main()` to build the BM25 index once inside the connection `with` block before the eval loop, then call `retrieve_hybrid` instead of `retrieve`. Built `scratchpad1/debug_hybrid.py` as a throwaway single-query diagnostic printing vector-only and BM25-only top-10 side-by-side for arbitrary query strings.

Two real correctness bugs caught and fixed during the build. **First**: initial `rrf_fuse` used `chunk[3]` (the composite `chunk_id`) as the fusion join key, but discovered via `print(chunks[0])` in the REPL that `chunk_id` in the DB is a per-page index (integers like `0`, `1`, `2`), not globally unique across pages. Chunks with the same `chunk_id` from different pages were silently merging under a single dict key in `chunk_lookup`, collapsing to 3 results when `k=5` was requested. Fixed by switching to `chunk[0]` (the auto-increment DB primary key). **Second**: standard REPL-caching bug — file edits weren't visible until process restart. Both lessons logged.

**Numbers:** Ran the eval harness with hybrid retrieval against the same 37 scoreable questions from Day 12.

| Metric | Vector (Day 12) | Hybrid (Day 13) | Delta |
|---|---|---|---|
| Recall@1 | 0.568 | 0.054 | **–0.514** |
| Recall@5 | 0.811 | 0.730 | –0.081 |
| Recall@10 | 0.811 | 0.811 | 0.000 |
| MRR | 0.676 | 0.239 | –0.437 |
| NDCG@10 | 0.710 | 0.376 | –0.334 |

**What surprised me:** Recall@10 completely unchanged while Recall@1 collapsed by 91% — that combination is diagnostic. Retrieval is finding the same right chunks at the same rate; it's just reordering them out of position 1. My first instinct was "hybrid retrieval is broken" — but `debug_hybrid.py` on the Ajit Jain query showed **both vector and BM25 correctly agreeing on id=57 (page 14) at rank 1**, meaning RRF fusion *should* place it firmly at fused rank 1 too (score ≈ 0.033, well above any single-list contributor at ≈ 0.015). Cross-referenced against `is_relevant()`: the golden hint for q_022 is `"Ajit Jain had not joined Berkshire in 1986. ... February 24, 2024"` — a cross-page concatenation with `...` in it that literally cannot appear as a substring in any single chunk. **The Recall@1 collapse isn't hybrid degrading retrieval — it's the labeler false-negative problem from Day 12 becoming visible under reranking.** Vector-only happened to place labeler-friendly extractive chunks at rank 1 often enough to score 0.568; hybrid slightly reorders those same chunks, and the fragile substring-match labeler now says False for the exact same correct retrievals. The real Recall@10 = 0.811 identity between vector and hybrid confirms the retrievers are equivalent at the "did we find it" level; MRR/NDCG differences are labeler artifacts, not retrieval facts. Bigger lesson: I wasn't measuring retrieval quality — I was measuring the interaction between retrieval quality and labeler robustness. Cannot fairly compare retrieval strategies until the labeler is a stable measuring stick. Day 14 opens with manually annotating `relevant_chunk_ids` on the 37 scoreable questions, upgrading `is_relevant()` to prefer ID-based matching with hint-substring as fallback, then re-running both vector and hybrid evals against the honest labeler.

Day 14 (Aug 25, 2026):

What changed:
- Built annotate_golden_set.py: interactive harness that shows top-15 vector candidates per question and prompts for DB primary key labels
- Built show_chunk.py: on-demand full-content viewer for one chunk by DB id
- Annotated relevant_chunk_ids on all 37 scoreable questions
- Adopted rule: single best chunk (or minimum set); [] if no top-15 chunk contains the answer; for inferential/multi-hop, label chunks with raw facts needed for reasoning

What happened:
- 37/37 questions labeled with DB primary key integers
- 4 questions required audit/re-annotation after catching a preview-truncation bug (PREVIEW_CHARS=160 was cutting off exactly where numeric answers lived)
- Bumped PREVIEW_CHARS to 400; still needed show_chunk.py for boundary cases
- Pattern noticed: many multi_hop questions use adjacent chunks (19+20, 49+50, 58+59), suggesting parent-document / context-window retrieval as a Week 3 experiment
- Category noise found: some multi_hop questions collapse to single-chunk in actual chunking (q_028, q_029)

What surprised me:
- The preview bug was silently corrupting my judgments — caught only when I trusted my instinct that "the answer should be in the letter, so why isn't it in these chunks"
- Partial questions are cleaner to label than expected: single chunk covers the answerable half, the unanswerable half is generation's problem
- q_022 required exactly the ellipsis-cross-page pattern that broke yesterday's labeler — the ID-based labeler will finally credit retrieval correctly on it tomorrow

## Week 3 Day 15 — Eval labeler fixed; vector baseline established

**what changed:** Added ID-based relevance labeling to `is_relevant()` using `relevant_chunk_ids`, with hint-substring matching retained only as a backward-compatible fallback. Added `chunk_coverage_at_k()` to measure how many annotated relevant chunks are retrieved, exposing multi-hop gaps. Updated `label_ranking()`, metric reporting, and result generation for the new evaluation.

**Numbers:** Re-ran the same 37 scoreable questions with vector retrieval and the new labeler.

| Metric            | Vector (Day 12 old) | Vector (Day 15 new) |
| ----------------- | ------------------: | ------------------: |
| Recall@1          |               0.568 |           **0.757** |
| Recall@5          |               0.811 |           **1.000** |
| Recall@10         |               0.811 |           **1.000** |
| Precision@5       |                   — |           **0.222** |
| MRR               |               0.676 |           **0.855** |
| NDCG@10           |               0.710 |           **0.891** |
| Chunk coverage@5  |                   — |           **0.950** |
| Chunk coverage@10 |                   — |           **0.977** |

**By category:**

| Category    |  n |   R@1 |   R@5 |  R@10 |   P@5 |   MRR | NDCG@10 | Coverage@5 | Coverage@10 |
| ----------- | -: | ----: | ----: | ----: | ----: | ----: | ------: | ---------: | ----------: |
| Extractive  | 17 | 0.765 | 1.000 | 1.000 | 0.200 | 0.858 |   0.894 |      1.000 |       1.000 |
| Inferential |  6 | 0.833 | 1.000 | 1.000 | 0.200 | 0.917 |   0.938 |      0.917 |       1.000 |
| Multi-hop   |  8 | 0.750 | 1.000 | 1.000 | 0.275 | 0.817 |   0.861 |      0.833 |       0.896 |
| Partial     |  6 | 0.667 | 1.000 | 1.000 | 0.233 | 0.833 |   0.877 |      1.000 |       1.000 |

**What surprised me:** The new labeler completely changed the interpretation of the old results. Day 12's **Recall@10 = 0.811 ceiling was a labeler artifact**; real vector Recall@10 is **1.000**, with Recall@5 also reaching **1.000**. The new coverage metric shows that retrieval is not equally complete for every question: overall coverage is **0.950@5** and **0.977@10**, with multi-hop being the weakest at **0.833@5 / 0.896@10**.

The hybrid run was unexpectedly poor at Recall@1 despite maintaining Recall@10 = 1.000, showing that the problem is primarily **ranking/reordering rather than retrieval recall**.

## Week 3 Day 16 — Reranker built; net +1 Recall@1, but real churn underneath

**What changed:** Added `rerank_search.py` — two-stage retrieval: vector wide_k=20 → `bge-reranker-v2-m3` → top-10. Refactored `eval_retrieval.py` to support `--retriever {vector, hybrid, reranked}` with per-retriever output files so runs no longer overwrite each other. The reranker is lazy-loaded and returns the same tuple shape as `retrieve()` for a drop-in swap.

**Numerical results:** Re-confirmed vector baseline at R@1 = 0.757, MRR = 0.855, NDCG@10 = 0.891, Coverage@5 = 0.950. Reranked: R@1 = 0.784, MRR = 0.879, NDCG@10 = 0.910, Coverage@5 = 0.973. Delta: +0.027 R@1, +0.024 MRR, +0.019 NDCG@10, +0.023 Coverage@5. Recall@10 stayed at 1.000, as predicted — reranking can reorder retrieved candidates but cannot recover candidates that were never retrieved. Per-question breakdown: 5 fixes, 4 losses, 4 still-broken → net +1 question. Biggest category win was partial (R@1: 0.667 → 0.833). Inferential showed no movement across the metrics.

**What surprised me:** My prediction was too optimistic. I predicted R@1 = 0.865 based on the expectation that the cross-encoder would move most rank-1 misses into the correct position; the actual result was 0.784. More importantly, the reranker did not simply improve the existing ranking — it introduced churn: 5 questions were fixed, but 4 previously-correct questions were pushed into failure. The aggregate numbers are positive, but the per-question result shows that the reranker is not strictly better than vector retrieval on this corpus; it has a different failure profile. Net +1 Recall@1 is the result, not the whole story.

# Day 17 — Contextual Retrieval
## What changed
Today I worked on **contextual retrieval**. The basic idea was to add some extra context to the existing chunks before creating their embeddings, so that the chunks have more meaning when they are searched.
I made 3 main scripts for this:
* `gemini_generate.py` — generates the extra context for the chunks.
* `embed_contextual.py` — creates embeddings using the contextualized chunks.
* `ingest_contextual.py` — puts the contextual embeddings into the database so I can test them.
The tricky part wasn't the scripts, it was the small design calls. I made the embedding column nullable so ingest and embed could run as separate resume-safe scripts. And I hit an ordering bug when picking neighbor chunks — chunk_id is position-within-page (0-3), not document order — so the correct ORDER BY is (source_doc, page, chunk_id). Caught it on the 3-chunk smoke test before scaling.
## Numerical Results
| Metric      | Vector |  Reranked | Contextual |
| ----------- | -----: | --------: | ---------: |
| Recall@1    |  0.757 |     0.784 |  **0.784** |
| Recall@5    |  1.000 |     1.000 |      1.000 |
| Recall@10   |  1.000 |     1.000 |      1.000 |
| MRR         |  0.855 | **0.879** |      0.863 |
| NDCG@10     |  0.891 | **0.910** |      0.897 |
| Coverage@5  |  0.950 |     0.973 |  **0.977** |
| Coverage@10 |  0.977 | **0.986** |      0.977 |
The result was honestly a bit surprising. Recall@1 only went from **0.757 to 0.784 (+0.027)**, which is the same Recall@1 that I got with the reranker.
Recall@5 and Recall@10 were already at **1.0**, so contextual retrieval couldn't really improve those. The interesting part was Coverage@5, which increased to **0.977**, the best out of the three setups.
## What surprised me
My guess for the inferential drop: reasoning-shaped questions don't have obvious keyword hooks, so prepended narrative context just adds topical noise instead of helping. And even though contextual matched the reranker on R@1, they're doing different things — reranker reorders, contextual changes the representation — so they're not really competitors, could probably stack them.

# Day 18 – Generation + Abstention Eval

## What changed

Today I built the **generation half of RAG** and connected retrieval to Gemini for grounded answers. I added `generate_answer.py` with context-only answering, composite citations, and `INSUFFICIENT_CONTEXT` for questions that cannot be answered from the retrieved chunks.

I also refactored `gemini_generate.py` into a generic retry decorator for 429/503 errors and applied it to the embedding call as well. Then I built `eval_generation.py` to test all golden questions for hallucinations and over-refusals.

## Problems / Findings

The sanity test worked: an answerable question produced a grounded answer with the correct citations, while an unanswerable question correctly returned `INSUFFICIENT_CONTEXT`.

The full eval hit the **free-tier embedding rate limit at question 11**, so I couldn't get final generation metrics yet. I also fixed a Windows `cp1252` encoding issue by using UTF-8 explicitly.

One important thing I learned was that **retries and rate-limit pacing are different**. Retries help with temporary 429/503 failures, but repeated rate limits need delays between requests.

## What surprised me

I expected the retry logic to handle the rate-limit problem, but it didn't because the limit was sustained rather than temporary. It made me realize that a production RAG pipeline needs both **error recovery and request pacing**.

**# Day 19 – Generation Eval + Quota Failure**

**## What changed**

Today I added **incremental JSONL writes with `flush()`** to `eval_generation.py` and added `time.sleep(4)` between questions to pace API requests. I also fixed the `load_golden_set()` file-opening bug.

**## Problems / Findings**

The full 50-question eval completed **20 questions successfully** before crashing at question 21 due to Gemini's **per-day generation quota**. All 20 completed answers were correctly attempted and cited.

The incremental writes worked as intended: all **20 results were saved despite the crash**. I also learned that **per-minute and per-day quotas are different problems** — retries can help with temporary limits, but not a daily cap.

**## What surprised me**

The completed answers were strong: **q_005** correctly surfaced conflicting numerical values, and **q_017** performed arithmetic from retrieved data with citations.

The partial result was **20/20 answerable attempts (100%)**, but no abstention rate could be measured because no unanswerable questions were reached.
