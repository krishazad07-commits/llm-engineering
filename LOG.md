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