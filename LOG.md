# LOG.md — Experiment Journal

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