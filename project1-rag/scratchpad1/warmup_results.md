# Warmup Q Results — 5 diagnostic questions on Berkshire 2023

## Q1: "What was Berkshire's net earnings in 2023?"

**Prediction:** yes/yes (would retrieve and answer correctly)
**Result:** Retrieved fine; answered $96B with citation p3_c1.
**Verdict:** ❌ Correct-but-misleading. Buffett explicitly calls this figure "worse-than-useless"; operating earnings ($37.4B) is what he says matters. Both are in the source. Model picked literal-match over meaningful-match.
**Failure class:** Extrinsic hallucination via literal grounding — answered the exact question with the exact term match, ignored authorial context.
**Interview angle:** groundedness ≠ usefulness.

## Q2: "How did Charlie Munger and Warren Buffett meet?"

**Prediction:** yes/yes
**Result:** Retrieved perfectly — p1_c0 at rank 1, sim 0.71, contains "it was not until 1959 when he was 35 that I first met him." Model answered INSUFFICIENT_CONTEXT.
**Verdict:** ❌ Over-abstention. Retrieval got the right chunk; generation refused because there's no narrative about *how* (mechanism of meeting), only *when*.
**Failure class:** Over-abstention (mirror of extrinsic hallucination) — strict grounding prompt refuses partially-answerable questions.
**Interview angle:** precision/recall tradeoff on abstention; two competing fixes (loosen prompt vs query rewriting).T

## Q3: "What is Sundar Pichai's opinion on Berkshire?"

**Prediction:** no/no — expected hallucination
**Result:** INSUFFICIENT_CONTEXT (abstained correctly).
**Verdict:** ✅ Abstention mechanism works when it should. Confirms Q2 over-abstention is a strictness tuning issue, not a broken prompt.
**Failure class:** None — this is the desired behavior on an unanswerable question.
**Interview angle:** proves the abstention pattern works; the golden set needs BOTH answerable and unanswerable questions to catch both over- and under-abstention.

## Q4: "Compare Berkshire's insurance and railroad segments in 2023"

**Prediction:** yes/yes — but Claude flagged multi-hop synthesis as a risk
**Result:** Excellent structured comparison across 3 dimensions (financials, expectations vs actuals, operational). Cited 5 distinct chunks across pages 10-14.
**Verification:** All 12+ factual claims checked against source and verified exactly.
**Verdict:** ✅ Working RAG at its best. Multi-hop retrieval succeeded because insurance and BNSF live in the same "Operating Results" section — co-located topics don't stress retrieval the way scattered topics do.
**Failure class:** None — but hidden lesson: this isn't a hard multi-hop test.
**Interview angle:** Comparison questions look multi-hop but often aren't. A real multi-hop test needs topics that live in different sections. Golden set must include both easy (co-located) and hard (scattered) comparisons.
**Minor artifact:** Model said "the author" instead of "Buffett" — grounding-prompt shyness with proper nouns not in immediate context.

## Q5: "List the exact float figures from the insurance operations table."

**Prediction:** no/no — I guessed table parsing would break; couldn't articulate why
**Result:** INSUFFICIENT_CONTEXT. Retrieval returned insurance-related chunks (p13_c2, p14_c0) at 0.66-0.67 similarity — no float figures in them. One chunk literally says "direct newcomers to page 18."
**Verdict:** ✅ Refused correctly BUT for the wrong-looking reason. This looks like a retrieval failure (no chunk had the answer) but is actually a parsing failure at ingest — page 18 IS in the corpus but the table structure was destroyed by pymupdf. Numbers are in the chunks somewhere, disconnected from their column headers, unretrievable by semantic search.
**Failure class:** Parsing failure wearing a retrieval costume.
**Interview angle:** Table extraction is where naive RAG actually breaks; fix is upstream (pdfplumber, keep tables as intact markdown chunks). Similarity scores 0.05 lower than answerable questions — consistent with unanswerable pattern from Q3.
**Bonus catch:** Retrieved chunk p14_c0 contains a *pointer* ("see page 18") — a subtle failure mode where the model can pass documentation smell through to the user.
