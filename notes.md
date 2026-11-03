# Project notes and decisions

## The question
Basketball operations staff need answers from game data, but getting them means writing SQL
or digging through stat sites. This project builds a retrieval-augmented generation (RAG)
system that answers natural-language questions about NBA performance, grounded in actual
game records rather than the language model's own training knowledge.

The distinguishing piece isn't the RAG pipeline itself — those are easy to wire up. It's
measuring whether it actually works: retrieval accuracy, answer correctness, and how often
the system fabricates answers to questions the data can't support.

## Step 1: Data collection

### What was collected
- **Team game logs:** 4,920 team-game rows (complete for both seasons)
- **Player box scores:** 31,597 player-game rows across 1,200 of 2,460 games
- **Coverage:** 585 unique players, 2023-24 and 2024-25 regular seasons
- Source: nba_api 1.12.0 wrapping stats.nba.com. No API key required.

Team-level data is complete; player box scores are partial. Team questions (records, point
differential, home/away splits) can be answered from full seasons, while player averages
come from a subset.

### Rate limiting
stats.nba.com throttles after roughly 600 requests per session regardless of request delay.

- First attempt: 1s delays, 30s timeout → blocked at ~600 games
- Second attempt: 2s delays, 60s timeout, 5-minute backoff after 5 consecutive failures
  → blocked again at ~1,200 games

The 5-minute pause was insufficient. The block appears to be session-level and needs 20+
minutes to clear. Stopped at 1,200 games rather than spending hours fighting the rate limiter.

Checkpointing every 50 games meant no progress was lost across restarts, and the script
resumes automatically by skipping games already in the parquet file.

### Deprecation note
BoxScoreTraditionalV2 is deprecated as of the 2025-26 season but still serves historical
data for 2023-24 and 2024-25. A future version would use V3, which has different column names.

## Decisions log

### Partial collection accepted
The corpus is built from aggregated summaries (one document per player-season, team-season,
and situational split), not per-game documents. Partial game collection therefore affects the
*precision* of averages, not the *structure* of the corpus. Each document records how many
games it summarizes so answers can caveat appropriately.

### LLM backend abstracted
Generation sits behind a single interface (src/llm.py) with Ollama for local development and
an optional hosted API for deployment. Running locally means free, unmetered iteration during
evaluation, which matters when tuning retrieval requires running the full question set
repeatedly. Switching backends is one line in .env.

### Evaluation set written before building
40 questions across six tiers, drafted before any retrieval code existed, so the system
isn't tuned to questions it happens to handle well. Tiers include simple lookups, filtered
queries, comparisons, multi-step reasoning, **out-of-scope questions the system should
decline**, and ambiguous questions it should ground in retrieved stats rather than general
knowledge.

The out-of-scope tier matters most. A RAG system's biggest failure mode isn't missing an
answer — it's confidently inventing one. Questions about contracts, playoff games, draft
history, and seasons outside the corpus test whether the system knows what it doesn't know.



## Step 2: Building the document corpus

### What got built
557 documents, averaging 89 words each:
- **497 player-season documents** (one per player per season, minimum 10 games)
- **60 team-season documents** (30 teams × 2 seasons)

Box scores carry no date or season of their own, so they were joined to the team game logs
to attach GAME_DATE, SEASON, MATCHUP, and WL. The MIN column is text ("34:12"), so whole
minutes were parsed out of it.

### Design decisions in the document text

**Honest about partial collection.** Every player document says "across N games in this
dataset" rather than stating a season average it can't actually support. LeBron James'
2023-24 document covers 69 games; the real figure is higher. The system can only claim what
the corpus contains.

**Splits only when the sample supports them.** Home/road and win/loss breakdowns appear only
when there are 3+ games on each side. Otherwise you get "in wins (1 game) averaged 31 points,"
which is noise dressed up as a fact.

**Percentages computed from season totals, not averaged.** Averaging per-game shooting
percentages overweights low-volume games — a 1-for-1 night counts the same as 10-for-20.
Summing makes and attempts first gives the correct figure.

**Minimum 10 games per player-season.** Below that, averages are too noisy to be useful and
the documents would mostly add retrieval noise.

### Predicted limitation (to test in evaluation)
Tier 4 questions like "which players averaged over 25 points while shooting above 50%" are
probably unanswerable by this corpus. No single document contains a league-wide ranking, and
retrieval returns documents rather than computing aggregations across them. This is a
fundamental property of RAG, not a flaw in the corpus design — recording the prediction now
so the evaluation confirms it rather than appearing to excuse it after the fact.

A hybrid system (route aggregate questions to SQL, everything else to retrieval) would be the
real fix. Noted as a next step.


## Step 3: Embedding and retrieval

### Setup
557 documents embedded with sentence-transformers all-MiniLM-L6-v2 (384 dimensions),
normalized so dot product gives cosine similarity. With a corpus this small, brute-force
numpy search runs in under a millisecond, so no approximate index was needed.

### Finding: pure semantic search fails on filtered queries
Embeddings capture meaning, not constraints. Two failure modes showed up immediately:

**Season is ignored.** "How many points did Nikola Jokić average in 2024-25?" returned the
2023-24 document at 0.804 similarity. "2024-25" and "2023-24" are nearly identical in vector
space, so the model matched on "Jokić points average" and the year barely registered.

**Names confuse the model.** Jokić retrieved Jović and Vučević. Stephen Curry retrieved Seth
Curry. Semantic similarity treats similar-sounding names as similar meanings.

### Fix: hybrid retrieval
Added metadata filtering before semantic ranking — parse the query for season and named
entities, narrow the candidate set, then rank what survives by embedding similarity.

| Query | Semantic only | Hybrid |
|---|---|---|
| Jokić points 2023-24 | Jokić, Jović, Nurkić | Jokić only (1 candidate) |
| Celtics record 2023-24 | Both Celtics seasons + 76ers | Correct season first |
| Luka points 2024-25 | 2023-24 doc, no signal it's wrong | 2023-24 doc + explicit note |

### Finding: similarity scores can't detect out-of-scope questions
Tested whether a score threshold could identify questions the corpus can't answer:

| | Score range |
|---|---|
| In scope | 0.401 – 0.770 |
| Out of scope | 0.335 – 0.547 |

They overlap badly. "What is Nikola Jokić's contract worth?" scored 0.547, higher than two
legitimate in-scope questions. **No threshold separates them**, so refusal has to be handled
at generation time by the LLM, not by filtering on retrieval score.

This is worth stating plainly: retrieval always returns its top-k. It has no concept of
"nothing relevant exists." That property is the root of most RAG hallucination.

### Fix: surface filter failures to the generation step
When a season filter matches zero documents, retrieval now returns `season_applied=False`
plus an explanatory note, instead of silently falling back to the wrong season. That note
gets passed into the generation prompt so the answer can caveat rather than mislead.



## Step 4: Generation

### Setup
Retrieval feeds the top 3 documents into a prompt for Llama 3.1 8B (via Ollama). The prompt
enforces four rules: use only numbers in the context, say what's missing rather than guess,
mention when a figure covers N games rather than a full season, and stay concise. A scope
statement tells the model what the corpus does and doesn't cover, and retrieval notes (like a
season filter that couldn't be applied) are injected directly into the prompt.

Every answer returns its source documents, so any number can be traced back to where it came
from.

### Spot-check results

| Tier | Question | Result |
|---|---|---|
| 1 | Jokić points 2023-24 | Correct, mentioned game count |
| 2 | LeBron road games | Correct split pulled from document |
| 3 | Curry vs Edwards from three | Correct comparison, both docs retrieved |
| 4 | Players with 25+ pts on 50%+ | **Misleading** (see below) |
| 5 | Jokić contract | Correctly refused |
| 5 | Luka 2024-25 points | Correctly refused, using retrieval note |

**The contract refusal is meaningful.** Llama 3.1 very likely knows Jokić's contract from
training data, so declining shows the context-only instruction actually holds.

**Minor inconsistency:** the game-count caveat appeared in some answers but not the Curry vs
Edwards comparison. The rule is applied unevenly.

### Finding: retrieval-limited answers that sound complete
"Which players averaged over 25 points while shooting above 50%?" names no specific player,
so retrieval fell back to semantic similarity and returned three loosely related documents
(Amen Thompson, Jalen Williams, Jalen Green). The model correctly reported that none of those
three met the criteria.

Every sentence in the answer is true, and the answer is still misleading. It treats a
three-document sample as the full population, so a reader could reasonably conclude nobody
in the league met the threshold. The honest answer is that this system cannot rank across
all players.

This is arguably worse than an obvious hallucination, since nothing about it looks wrong.
It confirms the Step 2 prediction that aggregate questions are out of reach for pure
retrieval. Deliberately not fixed yet — measuring the baseline across all 40 questions first,
so the fix can be evaluated against it.


## Step 5: Evaluation (baseline)

### Method
40 questions across five expected behaviors, written before any retrieval code existed so the
system wasn't tuned to questions it happens to handle well:

- **answer** (17) — the answer exists in one or two documents
- **missing** (8) — the entity exists but the specific stat doesn't
- **refuse** (8) — outside the corpus entirely
- **aggregate** (4) — needs ranking across the whole corpus
- **grounded** (3) — subjective, should cite retrieved stats rather than general knowledge

Each question was run through the full pipeline, scored automatically for retrieval recall
and unsupported numbers, then graded by hand.

### Results
**Overall: 33/40 correct.**

| Behavior | n | Correct |
|---|---|---|
| refuse | 8 | 8/8 |
| grounded | 3 | 3/3 |
| answer | 17 | 16/17 |
| missing | 8 | 5/8 |
| aggregate | 4 | 1/4 |

Retrieval recall@3 was 100% on every question with an expected document. When the answer
exists in the corpus, the right document comes back — the hybrid filtering from Step 3 did
its job, and no further retrieval work was needed.

### Failure taxonomy
- **Sample treated as population (3).** League-wide questions answered from the 3 retrieved
  documents. Q29 declared the Lakers (+1.2) the best point differential in 2024-25; the
  Thunder were +12.9.
- **Stat substitution (2).** Answered a related stat that was present instead of saying the
  requested one wasn't. Q11 gave Curry's overall 3PT% when asked for his 3PT% in wins.
- **Fabrication via reasoning (1).** Q26 invented arithmetic ("30 − 11 = 19 wins against
  teams with winning records") to produce a 53.6% figure the data doesn't support.
- **Reasoning error (1).** Q23 retrieved correct numbers and drew the opposite conclusion.

The common thread: **partial relevance is the hallucination trigger, not irrelevance.** Fully
out-of-scope questions (contracts, wingspans, 2016-17 seasons) were refused 8/8 with zero
fabricated numbers. The failures cluster where the right document comes back but doesn't
contain the specific stat asked for.

### Automated checks were misleading in both directions
- The keyword refusal detector missed "not present" and "does not mention," undercounting
  correct refusals at 75% when the true rate was 100%.
- It counted one fabricated answer as a refusal because the word "missing" appeared inside
  invented reasoning.
- The unsupported-number check flagged "Document 1" and legitimate arithmetic as fabrications.
- Most importantly: **the two most dangerous failures use real numbers from the context.**
  Stat substitution and sample-as-population both cite numbers that genuinely appear in the
  retrieved documents, so no number-matching check can catch them.

Manual grading was necessary, not optional. Worth stating plainly, since automated
faithfulness metrics are often presented as sufficient.

### Reproducibility
The initial evaluation ran at Ollama's default temperature. Re-running the identical 40
questions produced different answers for 37 of them — not just rewording, but changes that
altered correctness in both directions. Q26 went from fabricating arithmetic to correctly
declining; Q11 went from a partial answer to explicitly stating a false inference ("we can
infer his three-point shooting in wins is the same as his overall percentage").

Pinned temperature to 0 with a fixed seed, verified by three identical responses to the same
prompt, then re-ran the baseline under those settings so the Step 6 fix could be isolated.

**Takeaway: an LLM pipeline evaluated once at default sampling settings is not measuring the
system, it's measuring one draw from it.**

## Step 6: Query router

### The problem
Three of seven baseline failures were the same thing: questions asking about all players or
all teams, answered from whichever 3 documents retrieval returned. No retrieval improvement
fixes this — retrieval returns documents, not computed rankings.

### The fix
Built a router that detects aggregate questions by pattern (which player/team, most/best/
highest/biggest, over N points) and answers them from pandas tables built from the same
source data, bypassing retrieval and the LLM entirely.

Skipping the LLM is deliberate: the table output is already a clean factual answer, and
paraphrasing it through a language model only adds risk. **When the data can answer directly,
don't paraphrase it.**

### The router refuses more than it answers
Four guards send flagged questions back to retrieval:

1. **Single-game highs** — the tables hold season aggregates, not per-game records
2. **Cross-season comparisons** — need two seasons of data per entity
3. **Questions scoped to a named entity** — not league-wide rankings
4. **Subjective "best player"** with no metric specified

Of 7 questions the pattern detector flagged, the router handled 3 and declined 4.

An earlier version without these guards answered all 7 — and got Q12, Q14, and Q21 badly
wrong by ranking the entire league when the question asked about one team's single-game high.
**A tool that confidently answers the wrong question is worse than one that declines.**

### Minimum-games thresholds
Rate and difference metrics are unstable on small samples. The first version ranked Jaylen
Nowell first on home/road scoring gap off an 11-game sample — technically correct, practically
noise. Added a 40-game floor for gap metrics and 30 games for shooting percentages, with the
exclusion count reported in the answer ("minimum 40 games; 150 player-seasons excluded").

### Results

| Behavior | n | Before | After |
|---|---|---|---|
| refuse | 8 | 8 | 8 |
| grounded | 3 | 3 | 3 |
| answer | 17 | 16 | 16 |
| missing | 8 | 5 | 5 |
| **aggregate** | 4 | **1** | **4** |
| **Total** | **40** | **33** | **36** |

The three fixed answers:

| Q | Before | After |
|---|---|---|
| 25 | "There are no players who averaged over 25 points while shooting above 50%" | Embiid, Giannis, Shai, Durant, Jokić — "5 of 497 player-seasons" |
| 27 | "Georges Niang" (from 3 bench players) | Donovan Mitchell, 40-game minimum, 150 exclusions disclosed |
| 29 | "The Lakers had the best point differential, +1.2" | Thunder +12.9, all 30 teams ranked |

**Verified clean:** with temperature pinned to 0, 0 of 37 retrieval answers changed between
the baseline and routed runs, and all 3 structured answers did. The router is the only
variable in the comparison.

### Remaining failures
All three are in the `missing` category and share one cause: the correct document is retrieved
but doesn't contain the specific statistic asked for, and the model fills the gap instead of
saying so.

- **Q11** — asked for Curry's 3PT% in wins; gave his overall 3PT% and stated "we can infer
  his three-point shooting in wins is the same as his overall percentage"
- **Q26** — invented arithmetic to produce a record against teams with winning records
- **Q28** — read the win/loss game counts (44 wins) as the number of games Giannis scored 30+

The fix would be a stricter prompt rule (name the exact statistic requested and confirm the
context contains it before answering) plus expanding the documents to include the splits these
questions ask for. Not attempted here; recorded as the next iteration.

## Decisions log (additions)

### Manual grading over automated metrics
Automated faithfulness checks (does every number in the answer appear in the context?) caught
fabricated numbers but missed the two failure modes that use real numbers to answer a different
question. Graded all 40 answers by hand against a six-category taxonomy.

### Structured answers skip the LLM
Aggregate questions return the pandas output directly. Routing it through the language model
for "natural phrasing" would introduce distortion risk for no informational gain.

### Router guards over router coverage
Chose to have the router decline 4 of 7 flagged questions rather than attempt all of them.
Measured both versions: the permissive one answered more questions and got three of them
confidently wrong.

### Determinism before measurement
Pinned temperature to 0 and seed to 42, and re-ran the baseline under those settings before
measuring the router's effect. Without this, a 3-question improvement could not be
distinguished from run-to-run variance, which had already changed 37 of 40 answers between
two default-temperature runs.