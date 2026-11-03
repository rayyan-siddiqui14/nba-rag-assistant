# NBA Basketball Operations Assistant (RAG)

> **Question:** Can a retrieval-augmented system answer natural-language questions about NBA performance from actual game data — and more importantly, can you measure whether it's telling the truth?

Wiring up a RAG pipeline is easy. This project is about the part most demos skip: a 40-question evaluation set with labeled expected behaviors, manual grading against a failure taxonomy, and a measured before/after on a targeted fix.

## Key findings

**1. Pure semantic search ignores the constraints that matter.** Asking for Nikola Jokić's 2024-25 stats returned his 2023-24 document at 0.80 similarity — "2024-25" and "2023-24" are nearly identical in embedding space. Names confused it too: Jokić retrieved Jović and Vučević; Stephen Curry retrieved Seth Curry. Adding metadata filtering before semantic ranking brought retrieval recall@3 to **100%** on every question with an expected document.

**2. Similarity scores cannot detect out-of-scope questions.** In-scope questions scored 0.401–0.770; out-of-scope scored 0.335–0.547. "What is Nikola Jokić's contract worth?" scored 0.547 — higher than two legitimate questions. No threshold separates them, because retrieval always returns its top-k and has no concept of "nothing relevant exists." Refusal has to be handled at generation time.

**3. Partial relevance is the hallucination trigger, not irrelevance.** Fully out-of-scope questions were refused 8/8 with zero fabricated numbers, even though the base model knows the answers from training data. Every hallucination occurred where the *correct* document was retrieved but didn't contain the specific statistic asked for.

**4. The two most dangerous failures use real numbers.** Automated faithfulness checks (does every number in the answer appear in the context?) caught fabrications but missed *stat substitution* and *sample-treated-as-population* — both of which cite numbers that genuinely appear in the retrieved documents. Manual grading was necessary, not optional.

**5. A query router fixed every aggregate failure: 33/40 → 36/40.** Questions asking about all players or teams can't be answered by retrieval at any quality level. Routing them to a pandas query over the same source data fixed 3 of 3.

## The system

```
Question
   │
   ├─ aggregate pattern? ──→ guards ──→ pandas query over season tables ──→ answer
   │                           │                                  (no LLM involved)
   │                           └─ declined ─┐
   │                                        ▼
   └──────────────────→ hybrid retrieval (metadata filter → semantic rank)
                                 │
                                 ▼
                        grounded prompt → Llama 3.1 8B → answer + sources
```

**Corpus:** 557 documents (497 player-seasons, 60 team-seasons) generated from 31,597 player-game rows and 4,920 team-game rows pulled from stats.nba.com via `nba_api`.

**Retrieval:** sentence-transformers `all-MiniLM-L6-v2` (384-dim), with entity and season filtering applied before semantic ranking.

**Generation:** Llama 3.1 8B locally via Ollama, temperature 0, seed 42. The generation layer is abstracted so the same code runs against a hosted API for deployment.

## Evaluation

40 questions written **before any retrieval code existed**, so the system isn't tuned to questions it happens to handle. Each is labeled with an expected behavior:

| Behavior | n | Meaning |
|---|---|---|
| `answer` | 17 | The answer exists in the documents |
| `missing` | 8 | The entity exists, the specific stat doesn't |
| `refuse` | 8 | Outside the corpus entirely |
| `aggregate` | 4 | Requires ranking across the whole corpus |
| `grounded` | 3 | Subjective — should cite stats, not general knowledge |

The `refuse` tier matters most. A RAG system's biggest failure mode isn't missing an answer; it's confidently inventing one. Questions about contracts, draft history, playoff games, and seasons outside the corpus test whether the system knows what it doesn't know.

### Results

| Behavior | n | Baseline | With router |
|---|---|---|---|
| refuse | 8 | 8 | 8 |
| grounded | 3 | 3 | 3 |
| answer | 17 | 16 | 16 |
| missing | 8 | 5 | 5 |
| **aggregate** | 4 | **1** | **4** |
| **Total** | **40** | **33** | **36** |

Retrieval recall@3: **100%** on all questions with an expected document.

<table>
<tr><th>Question</th><th>Baseline (retrieval)</th><th>With router</th></tr>
<tr>
<td>Which players averaged over 25 points while shooting above 50%?</td>
<td>"There are no players who averaged over 25 points while shooting above 50%"</td>
<td>Embiid, Giannis, Shai, Durant, Jokić — "5 of 497 player-seasons"</td>
</tr>
<tr>
<td>Which team had the best point differential in 2024-25?</td>
<td>"The Lakers had the best point differential, +1.2"</td>
<td>Thunder +12.9, all 30 teams ranked</td>
</tr>
<tr>
<td>Which players had the biggest home/road scoring gap?</td>
<td>"Georges Niang" (compared 3 bench players)</td>
<td>Donovan Mitchell — 40-game minimum, 150 exclusions disclosed</td>
</tr>
</table>

The point-differential case is the clearest: the baseline didn't hedge at all. It declared one team the league's best from three documents.

## The query router

Aggregate questions get detected by pattern and answered from pandas tables built from the same source data, bypassing retrieval **and the LLM**. The table output is already a clean factual answer; paraphrasing it through a language model adds distortion risk for no informational gain.

**The router declines more than it answers.** Four guards send flagged questions back to retrieval:

1. Single-game highs — the tables hold season aggregates
2. Cross-season comparisons — need two seasons per entity
3. Questions scoped to a named entity — not league-wide rankings
4. Subjective "best player" with no metric specified

Of 7 questions the detector flagged, the router handles 3 and declines 4. An earlier version without these guards answered all 7 and got three badly wrong — ranking the entire league when the question asked about one team. **A tool that confidently answers the wrong question is worse than one that declines.**

**Minimum-games thresholds** were added after the first version ranked a player first on home/road scoring gap off an 11-game sample. Rate and difference metrics are unstable on small samples, so gap metrics require 40 games and shooting percentages 30, with the exclusion count reported in the answer.

## Reproducibility

The first evaluation ran at Ollama's default temperature. Re-running the identical 40 questions changed **37 of 40 answers** — not just phrasing, but changes that altered correctness in both directions. One question went from fabricating arithmetic to correctly declining; another went from a partial answer to explicitly stating a false inference.

Temperature was pinned to 0 with a fixed seed, verified by three identical responses to the same prompt, and the baseline re-run under those settings. The final comparison shows 0 of 37 retrieval answers changed and all 3 structured answers did, confirming the router is the only variable.

**An LLM pipeline evaluated once at default sampling settings isn't measuring the system — it's measuring one draw from it.**

## Data

- **Source:** stats.nba.com via `nba_api` 1.12.0 (no API key required)
- **Seasons:** 2023-24 and 2024-25 regular season
- **Team data:** complete — 4,920 team-game rows
- **Player data:** 31,597 player-game rows across 1,200 of 2,460 games, **2023-24 only**

### Rate limiting
stats.nba.com throttles after roughly 600 requests per session regardless of delay. First attempt (1s delays) blocked at ~600 games; second attempt (2s delays, 60s timeout, 5-minute backoff after 5 consecutive failures) blocked again at ~1,200. The block is session-level and needs 20+ minutes to clear.

Because game IDs were pulled in sorted order, all 1,200 collected games came from 2023-24. The project was scoped accordingly rather than spending hours fighting the rate limiter. The limitation is surfaced in three places: the document text ("across N games in this dataset"), the retrieval return value, and the generation prompt — so the system reports what it doesn't have rather than quietly answering about the wrong year.

## Remaining failures

All three are in the `missing` category and share one cause: the correct document is retrieved but doesn't contain the specific statistic requested, and the model fills the gap instead of saying so.

| Question | Failure |
|---|---|
| Curry's 3PT% in wins | Gave his overall 3PT%, stating "we can infer his three-point shooting in wins is the same" |
| Timberwolves vs. winning teams | Invented arithmetic to produce a record the data can't support |
| Did the Bucks win more when Giannis scored 30+? | Read the win-count (44) as the number of 30-point games |

The fix would be a stricter prompt rule — name the exact statistic requested and confirm the context contains it before answering — plus expanding documents to include the splits these questions ask for.

## Limitations

- **Player data covers 2023-24 only.** Cross-season player comparisons are unanswerable and are tested as out-of-scope questions instead.
- **Partial game collection.** Player averages come from a subset, which affects precision but not corpus structure. Every document states its game count.
- **40 questions is a small evaluation set.** Enough to surface failure modes, not enough for tight confidence intervals on the 36/40.
- **Grading is single-rater.** No inter-annotator agreement.
- **Local 8B model.** A larger hosted model would likely handle the `missing` category better; not tested, since free local iteration mattered more during development.
- **No play-by-play data**, so the documents can't support situational splits like clutch performance or lineup data.

## Next steps

- Stricter prompt handling for the `missing` category, then re-measure against the same 40
- Expand documents with the splits that caused the remaining failures
- Collect 2024-25 player data across multiple sessions to enable cross-season questions
- Compare against a larger hosted model on the identical question set

## Repository

```
├── src/
│   ├── collect.py            # NBA API collection with checkpointing and backoff
│   ├── llm.py                # generation layer (Ollama / hosted API)
│   └── rag.py                # hybrid retrieval + grounded generation
├── notebooks/
│   ├── 01_explore.ipynb      # data exploration
│   ├── 02_build_corpus.ipynb # document generation
│   ├── 03_build_index.ipynb  # embedding, hybrid retrieval, semantic baseline
│   ├── 04_generate.ipynb     # prompt design and spot checks
│   ├── 05_evaluate.ipynb     # 40-question baseline evaluation
│   └── 06_router.ipynb       # query router and before/after
├── eval/
│   ├── eval_set.csv          # 40 questions with expected behaviors
│   ├── grading_sheet.csv     # baseline, manually graded
│   ├── grading_sheet_routed.csv
│   └── before_after.csv
├── data/corpus/              # 557 documents + structured tables (committed)
├── notes.md                  # full analysis log and decisions
└── requirements.txt
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Local generation (free)
ollama pull llama3.1:8b
echo "LLM_BACKEND=ollama" > .env
```

The 557-document corpus and structured tables are committed, so notebooks 03–06 run without re-downloading from the NBA API. To rebuild from source, run `python src/collect.py` (expect multiple sessions due to rate limiting) then notebooks 01–02.

## Author

[Rayyan Siddiqui] · [https://www.linkedin.com/in/rayyan-siddiqui12/] · [rayyansiddiquizz060@gmail.com]