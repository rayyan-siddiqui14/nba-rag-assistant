"""RAG pipeline: hybrid retrieval + grounded generation.
From a notebook: sys.path.append("../src"); import rag
"""
import pickle, re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from llm import generate

ROOT = Path(__file__).resolve().parent.parent

docs = pickle.load(open(ROOT / "data" / "index" / "docs.pkl", "rb"))
embeddings = np.load(ROOT / "data" / "index" / "embeddings.npy")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SEASONS = sorted({d["season"] for d in docs})
PLAYER_SEASONS = sorted({d["season"] for d in docs if d["type"] == "player_season"})

player_idx, team_idx = defaultdict(set), defaultdict(set)
for i, d in enumerate(docs):
    if d["type"] == "player_season":
        full = d["player"].lower()
        player_idx[full].add(i)
        player_idx[full.split()[-1]].add(i)
    else:
        team = d["team"].lower()
        team_idx[team].add(i)
        team_idx[team.split()[-1]].add(i)


def parse_query(q):
    ql = q.lower()
    season = None
    m = re.search(r"(20\d\d)[-–](\d\d)", q)
    if m and f"{m.group(1)}-{m.group(2)}" in SEASONS:
        season = f"{m.group(1)}-{m.group(2)}"
    else:
        m = re.search(r"\b(20\d\d)\b", q)
        if m:
            y = m.group(1)
            for s in SEASONS:
                if s.startswith(y) or s.endswith(y[-2:]):
                    season = s
                    break
    players = {n for n in player_idx if len(n) > 3 and n in ql}
    teams = {n for n in team_idx if len(n) > 3 and n in ql}
    return {"season": season, "players": players, "teams": teams}


def search_hybrid(query, k=3):
    p = parse_query(query)
    candidates = set(range(len(docs)))
    notes = []

    hits = set()
    for n in p["players"]:
        hits |= player_idx[n]
    for n in p["teams"]:
        hits |= team_idx[n]
    if hits:
        candidates &= hits

    season_applied = False
    if p["season"]:
        in_season = {i for i, d in enumerate(docs) if d["season"] == p["season"]}
        narrowed = candidates & in_season
        if narrowed:
            candidates, season_applied = narrowed, True
        else:
            notes.append(f"No documents match {p['season']} for this entity. "
                         f"Player data covers {', '.join(PLAYER_SEASONS)} only.")

    if not candidates:
        return {"results": [], "parsed": p, "season_applied": False, "notes": notes}

    q = model.encode([query])[0].astype("float32")
    q = q / np.linalg.norm(q)
    cand = sorted(candidates)
    scores = embeddings[cand] @ q
    order = np.argsort(-scores)[:k]
    return {
        "results": [{**docs[cand[i]], "score": float(scores[i])} for i in order],
        "parsed": p, "season_applied": season_applied, "notes": notes,
    }


CORPUS_SCOPE = (
    "NBA regular season data only. "
    f"Player statistics cover {', '.join(PLAYER_SEASONS)}. "
    f"Team statistics cover {', '.join(SEASONS)}. "
    "No playoff data, contracts, biographical information, draft history, "
    "or seasons outside this range."
)

PROMPT = """You are a basketball operations analyst. Answer the question using ONLY the \
statistics in the provided context.

Dataset scope: {scope}

Rules:
- Use only numbers that appear in the context. Never estimate, infer, or recall from memory.
- If the context does not contain the answer, say exactly what is missing. Do not guess.
- When a document says "across N games in this dataset," mention that the figure is based \
on N games rather than a full season.
- Be concise. Two or three sentences unless the question needs more.
{notes}
Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question, retrieval):
    context = ("\n\n".join(f"[Document {i+1}]\n{r['text']}"
                           for i, r in enumerate(retrieval["results"]))
               or "(no relevant documents found)")
    notes = ("\nImportant: " + " ".join(retrieval["notes"]) + "\n") if retrieval["notes"] else ""
    return PROMPT.format(scope=CORPUS_SCOPE, notes=notes, context=context, question=question)


def answer(question, k=3):
    retrieval = search_hybrid(question, k)
    text = generate(build_prompt(question, retrieval)).strip()
    return {
        "question": question,
        "answer": text,
        "sources": [r["id"] for r in retrieval["results"]],
        "source_names": [f"{r.get('player') or r.get('team')} {r['season']}"
                         for r in retrieval["results"]],
        "context": "\n\n".join(r["text"] for r in retrieval["results"]),
        "notes": retrieval["notes"],
    }