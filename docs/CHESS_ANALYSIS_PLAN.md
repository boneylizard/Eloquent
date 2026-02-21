# Chess Analysis Improvement Plan (Tier 1 + Tier 2)

## Critical context: LLMs are bad at chess

- **Claude 3.7 / GPT-4o**: ~1800 Elo with high blunder rates; hallucinations, illegal moves, "positional memory loss".
- **LLMs cannot "see" the board**: FEN is just text; they cannot evaluate like humans or engines.
- **Implication**: Never ask the model to "understand" or "evaluate" a position. Use it only to **translate engine output** (Tier 1) or **research and cite** external sources (Tier 2).

---

## TIER 1: Real-time commentary (anti-hallucination)

### Goal
Keep Stockfish eval + AI commentary in real time, but the AI **only** turns Stockfish’s eval/PV into plain English. No own chess judgments.

### Approach
- **System prompt**: State explicitly that the model must NOT add its own analysis, only paraphrase the engine’s evaluation, best move, and principal variation.
- **User prompt**: Provide only engine data (eval, best move, PV) and ask for a short, conversational explanation of “what the engine says,” in character.

### Implementation
- New prompts in `chess_ai_service.py`: `_chess_character_system` and the user prompt in `select_move_with_llm` rewritten so the model is framed as a “translator” of engine output, not an analyst.
- Optional: tighten `PER_MOVE_COMMENTARY_SYSTEM` and `GAME_COMMENTARY_SYSTEM` so post-game text is also “based on engine evals only” where applicable.

### Test (5 positions)
Run real-time commentary on these and confirm output only paraphrases engine data:
1. **Starting position** – engine 0.0; commentary should not claim “White is better” or suggest moves not in PV.
2. **Open game** (e.g. after 1.e4 e5 2.Nf3 Nc6 3.Bb5) – check no invented plans.
3. **Tactical** (a position with a clear best move, e.g. capture) – commentary should describe the engine’s line, not “I see a tactic.”
4. **Endgame** (e.g. K+P vs K) – only engine eval and PV in words.
5. **Unclear position** (engine ~0.0) – no “the position is equal” unless the engine says so; stick to “the engine evaluates the position as roughly equal.”

---

## TIER 2: Deep Analysis — agentic research system

### Idea
Triggered by a **“Deep Analysis”** button on a position. The system does **not** ask the LLM to analyze the position. It runs an **agent** that:
1. Takes FEN + Stockfish eval + game context.
2. Uses **tools** to research what strong players/sources say about this or similar positions.
3. Synthesizes findings **with citations**.

Output reads like: “Lichess database shows … [Lichess Explorer]. Daniel Naroditsky discusses this structure in … [YouTube]. In this opening, GMs often … [chess.com/article].”

### Research: APIs and tools

| Tool | What it does | API / source |
|------|----------------|--------------|
| **Lichess Opening Explorer** | Position lookup by FEN; returns move stats, game counts, top moves. | `https://explorer.lichess.ovh/lichess?variant=standard&fen=<FEN>` (URL-encode FEN). Host: explorer.lichess.ovh (not lichess.org). |
| **Lichess Masters** | Same idea for master games. | `https://explorer.lichess.ovh/master?variant=standard&fen=<FEN>` |
| **Lichess Cloud Eval** | Engine eval for a position (optional). | Already have local Stockfish; can skip or use `https://lichess.org/api/cloud-eval` for consistency. |
| **Web search** | “[Opening name] GM analysis”, “position X chess”, forum discussions. | Existing `perform_web_search` / `web_search_service`. |
| **YouTube search** | “Naroditsky [opening]”, “chess speedrun position”. | Web search with “site:youtube.com” or dedicated YouTube API if added later. |
| **Chess forums / articles** | Reddit, chess.com, etc. | Same web search with targeted queries. |
| **Vision / diagram search** | Search by board image. | Optional later; would need image search or diagram-to-FEN then FEN search. |

### Agent workflow (step-by-step)

1. **Input**: FEN, Stockfish eval (and best move/PV if available), optional opening name or move list.
2. **Step 1 — Opening/position context**:  
   - Optionally infer opening name from move list or a small lookup.  
   - Call **Lichess Explorer** (and optionally Masters) with FEN; agent gets move stats and sample games.
3. **Step 2 — Web research**:  
   - Tool: `web_search(query, max_results=5)`.  
   - Agent chooses queries, e.g. “[Opening name] GM analysis”, “[Opening name] plan”, “chess [position description]”.
4. **Step 3 — Optional YouTube**:  
   - Same web search with “site:youtube.com” or a dedicated tool later.
5. **Step 4 — Synthesize**:  
   - Single LLM call with: FEN, engine eval, and **all tool results**.  
   - Instruction: “Summarise what these sources say about this position. Quote or paraphrase with clear citations (e.g. [Lichess Explorer], [chess.com], [YouTube: Title]). Do not add your own chess evaluation.”
6. **Output**: One structured report (markdown or JSON) with sections and citations.

### Function-calling design

- **Tools the agent can call**:
  - `lichess_explorer(fen: str, variant: str = "standard", source: str = "lichess")`  
    - Calls explorer.lichess.ovh/lichess or /master; returns top moves, counts, maybe game links.
  - `web_search(query: str, max_results: int = 5)`  
    - Wraps existing `perform_web_search`; used for GM articles, forums, YouTube (via query design).
- Agent loop:  
  - Prompt: “You have FEN and engine eval. Research the position using the tools; then write a short report with citations. Do not evaluate the position yourself.”  
  - Model either returns final answer or requests tool calls; execute tools, append results to context, repeat until final answer or max steps.

### Implementation outline

- **Module**: `chess_research_agent.py`
  - `query_lichess_explorer(fen, variant, source)` → dict/str.
  - `run_deep_analysis(fen, engine_eval, move_history, model_name, model_manager)` → runs agent loop (2–4 steps), returns `{ "report": "...", "citations": [...], "sources_used": [...] }`.
- **Endpoint**: `POST /chess/deep-analysis`  
  - Body: `{ "fen", "move_history?", "model_name?" }`.  
  - Backend runs Stockfish for eval (and best move/PV), then runs research agent; returns report + citations.
- **UI**: In Chess tab, add **“Deep Analysis”** button (e.g. next to “Analyse game”). On click: send current FEN (and optional move history), show “Researching…” then render report in an expandable or modal panel with citations.

### Error handling

- Explorer or web search fails: return partial report + “Sources X and Y could not be loaded.”
- Rate limits (e.g. Lichess 429): back off, then retry or skip that source and note it in the report.

---

## Summary

| Tier | Purpose | How the LLM is used |
|------|---------|----------------------|
| **1** | Real-time commentary | Only translates Stockfish eval/PV into natural language; no own analysis. |
| **2** | Deep Analysis | Researches external sources via tools; synthesizes and cites; does not evaluate the position. |

Both tiers avoid asking the model to “understand” or “evaluate” the position; Tier 1 relies on the engine, Tier 2 on researched sources and citations.
