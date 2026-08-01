# Banking Customer Support AI Agent — Multi-Agent Architecture

Capstone project (Applied Generative AI). A **multi-agent GenAI system** for banking
customer support. A **Classifier Agent** routes each customer message to the right
specialist: a **Feedback Handler** (positive → LLM thank-you; negative → new ticket in a
support database) or a **Query Handler** (extract ticket number → return its status).
Built with **LangChain + OpenAI + SQLite**, with a **Streamlit** dashboard, evaluation,
and logging.

## What's in this folder

| File | Purpose |
|---|---|
| `BankingSupport_Capstone.ipynb` | End-to-end notebook (Part 1 + Part 2) with explanations. |
| `support_agents_core.py` | Engine: the 3 agents, Orchestrator, SQLite `support_tickets` DB, logging, evaluation. |
| `banking_support_app.py` | Streamlit dashboard (Part 2). |
| `requirements.txt` | Dependencies. |
| `.env.example` | Template for your OpenAI API key. |
| `support.db` | SQLite database (auto-created and seeded on first run). |

## Setup (in Visual Studio Code)

1. Open this folder in VS Code and open a terminal (`Terminal → New Terminal`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your OpenAI key — copy `.env.example` to `.env` and paste your key:
   ```
   OPENAI_API_KEY=sk-...
   ```

## Run the notebook
Open `BankingSupport_Capstone.ipynb` and run all cells. It runs even without a key
(rule-based fallback); with a key the LLM agents are active.

## Run the Streamlit app
```bash
streamlit run banking_support_app.py
```
Opens at **http://localhost:8501** with four pages: **Support Assistant** (routing
simulator + one-click test scenarios), **Support Tickets (DB)**, **Logs & Debug**, and
**Evaluation**. Press `Ctrl+C` in the terminal to stop.

## How each capstone requirement is met

**Part 1 — Multi-Agent Design & Execution Logic**
1. **Classifier Agent** — `ClassifierAgent` labels each message Positive / Negative / Query and routes it.
2. **Feedback Handler Agent** — `FeedbackHandlerAgent`: positive → personalized thank-you; negative → unique 6-digit ticket inserted into `support_tickets` + empathetic reply.
3. **Query Handler Agent** — `QueryHandlerAgent` extracts the ticket number and returns its status from the DB.
4. **Agent coordination** — `Orchestrator.process()` runs classify → route → respond → log.

**Part 2 — LLMOps**
7. **Model evaluation** — `evaluate_system()` reports classification accuracy, routing success rate, and test-case coverage; `qa_score_responses()` adds optional LLM quality scoring.
8. **Streamlit UI** — `banking_support_app.py`.
9. **Logs & debugging** — `AgentLogger` traces every interaction and the agent success rate.

## Notes
- **Model:** OpenAI `gpt-4o-mini` (change in `support_agents_core.py`).
- Without a key, the Classifier uses a keyword heuristic and positive replies use a
  template — the full pipeline (routing, ticketing, DB, logging, evaluation) still runs.
- Keep `langchain` on the **0.3.x** line.
