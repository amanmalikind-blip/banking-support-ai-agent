"""
Banking Customer Support AI Agent — Multi-Agent Core Engine
===========================================================

A multi-agent GenAI system for banking customer support. A Classifier Agent
routes each incoming message to the correct downstream agent:

    User message
        │
        ▼
   ┌──────────────┐   Positive Feedback ─▶ Feedback Handler (LLM thank-you)
   │  Classifier  │─▶ Negative Feedback ─▶ Feedback Handler (new ticket + DB)
   │    Agent     │   Query             ─▶ Query Handler   (look up ticket)
   └──────────────┘

Capstone mapping
----------------
Part 1 — Multi-Agent Design & Execution Logic
  1. Classifier Agent          -> ClassifierAgent
  2. Feedback Handler Agent     -> FeedbackHandlerAgent (positive & negative)
  3. Query Handler Agent        -> QueryHandlerAgent
  4. Agent coordination         -> Orchestrator.process()
Part 2 — LLMOps
  7. Model evaluation           -> evaluate_system()
  8. Streamlit UI               -> banking_support_app.py
  9. Logs & debugging           -> AgentLogger (trace of every interaction)

Design notes
------------
* Every agent works WITH or WITHOUT an OpenAI key. With a key, the Classifier
  and the positive thank-you use the LLM; without one, deterministic fallbacks
  keep the whole pipeline runnable (and testable). The database, ticketing,
  routing, and logging are pure Python and always deterministic.
"""

from __future__ import annotations

import os
import re
import json
import sqlite3
import random
import string
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

DB_PATH_DEFAULT = "support.db"
STATUSES = ["Unresolved", "In Progress", "Resolved"]
LABELS = ["Positive Feedback", "Negative Feedback", "Query"]


# ---------------------------------------------------------------------------
# DATABASE LAYER  (support_tickets)
# ---------------------------------------------------------------------------
class SupportDB:
    """SQLite wrapper for the support_tickets table."""

    def __init__(self, path: str = DB_PATH_DEFAULT):
        self.path = path
        self._init()

    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS support_tickets (
                    ticket_number TEXT PRIMARY KEY,
                    customer_name TEXT,
                    issue         TEXT,
                    status        TEXT NOT NULL,
                    created_at    TEXT NOT NULL
                )""")

    # --- seeding ---
    def seed(self, force: bool = False):
        """Populate a few example tickets (idempotent)."""
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0]
            if n and not force:
                return
            if force:
                c.execute("DELETE FROM support_tickets")
            rows = [
                ("650932", "Rahul Verma",  "Net banking login not working",       "Resolved",    "2026-07-20T09:12:00Z"),
                ("784521", "Aisha Khan",   "Debit card replacement not received", "In Progress", "2026-07-25T14:03:00Z"),
                ("120945", "John Mathew",  "Incorrect interest charged",          "Unresolved",  "2026-07-28T11:20:00Z"),
                ("330011", "Neha Gupta",   "UPI transaction failed but debited",  "Resolved",    "2026-07-18T16:45:00Z"),
                ("905612", "David Lee",    "Loan statement not generated",        "In Progress", "2026-07-30T10:05:00Z"),
            ]
            c.executemany(
                "INSERT OR REPLACE INTO support_tickets VALUES (?,?,?,?,?)", rows)

    # --- operations ---
    def ticket_exists(self, num: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM support_tickets WHERE ticket_number=?",
                             (num,)).fetchone() is not None

    def new_ticket_number(self) -> str:
        """Unique 6-digit ticket number."""
        while True:
            num = "".join(random.choices(string.digits, k=6))
            if num[0] != "0" and not self.ticket_exists(num):
                return num

    def insert_ticket(self, customer_name: str, issue: str,
                      status: str = "Unresolved") -> str:
        num = self.new_ticket_number()
        with self._conn() as c:
            c.execute("INSERT INTO support_tickets VALUES (?,?,?,?,?)",
                      (num, customer_name or "Customer", issue, status,
                       datetime.now(timezone.utc).isoformat()))
        return num

    def get_ticket(self, num: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM support_tickets WHERE ticket_number=?",
                          (num,)).fetchone()
            return dict(r) if r else None

    def all_tickets(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM support_tickets ORDER BY created_at DESC").fetchall()]


# ---------------------------------------------------------------------------
# LOGGING / DEBUG LAYER
# ---------------------------------------------------------------------------
@dataclass
class LogEntry:
    timestamp: str
    user_input: str
    classification: str
    agent: str
    action: str
    ticket_number: Optional[str]
    response: str
    success: bool


class AgentLogger:
    """In-memory trace of every interaction (Part 2, step 9)."""
    def __init__(self):
        self.entries: List[LogEntry] = []

    def log(self, **kw) -> LogEntry:
        e = LogEntry(timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), **kw)
        self.entries.append(e)
        return e

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self.entries]

    def success_rate(self) -> float:
        if not self.entries:
            return 0.0
        return round(sum(e.success for e in self.entries) / len(self.entries), 3)


# ---------------------------------------------------------------------------
# LLM FACTORY  (optional — OpenAI if a key is present)
# ---------------------------------------------------------------------------
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.3):
    """Return a ChatOpenAI instance, or None if no key (enables fallbacks)."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature)


def _extract_name(message: str) -> Optional[str]:
    """Best-effort customer name extraction (e.g. 'I am John', 'This is Priya')."""
    m = re.search(r"\b(?:i am|i'm|this is|my name is)\s+([A-Z][a-z]+)", message, re.I)
    return m.group(1).capitalize() if m else None


# ---------------------------------------------------------------------------
# 1. CLASSIFIER AGENT
# ---------------------------------------------------------------------------
class ClassifierAgent:
    """Classifies a message into Positive Feedback / Negative Feedback / Query."""

    def __init__(self, llm=None):
        self.llm = llm

    def classify(self, message: str) -> Dict[str, str]:
        label = self._llm_classify(message) if self.llm else None
        if label not in LABELS:
            label = self._heuristic(message)
        return {"label": label}

    def _llm_classify(self, message: str) -> Optional[str]:
        prompt = (
            "You are a routing classifier for a bank's customer-support system. "
            "Classify the customer message into EXACTLY one of these three labels:\n"
            "- Positive Feedback: the customer is thanking, praising, or expressing satisfaction.\n"
            "- Negative Feedback: the customer is complaining or reporting a problem "
            "(something not working, delayed, missing, wrong, or a bad experience).\n"
            "- Query: the customer is ASKING for the status/update of an existing ticket or "
            "request (usually mentions a ticket number). Choose Query ONLY when they are asking "
            "for information, not reporting a problem.\n\n"
            "Examples:\n"
            "\"Thanks for sorting out my net banking login issue.\" -> Positive Feedback\n"
            "\"My debit card replacement still hasn't arrived.\" -> Negative Feedback\n"
            "\"I was charged twice for one transaction.\" -> Negative Feedback\n"
            "\"Could you check the status of ticket 650932?\" -> Query\n"
            "\"What is the update on ticket 784521?\" -> Query\n\n"
            f"Message: \"{message}\"\n"
            "Respond with ONLY the label text (Positive Feedback, Negative Feedback, or Query).")
        try:
            out = self.llm.invoke(prompt).content.strip()
            for lab in LABELS:
                if lab.lower() in out.lower():
                    return lab
        except Exception:
            return None
        return None

    def _heuristic(self, message: str) -> str:
        m = message.lower()
        has_num = bool(re.search(r"\d{4,6}", m))
        # 1) Query: asking about a ticket / status (usually references a number).
        if (has_num and re.search(r"\b(status|ticket|check|track|number|update|mark|progress)\b", m)) \
           or re.search(r"\bstatus of\b|\bticket\s*#?\s*\d", m):
            return "Query"
        # 2) Gratitude/praise dominates -> Positive (even if words like "issue" appear).
        pos = ["thank", "thanks", "appreciate", "great", "excellent", "happy",
               "wonderful", "awesome", "kind", "helpful", "love", "good job",
               "well done", "pleased", "delighted", "brilliant", "fantastic"]
        if any(w in m for w in pos):
            return "Positive Feedback"
        # 3) Complaint/problem language -> Negative.
        neg = ["not ", "n't", "never", "no ", "issue", "problem", "fail", "crash",
               "wrong", "delay", "still", "unable", "can't", "cannot", "error",
               "complaint", "worst", "bad", "disappoint", "frustrat", "angry",
               "poor", "slow", "missing", "charged", "double", "stuck", "frozen",
               "hang", "declined", "blocked", "unauthorized"]
        if any(w in m for w in neg):
            return "Negative Feedback"
        # 4) Default: treat an unclassified support message as feedback to log.
        return "Negative Feedback"


# ---------------------------------------------------------------------------
# 2. FEEDBACK HANDLER AGENT
# ---------------------------------------------------------------------------
class FeedbackHandlerAgent:
    """Handles Positive Feedback (thank-you) and Negative Feedback (new ticket)."""

    def __init__(self, db: SupportDB, llm=None):
        self.db = db
        self.llm = llm

    def handle(self, message: str, label: str,
               customer_name: Optional[str] = None) -> Dict[str, Any]:
        name = customer_name or _extract_name(message) or "Customer"
        if label == "Positive Feedback":
            return self._positive(message, name)
        return self._negative(message, name)

    def _positive(self, message: str, name: str) -> Dict[str, Any]:
        response = None
        if self.llm:
            try:
                prompt = (
                    "You are a warm, professional banking support agent. Write a short, "
                    f"personalized thank-you reply to this positive feedback from {name}. "
                    "One or two sentences, friendly, no ticket needed.\n"
                    f"Feedback: \"{message}\"")
                response = self.llm.invoke(prompt).content.strip()
            except Exception:
                response = None
        if not response:
            response = (f"Thank you for your kind words, {name}! "
                        "We're delighted to assist you.")
        return {"agent": "Feedback Handler (Positive)", "action": "thank_you",
                "ticket_number": None, "response": response, "success": True}

    def _negative(self, message: str, name: str) -> Dict[str, Any]:
        ticket = self.db.insert_ticket(customer_name=name, issue=message.strip(),
                                       status="Unresolved")
        response = (f"We apologize for the inconvenience, {name}. A new ticket "
                    f"#{ticket} has been generated, and our team will follow up shortly.")
        return {"agent": "Feedback Handler (Negative)", "action": "ticket_created",
                "ticket_number": ticket, "response": response, "success": True}


# ---------------------------------------------------------------------------
# 3. QUERY HANDLER AGENT
# ---------------------------------------------------------------------------
class QueryHandlerAgent:
    """Extracts a ticket number and returns its status from the database."""

    def __init__(self, db: SupportDB):
        self.db = db

    def handle(self, message: str) -> Dict[str, Any]:
        m = re.search(r"\b(\d{6})\b", message) or re.search(r"\b(\d{4,6})\b", message)
        if not m:
            return {"agent": "Query Handler", "action": "no_ticket_found",
                    "ticket_number": None,
                    "response": ("Could you please share your 6-digit ticket number "
                                 "so I can check its status?"), "success": False}
        num = m.group(1)
        rec = self.db.get_ticket(num)
        if not rec:
            return {"agent": "Query Handler", "action": "ticket_not_in_db",
                    "ticket_number": num,
                    "response": (f"I couldn't find ticket #{num} in our records. "
                                 "Please double-check the number."), "success": False}
        return {"agent": "Query Handler", "action": "status_returned",
                "ticket_number": num,
                "response": f"Your ticket #{num} is currently marked as: {rec['status']}.",
                "success": True}


# ---------------------------------------------------------------------------
# 4. ORCHESTRATOR  (agent coordination)
# ---------------------------------------------------------------------------
class Orchestrator:
    """Coordinates the agents: classify -> route -> respond -> log."""

    def __init__(self, db: Optional[SupportDB] = None, llm="auto",
                 logger: Optional[AgentLogger] = None):
        self.db = db or SupportDB()
        self.db.seed()
        # Generation LLM (warmer) for thank-you messages; classifier LLM at
        # temperature 0 for deterministic, reliable routing.
        self.llm = get_llm(temperature=0.4) if llm == "auto" else llm
        classifier_llm = get_llm(temperature=0.0) if llm == "auto" else llm
        self.classifier = ClassifierAgent(classifier_llm)
        self.feedback = FeedbackHandlerAgent(self.db, self.llm)
        self.query = QueryHandlerAgent(self.db)
        self.logger = logger or AgentLogger()

    def process(self, message: str,
                customer_name: Optional[str] = None) -> Dict[str, Any]:
        label = self.classifier.classify(message)["label"]
        if label == "Query":
            res = self.query.handle(message)
        else:
            res = self.feedback.handle(message, label, customer_name)
        self.logger.log(user_input=message, classification=label,
                        agent=res["agent"], action=res["action"],
                        ticket_number=res.get("ticket_number"),
                        response=res["response"], success=res["success"])
        return {"input": message, "classification": label, **res}


# ---------------------------------------------------------------------------
# 7. MODEL EVALUATION
# ---------------------------------------------------------------------------
EVAL_CASES = [
    {"message": "Thanks for sorting out my net banking login issue.", "expected": "Positive Feedback"},
    {"message": "I really appreciate how quickly you resolved my problem!", "expected": "Positive Feedback"},
    {"message": "My debit card replacement still hasn't arrived.", "expected": "Negative Feedback"},
    {"message": "I was charged twice for the same transaction, this is frustrating.", "expected": "Negative Feedback"},
    {"message": "Could you check the status of ticket 650932?", "expected": "Query"},
    {"message": "What is the update on ticket number 784521?", "expected": "Query"},
    {"message": "Your service was excellent, thank you so much!", "expected": "Positive Feedback"},
    {"message": "The mobile app keeps crashing when I try to pay.", "expected": "Negative Feedback"},
]


def evaluate_system(orchestrator: Orchestrator,
                    cases: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Classification accuracy, routing success, and (if LLM present) QA scoring."""
    cases = cases or EVAL_CASES
    rows, correct, routed = [], 0, 0
    expected_agent = {
        "Positive Feedback": "Feedback Handler (Positive)",
        "Negative Feedback": "Feedback Handler (Negative)",
        "Query": "Query Handler",
    }
    for c in cases:
        pred = orchestrator.classifier.classify(c["message"])["label"]
        ok = pred == c["expected"]
        correct += ok
        # routing check: does the predicted label reach the right agent family?
        routed += (expected_agent.get(pred, "") == expected_agent.get(c["expected"], "")) and ok
        rows.append({"message": c["message"], "expected": c["expected"],
                     "predicted": pred, "correct": ok})
    n = len(cases)
    return {
        "n": n,
        "classification_accuracy": round(correct / n, 3),
        "routing_success_rate": round(routed / n, 3),
        "n_correct": correct,
        "details": rows,
    }


def qa_score_responses(orchestrator: Orchestrator,
                       samples: Optional[List[str]] = None) -> Dict[str, Any]:
    """Optional LLM QA scoring of response quality (empathy/clarity). Needs a key."""
    samples = samples or [c["message"] for c in EVAL_CASES[:5]]
    llm = orchestrator.llm
    scored = []
    for msg in samples:
        res = orchestrator.process(msg)
        score = None
        if llm:
            try:
                prompt = (
                    "Rate the following banking-support reply from 1 to 5 for empathy and "
                    "clarity (5 = excellent). Reply with ONLY the integer.\n"
                    f"Customer: {msg}\nReply: {res['response']}")
                score = int(re.search(r"[1-5]", llm.invoke(prompt).content).group())
            except Exception:
                score = None
        scored.append({"message": msg, "classification": res["classification"],
                       "response": res["response"], "score": score})
    valid = [s["score"] for s in scored if s["score"] is not None]
    return {"scored": scored,
            "avg_score": round(sum(valid) / len(valid), 2) if valid else None}
