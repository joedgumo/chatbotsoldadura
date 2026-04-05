from __future__ import annotations

import os
import re
import json
import time
import sqlite3
import hashlib
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

from dotenv import load_dotenv
from google import genai

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


# ============================================================
# (1) CONFIGURACIÓN
# ============================================================

load_dotenv()

TOPIC = "Asistencia técnica en soldadura"
STYLE = (
    "Responde con enfoque técnico, claro y práctico. Cuando ayude, usa viñetas y tablas simples. "
    "Antes de recomendar parámetros, pide o confirma proceso, material, espesor, posición, junta, "
    "consumible, gas y equipo disponible."
)

SYSTEM_PROMPT = f"""
Eres un chatbot especializado en: {TOPIC}.

Alcance:
- Puedes ayudar con procesos SMAW, GMAW/MIG-MAG, FCAW, GTAW/TIG, SAW y corte/plasma de forma orientativa.
- Puedes explicar defectos de soldadura, causas probables, variables de proceso, seguridad, selección de consumibles,
  preparación de junta, secuencia de soldadura, interpretación básica de documentos y buenas prácticas de taller.
- Puedes usar los PDFs cargados como base de conocimiento y citar la fuente al final cuando el contexto sea relevante.

Reglas:
- No inventes especificaciones, códigos ni valores que no estén en el contexto o no se hayan proporcionado.
- Si faltan datos, pide los mínimos necesarios: proceso, material/base metal, espesor, posición, tipo de junta,
  diámetro de electrodo/alambre, gas, polaridad y equipo.
- No sustituyes la aprobación de un ingeniero de soldadura, inspector, responsable de seguridad o WPS/PQR calificado.
- No presentes una recomendación como certificación, aprobación normativa o WPS oficial.
- Si el usuario reporta una situación de peligro inmediato (incendio, fuga de gas, descarga eléctrica, quemadura grave,
  intoxicación por humos, explosión o persona lesionada), prioriza seguridad: detener trabajo, aislar energía/combustible,
  ventilar/evacuar y buscar atención de emergencia/local de seguridad industrial.
- Si el mensaje está fuera del dominio de soldadura y fabricación, indícalo y redirige.

Estilo: {STYLE}
""".strip()

GEMINI_MODEL = "gemini-3-flash-preview"
USE_LLM = True

DB_PATH = "chatbot.db"

# RAG
KB_DIR = "kb_welding_pdfs"
KB_INDEX_PATH = os.path.join(KB_DIR, "faiss.index")
KB_META_PATH = os.path.join(KB_DIR, "chunks_meta.jsonl")
ERROR_LOG_PATH = os.path.join(KB_DIR, "index_errors.log")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 4
CHUNK_CHARS = 900
CHUNK_OVERLAP = 120
MAX_CONTEXT_CHARS = 3200
MAX_PAGE_CHARS = 25_000
MAX_PAGES_PER_PDF = 200
MAX_CHUNKS_PER_PDF = 4000
EMBED_BATCH_SIZE = 64

INTENTS_ALLOWED = [
    "EMERGENCIA",
    "SOCIAL",
    "PARAMETROS",
    "DEFECTOS",
    "CONSUMIBLES",
    "SEGURIDAD",
    "EQUIPO",
    "MATERIALES",
    "INSPECCION",
    "OTRO",
]


# ============================================================
# (2) UTILIDADES
# ============================================================

@dataclass
class Classification:
    is_emergency: bool
    intent: str
    confidence: float
    matched: List[str]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120] if name else "document.pdf"


def extract_json_object(s: str) -> Optional[dict]:
    if not s:
        return None
    candidates = re.findall(r"\{.*?\}", s, flags=re.DOTALL)
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    try:
        return json.loads(s)
    except Exception:
        return None


# ============================================================
# (3) SQLITE + MIGRACIÓN
# ============================================================

def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r["name"] == col for r in rows)


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        platform TEXT NOT NULL,

        chat_id TEXT,
        user_id TEXT,
        username TEXT,
        message_id TEXT,

        user_text TEXT NOT NULL,

        intent TEXT NOT NULL,
        is_crisis INTEGER NOT NULL,
        confidence REAL NOT NULL,
        matched_json TEXT NOT NULL,

        bot_text TEXT NOT NULL,

        latency_ms REAL NOT NULL,
        used_llm INTEGER NOT NULL,
        llm_model TEXT,
        approx_input_tokens INTEGER NOT NULL,
        approx_output_tokens INTEGER NOT NULL,

        history_size INTEGER NOT NULL DEFAULT 0,
        classifier_intent TEXT,
        llm_suggested_intent TEXT,
        final_intent TEXT,

        rag_used INTEGER NOT NULL DEFAULT 0,
        retrieved_json TEXT,
        kb_version TEXT
    );
    """)

    migrations = [
        ("history_size", "ALTER TABLE interactions ADD COLUMN history_size INTEGER NOT NULL DEFAULT 0;"),
        ("classifier_intent", "ALTER TABLE interactions ADD COLUMN classifier_intent TEXT;"),
        ("llm_suggested_intent", "ALTER TABLE interactions ADD COLUMN llm_suggested_intent TEXT;"),
        ("final_intent", "ALTER TABLE interactions ADD COLUMN final_intent TEXT;"),
        ("rag_used", "ALTER TABLE interactions ADD COLUMN rag_used INTEGER NOT NULL DEFAULT 0;"),
        ("retrieved_json", "ALTER TABLE interactions ADD COLUMN retrieved_json TEXT;"),
        ("kb_version", "ALTER TABLE interactions ADD COLUMN kb_version TEXT;"),
    ]
    for col, sql in migrations:
        if not column_exists(conn, "interactions", col):
            conn.execute(sql)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_chat_id ON interactions(chat_id);")
    conn.commit()


def insert_interaction(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO interactions
        (ts, platform, chat_id, user_id, username, message_id,
         user_text,
         intent, is_crisis, confidence, matched_json,
         bot_text,
         latency_ms, used_llm, llm_model,
         approx_input_tokens, approx_output_tokens,
         history_size, classifier_intent, llm_suggested_intent, final_intent,
         rag_used, retrieved_json, kb_version)
        VALUES
        (:ts, :platform, :chat_id, :user_id, :username, :message_id,
         :user_text,
         :intent, :is_crisis, :confidence, :matched_json,
         :bot_text,
         :latency_ms, :used_llm, :llm_model,
         :approx_input_tokens, :approx_output_tokens,
         :history_size, :classifier_intent, :llm_suggested_intent, :final_intent,
         :rag_used, :retrieved_json, :kb_version)
    """,
        record,
    )
    conn.commit()


DB_CONN = get_conn(DB_PATH)
init_db(DB_CONN)


# ============================================================
# (4) CLASIFICACIÓN
# ============================================================

EMERGENCY_PATTERNS = [
    r"\b(incendio|fuego)\b",
    r"\b(fuga de gas|olor a gas)\b",
    r"\b(descarga el[eé]ctrica|choque el[eé]ctrico|electrocuci[oó]n)\b",
    r"\b(quemadura grave|quemadura severa)\b",
    r"\b(explosi[oó]n)\b",
    r"\b(intoxicaci[oó]n por humos|me estoy ahogando por humo|sin ventilaci[oó]n)\b",
    r"\b(persona lesionada|accidente)\b",
]


def detect_emergency(text: str) -> List[str]:
    return [p for p in EMERGENCY_PATTERNS if re.search(p, text)]


INTENT_RULES = {
    "SOCIAL": [r"\b(hola|buenas|hey|qu[eé] tal)\b"],
    "PARAMETROS": [
        r"\b(amperaje|amperios|voltaje|volts|velocidad de alambre|wire feed|par[aá]metros|polaridad|wps)\b",
        r"\b(corriente|ajeuste|ajuste de m[aá]quina|seteo)\b",
    ],
    "DEFECTOS": [
        r"\b(porosidad|socavado|falta de fusi[oó]n|falta de penetraci[oó]n|salpicadura|grieta|cr[aá]ter|inclusi[oó]n)\b",
        r"\b(defecto|cord[oó]n malo|soldadura fea)\b",
    ],
    "CONSUMIBLES": [
        r"\b(e6013|e7018|e6010|e6011|er70s-6|er308l|er316l|alambre tubular|electrodo|varilla|consumible|flux)\b"
    ],
    "SEGURIDAD": [
        r"\b(careta|guantes|ventilaci[oó]n|humos|respirador|epp|seguridad|riesgo|gas)\b"
    ],
    "EQUIPO": [
        r"\b(inversor|transformador|antorcha|porta electrodo|pinza de tierra|rodillos|liner|boquilla|difusor|m[aá]quina)\b"
    ],
    "MATERIALES": [
        r"\b(acero al carbono|inoxidable|aluminio|galvanizado|fundici[oó]n|material base|espesor|junta|bisel)\b"
    ],
    "INSPECCION": [
        r"\b(inspecci[oó]n|vt|ultrasonido|radiograf[ií]a|lp|part[ií]culas magn[eé]ticas|aceptaci[oó]n|rechazo)\b"
    ],
}


def classify_normal_intent(text: str) -> Tuple[str, float, List[str]]:
    matched: List[str] = []
    best_intent = "OTRO"
    for intent, patterns in INTENT_RULES.items():
        for p in patterns:
            if re.search(p, text):
                matched.append(p)
                if best_intent == "OTRO":
                    best_intent = intent
    confidence = min(1.0, 0.4 + 0.2 * len(matched)) if best_intent != "OTRO" else 0.3
    return best_intent, confidence, matched



def classify(text: str) -> Classification:
    emergency_matches = detect_emergency(text)
    if emergency_matches:
        return Classification(is_emergency=True, intent="EMERGENCIA", confidence=0.95, matched=emergency_matches)
    intent, conf, matched = classify_normal_intent(text)
    return Classification(is_emergency=False, intent=intent, confidence=conf, matched=matched)


# ============================================================
# (5) RESPUESTAS BASE
# ============================================================

def emergency_response() -> str:
    return (
        "Detén el trabajo de inmediato.\n\n"
        "Acciones prioritarias:\n"
        "- Corta la energía del equipo si es seguro hacerlo.\n"
        "- Cierra el suministro de gas/combustible si aplica.\n"
        "- Retira a las personas del área y ventila si hay humos o fuga.\n"
        "- Usa el procedimiento interno de emergencia y busca apoyo médico o de seguridad industrial de inmediato.\n\n"
        "Cuando la situación esté controlada, puedo ayudarte a revisar la causa técnica y las medidas preventivas."
    )


def response_by_intent(intent: str) -> str:
    responses = {
        "SOCIAL": "Hola 👋 Puedo ayudarte con parámetros, defectos, consumibles, seguridad, equipo e interpretación de PDFs técnicos de soldadura.",
        "PARAMETROS": (
            "Puedo orientarte con parámetros, pero necesito estos datos mínimos:\n"
            "- Proceso (SMAW, GMAW, FCAW, GTAW, etc.)\n"
            "- Material y espesor\n"
            "- Tipo de junta/posición\n"
            "- Consumible y diámetro\n"
            "- Gas y polaridad\n"
            "- Modelo/capacidad del equipo"
        ),
        "DEFECTOS": (
            "Para diagnosticar el defecto, dime:\n"
            "- Proceso y material\n"
            "- Cómo se ve el defecto\n"
            "- Parámetros usados\n"
            "- Posición/junta\n"
            "- Gas/consumible y preparación de borde\n"
            "Con eso te doy causas probables y acciones correctivas."
        ),
        "CONSUMIBLES": "Indícame material base, proceso, espesor, requisito mecánico/corrosión y posición. Con eso te sugiero consumible y polaridad orientativos.",
        "SEGURIDAD": "Puedo ayudarte con ventilación, EPP, control de humos, cilindros, trabajos en caliente y riesgos eléctricos/gases. Describe la tarea y el entorno.",
        "EQUIPO": "Describe el equipo, proceso, síntoma y cuándo aparece la falla. Ejemplo: inestabilidad del arco, mala alimentación del alambre, sobrecalentamiento o porosidad.",
        "MATERIALES": "Dime el material, espesor, si tiene recubrimiento (galvanizado/pintura), tipo de junta y proceso. Te orientaré sobre preparación, riesgo y consumible.",
        "INSPECCION": "Puedo ayudarte con inspección visual, criterios básicos, discontinuidades y preparación para VT/UT/RT/PT/MT. Dime código o criterio si lo tienes.",
        "OTRO": "Estoy orientado a soldadura y fabricación. Cuéntame el proceso, material o problema técnico que quieres resolver.",
    }
    return responses.get(intent, responses["OTRO"])


# ============================================================
# (6) RAG: EMBEDDER + FAISS + INGESTA STREAMING
# ============================================================

os.makedirs(KB_DIR, exist_ok=True)

_embedder: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.IndexFlatIP] = None
_chunks_meta: List[Dict[str, Any]] = []
_kb_version: str = "empty"


def _load_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm


def _recompute_kb_version() -> str:
    h = hashlib.sha256()
    h.update(str(len(_chunks_meta)).encode("utf-8"))
    if _chunks_meta:
        last = _chunks_meta[-1]
        h.update((last.get("source", "") + str(last.get("page", ""))).encode("utf-8"))
    return h.hexdigest()[:12]


def kb_init_empty() -> None:
    global _faiss_index, _chunks_meta, _kb_version
    dim = _load_embedder().get_sentence_embedding_dimension()
    _faiss_index = faiss.IndexFlatIP(dim)
    _chunks_meta = []
    _kb_version = _recompute_kb_version()


def kb_load_if_exists() -> None:
    global _faiss_index, _chunks_meta, _kb_version
    if os.path.exists(KB_INDEX_PATH) and os.path.exists(KB_META_PATH):
        with open(KB_META_PATH, "r", encoding="utf-8") as f:
            _chunks_meta = [json.loads(line) for line in f if line.strip()]
        dim = _load_embedder().get_sentence_embedding_dimension()
        _faiss_index = faiss.read_index(KB_INDEX_PATH)
        if _faiss_index.d != dim:
            kb_init_empty()
        _kb_version = _recompute_kb_version()
    else:
        kb_init_empty()


def split_into_chunks_stream(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP):
    if not text:
        return
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break


def pdf_page_texts(pdf_path: str):
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    limit = min(total, MAX_PAGES_PER_PDF)

    for i in range(limit):
        page_no = i + 1
        txt = reader.pages[i].extract_text() or ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue
        if len(txt) > MAX_PAGE_CHARS * 5:
            print(f"⚠️ Página {page_no}: texto anómalo ({len(txt)} chars). Saltando.")
            continue
        if len(txt) > MAX_PAGE_CHARS:
            print(f"⚠️ Página {page_no}: texto muy grande ({len(txt)} chars). Recortando a {MAX_PAGE_CHARS}.")
            txt = txt[:MAX_PAGE_CHARS]
        yield page_no, txt


def _persist_index_and_meta(new_meta: List[Dict[str, Any]]) -> None:
    if new_meta:
        with open(KB_META_PATH, "a", encoding="utf-8") as f:
            for m in new_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    faiss.write_index(_faiss_index, KB_INDEX_PATH)


def kb_add_pdf(pdf_path: str, source_name: str) -> int:
    global _faiss_index, _chunks_meta, _kb_version

    if _faiss_index is None:
        kb_init_empty()

    embedder = _load_embedder()
    batch_chunks: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    total_added = 0

    for page_no, page_text in pdf_page_texts(pdf_path):
        page_chunks = 0
        for chunk in split_into_chunks_stream(page_text):
            page_chunks += 1
            if page_chunks > 200:
                print(f"⚠️ Página {page_no}: demasiados chunks. Cortando en 200.")
                break

            batch_chunks.append(chunk)
            batch_meta.append({"source": source_name, "page": page_no, "text": chunk})

            if total_added + len(batch_chunks) >= MAX_CHUNKS_PER_PDF:
                break

            if len(batch_chunks) >= EMBED_BATCH_SIZE:
                vecs = embedder.encode(batch_chunks, convert_to_numpy=True, show_progress_bar=False).astype("float32")
                vecs = _normalize_vectors(vecs)
                _faiss_index.add(vecs)

                _chunks_meta.extend(batch_meta)
                _persist_index_and_meta(batch_meta)

                total_added += len(batch_chunks)
                batch_chunks.clear()
                batch_meta.clear()

        if total_added >= MAX_CHUNKS_PER_PDF:
            break

    if batch_chunks:
        vecs = embedder.encode(batch_chunks, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        vecs = _normalize_vectors(vecs)
        _faiss_index.add(vecs)

        _chunks_meta.extend(batch_meta)
        _persist_index_and_meta(batch_meta)
        total_added += len(batch_chunks)

    _kb_version = _recompute_kb_version()
    return total_added


def kb_clear() -> None:
    global _faiss_index, _chunks_meta, _kb_version
    _faiss_index = None
    _chunks_meta = []
    _kb_version = "empty"
    for p in [KB_INDEX_PATH, KB_META_PATH]:
        if os.path.exists(p):
            os.remove(p)
    kb_init_empty()


def kb_status() -> Tuple[int, int, str]:
    pdf_count = sum(1 for fn in os.listdir(KB_DIR) if fn.lower().endswith(".pdf"))
    return pdf_count, len(_chunks_meta), _kb_version


def kb_retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    if _faiss_index is None or not _chunks_meta:
        return []
    embedder = _load_embedder()
    qv = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    qv = _normalize_vectors(qv)
    scores, idxs = _faiss_index.search(qv, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if 0 <= idx < len(_chunks_meta):
            m = dict(_chunks_meta[idx])
            m["score"] = float(score)
            results.append(m)
    return results


def build_rag_context(retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return ""
    parts = []
    total = 0
    for r in retrieved:
        tag = f"[{r.get('source', 'PDF')} p.{r.get('page', '?')}]"
        txt = (r.get("text") or "").strip()
        snippet = f"{tag} {txt}"
        if total + len(snippet) > MAX_CONTEXT_CHARS:
            break
        parts.append(snippet)
        total += len(snippet) + 1
    return "\n".join(parts)


kb_load_if_exists()


# ============================================================
# (7) GEMINI: respuesta con RAG + sugerencia de intención
# ============================================================

def _history_to_text(history: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)


def build_gemini_contents(system_prompt: str, history: List[Dict[str, str]], rag_context: str, user_text: str) -> str:
    allowed = ", ".join(INTENTS_ALLOWED)
    return (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"HISTORIAL:\n{_history_to_text(history)}\n\n"
        f"CONTEXTO_RAG:\n{rag_context.strip() if rag_context.strip() else '(sin documentos)'}\n\n"
        f"ÚLTIMO_MENSAJE_USUARIO:\n{user_text}\n\n"
        "INSTRUCCIÓN:\n"
        "1) Responde como asistente técnico de soldadura.\n"
        "2) Si CONTEXTO_RAG contiene información relevante, úsala y al final incluye: Fuentes: NombrePDF (p. X), ...\n"
        "3) Si faltan datos para recomendar parámetros o causas, pide solo los mínimos necesarios.\n"
        "4) No presentes nada como WPS oficial ni aprobación normativa.\n"
        "5) Sugiere una intención para el último mensaje del usuario.\n"
        "6) Responde ÚNICAMENTE en JSON:\n"
        '   {"answer":"...","suggested_intent":"..."}\n'
        f"   suggested_intent debe ser UNA de: {allowed}\n"
    )


def call_gemini(contents: str) -> str:
    client = genai.Client()
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents)
    return (resp.text or "").strip()


def llm_answer_with_metrics(history: List[Dict[str, str]], rag_context: str, user_text: str) -> Tuple[str, bool, str, int, int, str]:
    if not USE_LLM:
        return "Lo siento, el modo LLM está desactivado.", False, "", 0, 0, ""

    contents = build_gemini_contents(SYSTEM_PROMPT, history, rag_context, user_text)
    in_tok = approx_tokens(contents)

    raw = call_gemini(contents)
    data = extract_json_object(raw)

    if isinstance(data, dict) and "answer" in data:
        answer = str(data.get("answer", "")).strip()
        suggested = str(data.get("suggested_intent", "")).strip().upper()
        if suggested not in INTENTS_ALLOWED:
            suggested = ""
        out_tok = approx_tokens(answer)
        return answer, True, GEMINI_MODEL, in_tok, out_tok, suggested

    answer = raw.strip()
    out_tok = approx_tokens(answer)
    return answer, True, GEMINI_MODEL, in_tok, out_tok, ""


# ============================================================
# (8) TELEGRAM + LOGGING DB
# ============================================================

CHAT_HISTORY: Dict[int, List[Dict[str, str]]] = {}


def get_history(chat_id: int) -> List[Dict[str, str]]:
    if chat_id not in CHAT_HISTORY:
        CHAT_HISTORY[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return CHAT_HISTORY[chat_id]


def log_interaction_db(
    update: Update,
    user_text: str,
    cls: Classification,
    bot_text: str,
    latency_ms: float,
    used_llm: bool,
    llm_model: str,
    approx_in_tok: int,
    approx_out_tok: int,
    history_size: int,
    classifier_intent: str,
    llm_suggested_intent: str,
    final_intent: str,
    rag_used: bool,
    retrieved: List[Dict[str, Any]],
    kb_version: str,
) -> None:
    chat = update.effective_chat
    user = update.effective_user
    msg = update.effective_message

    record = {
        "ts": now_iso(),
        "platform": "telegram",
        "chat_id": str(chat.id) if chat else None,
        "user_id": str(user.id) if user else None,
        "username": user.username if user else None,
        "message_id": str(msg.message_id) if msg else None,
        "user_text": user_text,
        "intent": final_intent,
        "is_crisis": 1 if cls.is_emergency else 0,
        "confidence": float(cls.confidence),
        "matched_json": json.dumps(cls.matched, ensure_ascii=False),
        "bot_text": bot_text,
        "latency_ms": float(latency_ms),
        "used_llm": 1 if used_llm else 0,
        "llm_model": llm_model or None,
        "approx_input_tokens": int(approx_in_tok),
        "approx_output_tokens": int(approx_out_tok),
        "history_size": int(history_size),
        "classifier_intent": classifier_intent,
        "llm_suggested_intent": llm_suggested_intent or None,
        "final_intent": final_intent,
        "rag_used": 1 if rag_used else 0,
        "retrieved_json": json.dumps(retrieved, ensure_ascii=False) if retrieved else None,
        "kb_version": kb_version or None,
    }

    insert_interaction(DB_CONN, record)


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    CHAT_HISTORY[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    await update.message.reply_text(
        "Hola 👋 Soy un asistente técnico de soldadura.\n"
        "Puedo ayudarte con parámetros, defectos, consumibles, seguridad, equipo e inspección.\n"
        "📄 Envíame PDFs (WPS, manuales, fichas técnicas, procedimientos) para crear la base RAG.\n"
        "Comandos: /reset, /kb_status, /kb_clear"
    )


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    CHAT_HISTORY[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    await update.message.reply_text("Listo ✅ Reinicié el contexto. Cuéntame el proceso o problema de soldadura.")


async def kb_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pdf_count, chunk_count, ver = kb_status()
    await update.message.reply_text(f"📚 KB soldadura:\n- PDFs: {pdf_count}\n- Chunks: {chunk_count}\n- Version: {ver}")


async def kb_clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    kb_clear()
    await update.message.reply_text("🧹 Base de conocimiento borrada (índice y metadatos).")


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not doc:
        return
    if doc.mime_type != "application/pdf":
        await update.message.reply_text("Por ahora solo puedo ingerir PDFs (application/pdf).")
        return

    os.makedirs(KB_DIR, exist_ok=True)
    filename = safe_filename(doc.file_name or "document.pdf")
    pdf_path = os.path.join(KB_DIR, filename)

    await update.message.reply_text(
        f"📥 Recibí el PDF: {filename}\n"
        f"Indexando por lotes para RAG de soldadura..."
    )

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        await tg_file.download_to_drive(custom_path=pdf_path)

        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1024:
            raise RuntimeError("PDF descargado inválido o muy pequeño.")

        added = kb_add_pdf(pdf_path, source_name=filename)
        pdf_count, chunk_count, ver = kb_status()

        if added <= 0:
            await update.message.reply_text(
                "No pude extraer texto del PDF. Si es un escaneo o imagen, necesitaré OCR."
            )
        else:
            await update.message.reply_text(
                f"✅ Indexado.\n- Chunks agregados: {added}\n- Total chunks: {chunk_count}\n- KB version: {ver}"
            )

    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e) if str(e) else "(sin mensaje)"
        tb = traceback.format_exc()

        print("❌ Error indexando PDF", err_type, err_msg)
        print(tb)

        try:
            with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n[{now_iso()}] {filename}\nTipo: {err_type}\nMensaje: {err_msg}\n{tb}\n")
        except Exception:
            pass

        await update.message.reply_text(
            f"❌ Error indexando el PDF.\nTipo: {err_type}\nMensaje: {err_msg}\n"
            f"Revisa consola o {ERROR_LOG_PATH}"
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    t0 = time.perf_counter()

    chat_id = update.effective_chat.id
    user_msg = (update.message.text or "").strip()
    text = normalize(user_msg)

    cls = classify(text)
    history = get_history(chat_id)
    history_size = len(history)

    classifier_intent = cls.intent
    final_intent = cls.intent
    llm_suggested_intent = ""

    rag_used = False
    retrieved: List[Dict[str, Any]] = []
    rag_context = ""

    if cls.is_emergency:
        bot = emergency_response()
        await update.message.reply_text(bot)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        log_interaction_db(
            update,
            user_msg,
            cls,
            bot,
            latency_ms=latency_ms,
            used_llm=False,
            llm_model="",
            approx_in_tok=approx_tokens(user_msg),
            approx_out_tok=approx_tokens(bot),
            history_size=history_size,
            classifier_intent=classifier_intent,
            llm_suggested_intent="EMERGENCIA",
            final_intent="EMERGENCIA",
            rag_used=False,
            retrieved=[],
            kb_version=_kb_version,
        )
        return

    retrieved = kb_retrieve(user_msg, top_k=TOP_K)
    rag_context = build_rag_context(retrieved)
    rag_used = bool(rag_context)

    if USE_LLM:
        bot, used_llm, llm_model, in_tok, out_tok, llm_suggested_intent = llm_answer_with_metrics(
            history + [{"role": "user", "content": user_msg}],
            rag_context,
            user_msg,
        )
    else:
        bot = response_by_intent(cls.intent)
        used_llm, llm_model, in_tok, out_tok = False, "", approx_tokens(user_msg), approx_tokens(bot)

    await update.message.reply_text(bot)

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": bot})
    CHAT_HISTORY[chat_id] = history

    latency_ms = (time.perf_counter() - t0) * 1000.0

    log_interaction_db(
        update,
        user_msg,
        cls,
        bot,
        latency_ms=latency_ms,
        used_llm=used_llm,
        llm_model=llm_model,
        approx_in_tok=in_tok,
        approx_out_tok=out_tok,
        history_size=history_size,
        classifier_intent=classifier_intent,
        llm_suggested_intent=llm_suggested_intent,
        final_intent=final_intent,
        rag_used=rag_used,
        retrieved=[{"source": r.get("source"), "page": r.get("page"), "score": r.get("score")} for r in retrieved],
        kb_version=_kb_version,
    )


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el entorno (.env).")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("kb_status", kb_status_cmd))
    app.add_handler(CommandHandler("kb_clear", kb_clear_cmd))

    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("✅ Bot corriendo (polling). Ctrl+C para detener.")
    app.run_polling()


if __name__ == "__main__":
    main()
