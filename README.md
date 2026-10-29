# AI knowledge-base assistant for Telegram

A Telegram bot that answers questions from your own documents: RAG over PostgreSQL + pgvector,
role-based system prompts, document upload into the conversation, and PDF export of long answers.
Built as a working prototype of an assistant for a product team and a sales network; the same approach
later cut the load on a support line by about a quarter.

## What it does

- **Answers from your corpus.** Questions are embedded with `BAAI/bge-m3`, the nearest chunks are pulled
  from PostgreSQL (`pgvector`, `<->` distance) and passed to the LLM together with the chat history.
- **Three roles, three behaviours.** `expert` — general Q&A over the whole corpus; `product` — the same,
  restricted to chunks tagged `source = 'product'`; `marketing` — writes HTML/PDF materials, can work from an
  uploaded file instead of the vector index, and keeps the previously generated HTML in context for edits.
- **Documents in, documents out.** Users drop PDF, Word, Excel or PowerPoint files into the chat; the text
  goes into the context. Long answers come back as a PDF (WeasyPrint).
- **Private by default.** Access is granted through an invite link (`/start <SECRET_INVITE_TOKEN>`); the
  allow-list lives in the `access_control` table. Works in private chats and in groups.
- **Prompts live in the database.** System prompts are rows in `promt_data`, so they change without a deploy.
- **Context management.** Token-aware trimming of history (`tiktoken`), large chunk sets processed in slices,
  `/clear` resets the conversation, switching roles resets it too.

## How it is put together

```
Telegram ──▶ aiogram 3 ──▶ handlers (roles, files, menu)
                               │
                               ├─▶ bge-m3 embeddings ──▶ PostgreSQL + pgvector (article_chunks)
                               │                                   │ top-k chunks
                               └─▶ OpenAI-compatible LLM ◀─────────┘
                                   (DeepSeek by default)
                                           │
                                     answer / HTML ──▶ WeasyPrint ──▶ PDF
```

Everything is in `main.py` on purpose: the bot was built to be read and changed by one person quickly.

## Run it

Requirements: Python 3.10+, PostgreSQL 14+ with the `vector` extension, a Telegram bot token, an API key
for an OpenAI-compatible endpoint (the code points at DeepSeek; change `base_url` in `main.py` for OpenAI).

```bash
git clone https://github.com/VSAVENKOV88/product_ai_bot_v2
cd product_ai_bot_v2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # WeasyPrint needs system libs: see weasyprint.org/docs
cp .env.example .env                       # fill in the values
psql "$DATABASE_URL" -f schema.sql         # tables and the vector extension
python main.py
```

Then fill the corpus: insert rows into `article_chunks` with the chunk text, its date, a `source` tag and a
1024-dimensional `bge-m3` embedding (an ingestion script is not part of this repo), and put at least one
prompt named `expert_promt` into `promt_data`.

Open the bot with the invite link `https://t.me/<your_bot>?start=<SECRET_INVITE_TOKEN>`; every later user
goes through the same link or is added to `access_control` by hand.

## Commands

| command  | what it does                                   |
|----------|------------------------------------------------|
| `/start` | access check; with the invite token — grants it |
| `/role`  | switch between expert / product / marketing    |
| `/mode`  | marketing only: vector search or uploaded file |
| `/clear` | forget the conversation                        |

## Limits worth knowing

- The access list and the vector index are the only state; chat history and uploaded files live in process
  memory and are lost on restart.
- One `main.py`, no tests. Fine for a prototype that one person runs; split it before a second developer joins.
- `aiogram` is pinned to a 3.0 beta; newer 3.x releases changed some imports.

## License

MIT
