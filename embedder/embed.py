import os
import psycopg2
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("MODEL_NAME", "google/embeddinggemma-300m")
DATABASE_URL = os.environ["DATABASE_URL"]
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MAX_TEXT_CHARS = 3000


def build_doc_text(name, nameen, objective, syllabus, schedule):
    title = " / ".join(filter(None, [name, nameen]))
    body = " ".join(filter(None, [objective, syllabus, schedule]))
    return (f"title: {title} | text: {body}")[:MAX_TEXT_CHARS]


def vec_to_pg(embedding):
    return "[" + ",".join(map(str, embedding.tolist())) + "]"


def create_vector_index(cur, conn):
    print("[embedder] Creating HNSW cosine index on embedding column...", flush=True)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS course_embedding_hnsw_idx
        ON public.course
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    conn.commit()
    print("[embedder] Vector index created.", flush=True)


def main():
    print(f"[embedder] Loading model {MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"torch_dtype": torch.float32},
    )
    EMBEDDING_DIM = model.get_sentence_embedding_dimension()
    print(f"[embedder] Model loaded. Embedding dimension: {EMBEDDING_DIM}", flush=True)

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) FROM public.course WHERE y = '114' AND s = '2' AND embedding IS NULL"
    )
    total = cur.fetchone()[0]
    print(f"[embedder] 1142 courses without embedding: {total}", flush=True)

    if total == 0:
        print("[embedder] Nothing to do — all courses already embedded.", flush=True)
        create_vector_index(cur, conn)
        cur.close()
        conn.close()
        return

    processed = 0

    while True:
        cur.execute("""
            SELECT id, name, nameen, objective, syllabus, schedule
            FROM public.course
            WHERE y = '114' AND s = '2' AND embedding IS NULL
            LIMIT %s
        """, (BATCH_SIZE,))
        rows = cur.fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [
            build_doc_text(r[1] or "", r[2] or "", r[3] or "", r[4] or "", r[5] or "")
            for r in rows
        ]

        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

        skipped = 0
        for course_id, emb in zip(ids, embeddings):
            if torch.isnan(torch.tensor(emb)).any():
                print(f"[embedder] WARNING: NaN for id={course_id}, storing zero vector", flush=True)
                emb = [0.0] * EMBEDDING_DIM
                skipped += 1
            cur.execute(
                "UPDATE public.course SET embedding = %s::vector WHERE id = %s",
                (vec_to_pg(emb), course_id),
            )

        conn.commit()
        processed += len(rows)
        pct = 100 * processed // total
        msg = f"[embedder] {processed}/{total} ({pct}%)"
        if skipped:
            msg += f"  [{skipped} skipped NaN]"
        print(msg, flush=True)

    print("[embedder] All courses embedded!", flush=True)
    create_vector_index(cur, conn)
    cur.close()
    conn.close()
    print("[embedder] Done.", flush=True)


if __name__ == "__main__":
    main()
