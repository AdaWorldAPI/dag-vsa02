"""
VSA02 — Warm Replica (vsa02.msgraph.de)
=======================================
4-node LanceDB DAG. 30s lag from AGI. Cascades to vsa03.
"""
import os, json, asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import lancedb
    import pyarrow as pa
    HAS_LANCE = True
except:
    lancedb = pa = None
    HAS_LANCE = False

LANCE_PATH = os.getenv("LANCE_DB_PATH", "/data/lancedb")
NODE_ID = "vsa02"
VSA03 = "https://vsa03.msgraph.de"
db = None

class Vec10k(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = {}
    cascade: bool = True

def init_db():
    global db
    if not HAS_LANCE: return False
    os.makedirs(LANCE_PATH, exist_ok=True)
    db = lancedb.connect(LANCE_PATH)
    if "vec10k" not in db.table_names():
        db.create_table("vec10k", schema=pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 10000)),
            pa.field("meta", pa.string()),
            pa.field("ts", pa.string()),
        ]))
    return True

async def cascade(id: str, vec: List[float], meta: Dict):
    await asyncio.sleep(60)  # 60s to vsa03
    try:
        async with httpx.AsyncClient() as c:
            await c.post(f"{VSA03}/vectors/upsert", json={"id": id, "vector": vec, "metadata": meta, "cascade": False}, timeout=30)
    except: pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title=f"DAG {NODE_ID}", lifespan=lifespan)

@app.get("/health")
async def health():
    n = db.open_table("vec10k").count_rows() if db and "vec10k" in db.table_names() else 0
    return {"node": NODE_ID, "role": "warm", "lag_ms": 30000, "cascade_to": "vsa03", "vectors": n, "ok": True}

@app.post("/vectors/upsert")
async def upsert(req: Vec10k):
    if not db: raise HTTPException(503, "no db")
    if len(req.vector) != 10000: raise HTTPException(400, f"need 10kD")
    db.open_table("vec10k").add([{"id": req.id, "vector": req.vector, "meta": json.dumps(req.metadata), "ts": datetime.now(timezone.utc).isoformat()}])
    if req.cascade: asyncio.create_task(cascade(req.id, req.vector, req.metadata))
    return {"ok": True, "id": req.id, "node": NODE_ID, "cascading": req.cascade}

@app.get("/vectors/count")
async def count():
    return {"count": db.open_table("vec10k").count_rows() if db else 0, "node": NODE_ID}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
