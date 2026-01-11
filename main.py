#!/usr/bin/env python3
"""
DAG Unified Server — Single Codebase for All VSA Nodes
═══════════════════════════════════════════════════════

ONE file. THREE servers. Config via ENV.

Boot Sequence:
1. Connect to L0 Redis (in-memory cache)
2. Subscribe to schema from Schema Registry (Upstash)
3. Validate local LanceDB against schema
4. If mismatch: BLOCK writes, request backfill
5. If match: Join quorum, accept writes

Environment Variables:
- NODE_ID:          vsa01 | vsa02 | vsa03
- LANCE_DB_PATH:    /data/lancedb
- REDIS_L0_URL:     Local Redis URL for L0 cache
- UPSTASH_URL:      Upstash URL for schema registry
- UPSTASH_TOKEN:    Upstash auth token
- PORT:             8080
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx

try:
    import lancedb
    import pyarrow as pa
    HAS_LANCE = True
except ImportError:
    lancedb = pa = None
    HAS_LANCE = False

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    aioredis = None
    HAS_REDIS = False


# =============================================================================
# VECTOR PACKING UTILITIES
# =============================================================================

def pack_bipolar(vector: List[float]) -> bytes:
    """Pack 10K bipolar vector to 1250 bytes."""
    result = bytearray(1250)
    for i, val in enumerate(vector[:DIMENSION]):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        if val > 0:
            result[byte_idx] |= (1 << bit_idx)
    return bytes(result)


def unpack_bipolar(data: bytes) -> List[float]:
    """Unpack 1250 bytes to 10K bipolar vector."""
    result = []
    for byte in data[:1250]:
        for bit in range(8):
            if len(result) >= DIMENSION:
                break
            val = 1.0 if (byte >> (7 - bit)) & 1 else -1.0
            result.append(val)
    return result


def pack_int4(vector: List[float]) -> bytes:
    """Pack 10K int4 vector to 5000 bytes."""
    result = bytearray(5000)
    for i in range(0, min(len(vector), DIMENSION), 2):
        # Quantize to -8..+7
        v1 = max(-8, min(7, int(vector[i] * 7)))
        v2 = max(-8, min(7, int(vector[i+1] * 7))) if i+1 < len(vector) else 0
        # Pack two nibbles into one byte
        result[i // 2] = ((v1 + 8) << 4) | (v2 + 8)
    return bytes(result)


def unpack_int4(data: bytes) -> List[float]:
    """Unpack 5000 bytes to 10K int4 vector (normalized to -1..+1)."""
    result = []
    for byte in data[:5000]:
        v1 = ((byte >> 4) & 0x0F) - 8
        v2 = (byte & 0x0F) - 8
        result.append(v1 / 7.0)  # Normalize to -1..+1
        result.append(v2 / 7.0)
    return result[:DIMENSION]


def quantize_to_bipolar(vector: List[float]) -> List[float]:
    """Quantize float vector to bipolar."""
    return [1.0 if v >= 0 else -1.0 for v in vector]


def quantize_to_int4(vector: List[float]) -> List[float]:
    """Quantize float vector to 16 levels."""
    return [max(-1.0, min(1.0, round(v * 7) / 7)) for v in vector]


# =============================================================================
# CONFIG
# =============================================================================

NODE_ID = os.getenv("NODE_ID", "vsa01")
LANCE_PATH = os.getenv("LANCE_DB_PATH", "/data/lancedb")
REDIS_L0_URL = os.getenv("REDIS_L0_URL", "redis://localhost:6379")
UPSTASH_URL = os.getenv("UPSTASH_URL", "https://upright-jaybird-27907.upstash.io")
UPSTASH_TOKEN = os.getenv("UPSTASH_TOKEN", "")
PORT = int(os.getenv("PORT", "8080"))

SCHEMA_VERSION = "2.0.0"
DIMENSION = 10_000

# =============================================================================
# MULTI-TABLE SCHEMA (Superposition Types)
# =============================================================================
#
# Three representation levels for 10KD vectors:
#
# 1. BIPOLAR (vec10k_bipolar)
#    - 10,000 × {-1, +1}
#    - 1250 bytes packed (8 dims per byte)
#    - 2^10000 orthogonal states
#    - Use: fast similarity, binary operations
#
# 2. INT4 (vec10k_int4)
#    - 10,000 × {-8..+7}
#    - 5000 bytes packed (2 dims per byte)
#    - 16 gradient levels within bipolar space
#    - Use: learned weights, smooth interpolation
#
# 3. SCHEMA/FLOAT32 (vec10k_schema)
#    - 10,000 × float32
#    - 40KB per vector
#    - Full precision ground truth
#    - Use: basis definitions, golden references, training
#
# Hierarchy: Schema → INT4 → Bipolar (increasing compression)
#

TABLE_DEFINITIONS = {
    "vec10k_bipolar": {
        "description": "Binary superposition: 10K × {-1, +1}",
        "storage": "packed_bits",  # 1250 bytes
        "columns": [
            {"name": "id", "type": "string"},
            {"name": "vector", "type": "binary"},  # Packed bits
            {"name": "meta", "type": "string"},
            {"name": "ts", "type": "string"},
        ],
        "bytes_per_vector": 1250,
    },
    "vec10k_int4": {
        "description": "Quantized superposition: 10K × [-8..+7]",
        "storage": "packed_nibbles",  # 5000 bytes
        "columns": [
            {"name": "id", "type": "string"},
            {"name": "vector", "type": "binary"},  # Packed nibbles
            {"name": "meta", "type": "string"},
            {"name": "ts", "type": "string"},
        ],
        "bytes_per_vector": 5000,
    },
    "vec10k_schema": {
        "description": "Full precision: 10K × float32",
        "storage": "float32",  # 40000 bytes
        "columns": [
            {"name": "id", "type": "string"},
            {"name": "vector", "type": f"list<float32>[{DIMENSION}]"},
            {"name": "meta", "type": "string"},
            {"name": "ts", "type": "string"},
        ],
        "bytes_per_vector": 40000,
    },
}

# Default table for backwards compatibility
TABLE_NAME = "vec10k_schema"

PEER_URLS = {
    "vsa01": "https://dag-vsa01.msgraph.de",
    "vsa02": "https://dag-vsa02.msgraph.de",
    "vsa03": "https://dag-vsa03.msgraph.de",
}


# =============================================================================
# TYPES
# =============================================================================

class NodeState(Enum):
    BOOTING = "booting"
    SYNCING = "syncing"
    READY = "ready"
    BLOCKED = "blocked"
    DEGRADED = "degraded"
    DRAINING = "draining"


@dataclass
class SchemaDefinition:
    version: str
    dimension: int
    tables: Dict[str, Dict]  # Multiple table definitions
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        content = json.dumps({
            "version": self.version,
            "dimension": self.dimension,
            "tables": self.tables
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "SchemaDefinition":
        # Handle legacy single-table format
        if "table_name" in d and "tables" not in d:
            d["tables"] = {d["table_name"]: {"columns": d.get("columns", [])}}
            del d["table_name"]
            if "columns" in d:
                del d["columns"]
        return cls(**{k: v for k, v in d.items() if k in ["version", "dimension", "tables", "checksum"]})


DEFAULT_SCHEMA = SchemaDefinition(
    version=SCHEMA_VERSION,
    dimension=DIMENSION,
    tables=TABLE_DEFINITIONS,
)


class Vec10k(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = {}
    ts: Optional[str] = None


class BatchVectors(BaseModel):
    vectors: List[Vec10k]
    source: str = "unknown"


# =============================================================================
# GLOBAL STATE
# =============================================================================

class ServerState:
    def __init__(self):
        self.node_state: NodeState = NodeState.BOOTING
        self.schema: SchemaDefinition = DEFAULT_SCHEMA
        self.db = None
        self.tables: Dict[str, Any] = {}  # Multiple tables
        self.redis_l0 = None
        self.vector_counts: Dict[str, int] = {}  # Count per table
        self.peers_online: List[str] = []
        self.last_heartbeat: str = ""
    
    @property
    def vector_count(self) -> int:
        """Total vectors across all tables."""
        return sum(self.vector_counts.values())
    
    @property
    def table(self):
        """Default table for backwards compatibility."""
        return self.tables.get(TABLE_NAME)


state = ServerState()


# =============================================================================
# REDIS L0
# =============================================================================

async def connect_redis_l0() -> bool:
    global state
    if not HAS_REDIS:
        return False
    try:
        state.redis_l0 = await aioredis.from_url(REDIS_L0_URL, decode_responses=True)
        await state.redis_l0.ping()
        return True
    except:
        state.redis_l0 = None
        return False


async def l0_get(key: str) -> Optional[str]:
    if not state.redis_l0:
        return None
    try:
        return await state.redis_l0.get(f"dag:{key}")
    except:
        return None


async def l0_set(key: str, value: str, ttl: int = 3600) -> bool:
    if not state.redis_l0:
        return False
    try:
        await state.redis_l0.set(f"dag:{key}", value, ex=ttl)
        return True
    except:
        return False


async def l0_publish(channel: str, message: str) -> bool:
    if not state.redis_l0:
        return False
    try:
        await state.redis_l0.publish(f"dag:{channel}", message)
        return True
    except:
        return False


# =============================================================================
# UPSTASH
# =============================================================================

async def upstash_cmd(cmd: List) -> Any:
    if not UPSTASH_TOKEN:
        return None
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                UPSTASH_URL,
                headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
                json=cmd,
                timeout=10.0
            )
            return resp.json().get("result")
    except:
        return None


async def fetch_schema_from_registry() -> Optional[SchemaDefinition]:
    data = await upstash_cmd(["GET", "dag:schema:definition"])
    if data:
        try:
            return SchemaDefinition.from_dict(json.loads(data))
        except:
            pass
    return None


async def register_schema(schema: SchemaDefinition) -> bool:
    existing = await upstash_cmd(["GET", "dag:schema:definition"])
    if existing:
        return True
    await upstash_cmd(["SET", "dag:schema:definition", json.dumps(schema.to_dict())])
    return True


async def join_quorum() -> bool:
    await upstash_cmd(["SADD", "dag:quorum:nodes", NODE_ID])
    await upstash_cmd(["HSET", f"dag:node:{NODE_ID}",
        "last_seen", datetime.now(timezone.utc).isoformat(),
        "state", state.node_state.value,
        "vectors", str(state.vector_count)])
    return True


async def leave_quorum() -> bool:
    await upstash_cmd(["SREM", "dag:quorum:nodes", NODE_ID])
    return True


async def get_quorum_nodes() -> List[str]:
    nodes = await upstash_cmd(["SMEMBERS", "dag:quorum:nodes"])
    return nodes or []


# =============================================================================
# LANCEDB
# =============================================================================

def init_lancedb(schema: SchemaDefinition) -> Tuple[bool, str]:
    """Initialize LanceDB with all table types."""
    global state
    if not HAS_LANCE:
        return False, "no_lance"
    
    try:
        os.makedirs(LANCE_PATH, exist_ok=True)
        state.db = lancedb.connect(LANCE_PATH)
        
        # Create each table type
        for table_name, table_def in schema.tables.items():
            try:
                if table_name == "vec10k_schema":
                    # Float32 vectors
                    pa_schema = pa.schema([
                        pa.field("id", pa.string()),
                        pa.field("vector", pa.list_(pa.float32(), schema.dimension)),
                        pa.field("meta", pa.string()),
                        pa.field("ts", pa.string()),
                    ])
                elif table_name == "vec10k_bipolar":
                    # Packed binary (1250 bytes)
                    pa_schema = pa.schema([
                        pa.field("id", pa.string()),
                        pa.field("vector", pa.binary()),  # Packed bits
                        pa.field("meta", pa.string()),
                        pa.field("ts", pa.string()),
                    ])
                elif table_name == "vec10k_int4":
                    # Packed nibbles (5000 bytes)
                    pa_schema = pa.schema([
                        pa.field("id", pa.string()),
                        pa.field("vector", pa.binary()),  # Packed nibbles
                        pa.field("meta", pa.string()),
                        pa.field("ts", pa.string()),
                    ])
                else:
                    continue
                
                if table_name not in state.db.table_names():
                    state.tables[table_name] = state.db.create_table(table_name, schema=pa_schema)
                    state.vector_counts[table_name] = 0
                else:
                    state.tables[table_name] = state.db.open_table(table_name)
                    state.vector_counts[table_name] = state.tables[table_name].count_rows()
                
                print(f"[{NODE_ID}] Table {table_name}: {state.vector_counts.get(table_name, 0)} vectors")
            
            except Exception as e:
                print(f"[{NODE_ID}] Warning: Failed to init {table_name}: {e}")
        
        return True, "ok"
    except Exception as e:
        return False, str(e)


# =============================================================================
# DRAIN
# =============================================================================

async def drain_to_peers(target_nodes: List[str] = None) -> Dict:
    global state
    if not state.table:
        return {"error": "No table"}
    
    peers = target_nodes or [n for n in state.peers_online if n != NODE_ID]
    if not peers:
        return {"error": "No peers"}
    
    state.node_state = NodeState.DRAINING
    results = {"node": NODE_ID, "pushed": {}, "errors": {}}
    
    df = state.table.to_pandas()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for peer in peers:
            peer_url = PEER_URLS.get(peer, peer)
            pushed, errors = 0, 0
            
            for _, row in df.iterrows():
                try:
                    payload = {
                        "id": row["id"],
                        "vector": row["vector"].tolist() if hasattr(row["vector"], "tolist") else list(row["vector"]),
                        "metadata": json.loads(row["meta"]) if row["meta"] else {},
                        "ts": row["ts"]
                    }
                    resp = await client.post(f"{peer_url}/vectors/upsert", json=payload)
                    if resp.status_code == 200:
                        pushed += 1
                    else:
                        errors += 1
                except:
                    errors += 1
            
            results["pushed"][peer] = pushed
            results["errors"][peer] = errors
    
    return results


# =============================================================================
# SUBSCRIPTION
# =============================================================================

async def subscribe_to_updates():
    global state
    if not state.redis_l0:
        return
    
    try:
        pubsub = state.redis_l0.pubsub()
        await pubsub.subscribe("dag:vectors:updated")
        
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                data = json.loads(message["data"])
                source_node = data.get("node")
                vector_id = data.get("id")
                if source_node != NODE_ID and source_node in PEER_URLS:
                    await pull_single_vector(source_node, vector_id)
            except:
                pass
    except asyncio.CancelledError:
        pass


async def pull_single_vector(source_node: str, vector_id: str) -> bool:
    if not state.table:
        return False
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{PEER_URLS[source_node]}/vectors/get/{vector_id}")
            if resp.status_code != 200:
                return False
            
            vec_data = resp.json()
            existing = state.table.search().where(f"id = '{vector_id}'").limit(1).to_list()
            
            if existing:
                if vec_data.get("ts", "") <= existing[0].get("ts", ""):
                    return False
                state.table.delete(f"id = '{vector_id}'")
            
            state.table.add([{
                "id": vec_data["id"],
                "vector": vec_data["vector"],
                "meta": json.dumps(vec_data.get("metadata", {})),
                "ts": vec_data["ts"]
            }])
            return True
    except:
        return False


# =============================================================================
# BOOT SEQUENCE
# =============================================================================

async def boot_sequence():
    global state
    print(f"[{NODE_ID}] BOOT START")
    state.node_state = NodeState.BOOTING
    
    l0_ok = await connect_redis_l0()
    if not l0_ok:
        state.node_state = NodeState.DEGRADED
    
    registry_schema = await fetch_schema_from_registry()
    state.schema = registry_schema or DEFAULT_SCHEMA
    if not registry_schema:
        await register_schema(state.schema)
    
    lance_ok, status = init_lancedb(state.schema)
    if not lance_ok:
        state.node_state = NodeState.BLOCKED
        return
    
    await join_quorum()
    state.peers_online = await get_quorum_nodes()
    
    if state.node_state != NodeState.DEGRADED:
        state.node_state = NodeState.READY
    
    print(f"[{NODE_ID}] BOOT COMPLETE: {state.node_state.value}, {state.vector_count} vectors")


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def heartbeat_loop():
    while True:
        await asyncio.sleep(30)
        if state.node_state in [NodeState.READY, NodeState.DEGRADED]:
            await join_quorum()
            state.peers_online = await get_quorum_nodes()
            # Update counts for all tables
            for table_name, tbl in state.tables.items():
                try:
                    state.vector_counts[table_name] = tbl.count_rows()
                except:
                    pass


async def subscription_loop():
    while True:
        if state.node_state in [NodeState.READY, NodeState.DEGRADED]:
            await subscribe_to_updates()
        await asyncio.sleep(5)


# =============================================================================
# APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await boot_sequence()
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    subscription_task = asyncio.create_task(subscription_loop())
    yield
    heartbeat_task.cancel()
    subscription_task.cancel()
    await leave_quorum()
    if state.redis_l0:
        await state.redis_l0.close()


app = FastAPI(title=f"DAG {NODE_ID}", lifespan=lifespan)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    return {
        "node": NODE_ID,
        "state": state.node_state.value,
        "schema_version": state.schema.version,
        "tables": {
            name: {
                "vectors": state.vector_counts.get(name, 0),
                "bytes_per_vector": TABLE_DEFINITIONS.get(name, {}).get("bytes_per_vector", 0)
            }
            for name in state.tables.keys()
        },
        "total_vectors": state.vector_count,
        "l0_redis": state.redis_l0 is not None,
        "quorum": state.peers_online,
        "ok": state.node_state in [NodeState.READY, NodeState.DEGRADED]
    }


@app.get("/schema")
async def get_schema():
    return state.schema.to_dict()


@app.get("/tables")
async def list_tables():
    """List all table types with their stats."""
    return {
        "node": NODE_ID,
        "tables": {
            name: {
                "description": TABLE_DEFINITIONS.get(name, {}).get("description", ""),
                "storage": TABLE_DEFINITIONS.get(name, {}).get("storage", ""),
                "bytes_per_vector": TABLE_DEFINITIONS.get(name, {}).get("bytes_per_vector", 0),
                "vectors": state.vector_counts.get(name, 0),
                "exists": name in state.tables
            }
            for name in TABLE_DEFINITIONS.keys()
        }
    }


@app.get("/vectors/count")
async def count(table: str = None):
    if table:
        return {"count": state.vector_counts.get(table, 0), "table": table, "node": NODE_ID}
    return {"count": state.vector_count, "tables": state.vector_counts, "node": NODE_ID}


@app.get("/vectors/list")
async def list_vectors(table: str = TABLE_NAME, limit: int = 10000, offset: int = 0):
    if state.node_state == NodeState.BLOCKED:
        raise HTTPException(503, "Node blocked")
    
    tbl = state.tables.get(table)
    if not tbl:
        raise HTTPException(404, f"Table {table} not found")
    
    df = tbl.to_pandas()
    df = df.sort_values("ts", ascending=False).iloc[offset:offset + limit]
    
    summaries = []
    for _, row in df.iterrows():
        meta = json.loads(row["meta"]) if row["meta"] else {}
        summaries.append({"id": row["id"], "ts": row["ts"], "meta_keys": list(meta.keys())})
    
    return {"node": NODE_ID, "table": table, "total": state.vector_counts.get(table, 0), 
            "count": len(summaries), "vectors": summaries}


@app.get("/vectors/get/{vector_id}")
async def get_vector(vector_id: str, table: str = TABLE_NAME, unpack: bool = True):
    tbl = state.tables.get(table)
    if not tbl:
        raise HTTPException(404, f"Table {table} not found")
    
    # Check L0 cache
    cached = await l0_get(f"vec:{table}:{vector_id}")
    if cached:
        return json.loads(cached)
    
    results = tbl.search().where(f"id = '{vector_id}'").limit(1).to_list()
    if not results:
        raise HTTPException(404, "Not found")
    
    row = results[0]
    vector_data = row["vector"]
    
    # Unpack if binary
    if unpack and table == "vec10k_bipolar" and isinstance(vector_data, bytes):
        vector_data = unpack_bipolar(vector_data)
    elif unpack and table == "vec10k_int4" and isinstance(vector_data, bytes):
        vector_data = unpack_int4(vector_data)
    elif isinstance(vector_data, bytes):
        import base64
        vector_data = base64.b64encode(vector_data).decode()
    
    result = {
        "id": row["id"],
        "vector": vector_data,
        "metadata": json.loads(row["meta"]) if row["meta"] else {},
        "ts": row["ts"],
        "table": table,
        "node": NODE_ID
    }
    await l0_set(f"vec:{table}:{vector_id}", json.dumps(result))
    return result


@app.post("/vectors/upsert")
async def upsert(req: Vec10k, table: str = TABLE_NAME, auto_pack: bool = True):
    """
    Upsert a vector to the specified table.
    
    If auto_pack=True and table is bipolar/int4, the float vector will be automatically packed.
    """
    if state.node_state in [NodeState.BLOCKED, NodeState.DRAINING]:
        raise HTTPException(503, f"Node {state.node_state.value}")
    
    tbl = state.tables.get(table)
    if not tbl:
        raise HTTPException(404, f"Table {table} not found")
    
    ts = req.ts or datetime.now(timezone.utc).isoformat()
    
    # Prepare vector based on table type
    if table == "vec10k_bipolar":
        if auto_pack and isinstance(req.vector[0], float):
            vector_data = pack_bipolar(req.vector)
        else:
            vector_data = bytes(req.vector) if not isinstance(req.vector, bytes) else req.vector
    elif table == "vec10k_int4":
        if auto_pack and isinstance(req.vector[0], float):
            vector_data = pack_int4(req.vector)
        else:
            vector_data = bytes(req.vector) if not isinstance(req.vector, bytes) else req.vector
    else:
        # Float32 schema table
        if len(req.vector) != state.schema.dimension:
            raise HTTPException(400, f"Expected {state.schema.dimension}D")
        vector_data = req.vector
    
    # Check for existing
    existing = tbl.search().where(f"id = '{req.id}'").limit(1).to_list()
    if existing:
        tbl.delete(f"id = '{req.id}'")
    
    tbl.add([{"id": req.id, "vector": vector_data, "meta": json.dumps(req.metadata), "ts": ts}])
    state.vector_counts[table] = tbl.count_rows()
    
    # Cache
    await l0_set(f"vec:{table}:{req.id}", json.dumps({
        "id": req.id, "vector": req.vector, "metadata": req.metadata, "ts": ts
    }))
    await l0_publish("vectors:updated", json.dumps({
        "id": req.id, "ts": ts, "node": NODE_ID, "table": table
    }))
    
    return {"ok": True, "id": req.id, "table": table, "node": NODE_ID, 
            "action": "update" if existing else "insert"}


@app.post("/vectors/upsert_all")
async def upsert_all_tables(req: Vec10k):
    """
    Upsert a float32 vector to ALL table types.
    
    Automatically quantizes and packs for bipolar and int4 tables.
    Useful for populating all representations at once.
    """
    if state.node_state in [NodeState.BLOCKED, NodeState.DRAINING]:
        raise HTTPException(503, f"Node {state.node_state.value}")
    
    if len(req.vector) != state.schema.dimension:
        raise HTTPException(400, f"Expected {state.schema.dimension}D")
    
    ts = req.ts or datetime.now(timezone.utc).isoformat()
    results = {}
    
    for table_name in ["vec10k_schema", "vec10k_bipolar", "vec10k_int4"]:
        tbl = state.tables.get(table_name)
        if not tbl:
            results[table_name] = {"ok": False, "error": "not_found"}
            continue
        
        try:
            # Prepare vector
            if table_name == "vec10k_bipolar":
                vector_data = pack_bipolar(req.vector)
            elif table_name == "vec10k_int4":
                vector_data = pack_int4(req.vector)
            else:
                vector_data = req.vector
            
            # Delete existing
            existing = tbl.search().where(f"id = '{req.id}'").limit(1).to_list()
            if existing:
                tbl.delete(f"id = '{req.id}'")
            
            tbl.add([{"id": req.id, "vector": vector_data, "meta": json.dumps(req.metadata), "ts": ts}])
            state.vector_counts[table_name] = tbl.count_rows()
            results[table_name] = {"ok": True, "action": "update" if existing else "insert"}
        except Exception as e:
            results[table_name] = {"ok": False, "error": str(e)}
    
    return {"id": req.id, "node": NODE_ID, "tables": results}


@app.post("/vectors/batch")
async def batch_upsert(req: BatchVectors, table: str = TABLE_NAME, auto_pack: bool = True):
    if state.node_state in [NodeState.BLOCKED, NodeState.DRAINING]:
        raise HTTPException(503, f"Node {state.node_state.value}")
    
    tbl = state.tables.get(table)
    if not tbl:
        raise HTTPException(404, f"Table {table} not found")
    
    results = {"inserted": 0, "updated": 0, "errors": 0, "table": table}
    
    for vec in req.vectors:
        try:
            ts = vec.ts or datetime.now(timezone.utc).isoformat()
            
            # Prepare vector based on table type
            if table == "vec10k_bipolar" and auto_pack:
                vector_data = pack_bipolar(vec.vector)
            elif table == "vec10k_int4" and auto_pack:
                vector_data = pack_int4(vec.vector)
            else:
                if len(vec.vector) != state.schema.dimension:
                    results["errors"] += 1
                    continue
                vector_data = vec.vector
            
            existing = tbl.search().where(f"id = '{vec.id}'").limit(1).to_list()
            if existing:
                tbl.delete(f"id = '{vec.id}'")
                results["updated"] += 1
            else:
                results["inserted"] += 1
            
            tbl.add([{"id": vec.id, "vector": vector_data, "meta": json.dumps(vec.metadata), "ts": ts}])
        except:
            results["errors"] += 1
    
    state.vector_counts[table] = tbl.count_rows()
    return results


@app.get("/vectors/diff/{peer_node}")
async def diff_with_peer(peer_node: str):
    if not state.table:
        raise HTTPException(503, "No table")
    
    peer_url = PEER_URLS.get(peer_node, peer_node)
    local_df = state.table.to_pandas()
    local_vectors = {row["id"]: row["ts"] for _, row in local_df.iterrows()}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{peer_url}/vectors/list?limit=100000")
            peer_data = resp.json()
    except Exception as e:
        raise HTTPException(502, f"Peer unreachable: {e}")
    
    peer_vectors = {v["id"]: v["ts"] for v in peer_data.get("vectors", [])}
    
    only_local = set(local_vectors.keys()) - set(peer_vectors.keys())
    only_peer = set(peer_vectors.keys()) - set(local_vectors.keys())
    conflicts = [{"id": vid, "local_ts": local_vectors[vid], "peer_ts": peer_vectors[vid]}
                 for vid in set(local_vectors.keys()) & set(peer_vectors.keys())
                 if local_vectors[vid] != peer_vectors[vid]]
    
    return {"node": NODE_ID, "peer": peer_node, "only_local": list(only_local), "only_peer": list(only_peer), "conflicts": conflicts}


@app.post("/sync/from/{peer_node}")
async def sync_from_peer(peer_node: str, dry_run: bool = False):
    diff = await diff_with_peer(peer_node)
    peer_url = PEER_URLS.get(peer_node, peer_node)
    
    results = {"node": NODE_ID, "peer": peer_node, "dry_run": dry_run,
               "to_pull": len(diff["only_peer"]), "pulled": 0, "errors": 0}
    
    if dry_run:
        return results
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for vid in diff["only_peer"]:
            try:
                resp = await client.get(f"{peer_url}/vectors/get/{vid}")
                vec_data = resp.json()
                state.table.add([{"id": vec_data["id"], "vector": vec_data["vector"],
                                  "meta": json.dumps(vec_data.get("metadata", {})), "ts": vec_data["ts"]}])
                results["pulled"] += 1
            except:
                results["errors"] += 1
    
    state.vector_count = state.table.count_rows()
    return results


@app.post("/drain")
async def drain(target_nodes: List[str] = None):
    if state.node_state == NodeState.DRAINING:
        raise HTTPException(409, "Already draining")
    return await drain_to_peers(target_nodes)


@app.post("/drain/prepare")
async def prepare_maintenance():
    global state
    state.node_state = NodeState.DRAINING
    drain_result = await drain_to_peers()
    await leave_quorum()
    return {"node": NODE_ID, "status": "ready_for_maintenance", "drain_result": drain_result}


@app.get("/hydrate/status")
async def hydration_status():
    golden = await upstash_cmd(["KEYS", "ada:golden:*"]) or []
    qualia = await upstash_cmd(["KEYS", "ada:qualia:dto:*"]) or []
    return {
        "node": NODE_ID,
        "local_vectors": state.vector_count,
        "upstash_golden": len(golden) if isinstance(golden, list) else 0,
        "upstash_qualia": len(qualia) if isinstance(qualia, list) else 0,
        "needs_hydration": state.vector_count == 0,
        "hydrate_command": f"python dag/hydrate_from_upstash.py --target {NODE_ID}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
