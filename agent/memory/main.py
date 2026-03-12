import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from redis import Redis

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "384"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

if REDIS_PASSWORD:
	redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
	redis_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# ttl uses minutes; refresh_on_read keeps active threads alive.
memory = RedisSaver(
	redis_url=redis_url,
	ttl={"default_ttl": 60 * 24 * 30, "refresh_on_read": True},
)

# Required: create RediSearch indexes used by checkpoint queries.
memory.setup()


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
store = RedisStore(
	Redis.from_url(redis_url),
	index={
		"dims": EMBEDDING_DIMS,
		"embed": embeddings,
		"fields": ["content"],
	},
	store_prefix="agent_store",
	vector_prefix="agent_store_vectors",
)
store.setup()


def _decode_index_name(value: object) -> str:
	if isinstance(value, bytes):
		return value.decode("utf-8", errors="ignore")
	return str(value)


def ensure_store_ready() -> None:
	"""Ensure Redis vector indices exist before put/search operations."""

	try:
		index_names = {
			_decode_index_name(name)
			for name in store._redis.execute_command("FT._LIST")
		}
	except Exception as exc:
		raise RuntimeError(
			"RediSearch is not available. Use Redis Stack or enable RediSearch module."
		) from exc

	required = {"agent_store", "agent_store_vectors"}
	if not required.issubset(index_names):
		store.setup()


def store_conversation_turn(
	thread_id: str,
	user_input: str,
	assistant_output: str,
) -> str:
	"""Store one conversation turn as an embedded item in the vector store."""

	ensure_store_ready()
	namespace = ("conversations", thread_id)
	key = f"turn-{uuid4().hex[:12]}"
	combined = f"User: {user_input}\nAssistant: {assistant_output}"

	store.put(
		namespace,
		key,
		{
			"content": combined,
			"user_input": user_input,
			"assistant_output": assistant_output,
		},
		index=["content"],
	)

	return key


def retrieve_conversation_chunks(
	thread_id: str,
	query: str,
	k: int = DEFAULT_TOP_K,
) -> list[str]:
	"""Return top-k relevant prior conversation chunks for the given thread."""

	ensure_store_ready()

	if not query.strip():
		return []

	results = store.search(("conversations", thread_id), query=query, limit=k)
	context_chunks: list[str] = []

	for item in results:
		value = getattr(item, "value", {}) or {}
		content = value.get("content")
		if isinstance(content, str) and content.strip():
			context_chunks.append(content.strip())

	return context_chunks


def build_retrieval_context(
	thread_id: str,
	query: str,
	k: int = DEFAULT_TOP_K,
) -> str:
	"""Format top-k retrieved conversation chunks into a compact context block."""

	chunks = retrieve_conversation_chunks(thread_id=thread_id, query=query, k=k)
	if not chunks:
		return ""

	lines = [f"[{idx}] {chunk}" for idx, chunk in enumerate(chunks, start=1)]
	return "\n".join(lines)