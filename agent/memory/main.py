import os

from dotenv import load_dotenv
from langgraph.checkpoint.redis import RedisSaver

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

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
