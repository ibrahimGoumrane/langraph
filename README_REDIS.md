
## Redis CLI Quick Ops (For This Project)

Use these commands exactly to inspect what your agent stores in Redis.

### 1) Connect to Redis

If no password:

```bash
redis-cli -h localhost -p 6379 -n 0
```

If password enabled:

```bash
redis-cli -h localhost -p 6379 -n 0 -a YOUR_PASSWORD
```

### 2) Check RediSearch indexes exist

```bash
FT._LIST
```

You should see at least these names:
- `checkpoint`
- `checkpoint_write`
- `agent_store`
- `agent_store_vectors`

### 3) Inspect checkpoint memory (RedisSaver)

All checkpoints:

```bash
FT.SEARCH checkpoint "*" LIMIT 0 20
```

Single thread (example `thread-1`):

```bash
FT.SEARCH checkpoint "@thread_id:{thread-1}" LIMIT 0 20
```

List raw keys:

```bash
SCAN 0 MATCH checkpoint:* COUNT 200
```

Read one checkpoint JSON document:

```bash
JSON.GET checkpoint:YOUR_KEY_HERE $
```

Pending writes index:

```bash
FT.SEARCH checkpoint_write "*" LIMIT 0 20
```

### 4) Inspect RAG document store (RedisStore)

List document rows:

```bash
FT.SEARCH agent_store "*" LIMIT 0 20
```

Search only docs namespace prefix:

```bash
FT.SEARCH agent_store "@prefix:docs*" LIMIT 0 20
```

List vector rows:

```bash
FT.SEARCH agent_store_vectors "*" LIMIT 0 20
```

List raw keys:

```bash
SCAN 0 MATCH agent_store:* COUNT 200
SCAN 0 MATCH agent_store_vectors:* COUNT 200
```

Read one store row:

```bash
JSON.GET agent_store:YOUR_KEY_HERE $
```

### 5) Inspect tool cache keys

The tool cache in `agent/cache/main.py` uses plain Redis keys (e.g. `add:{...}`).

```bash
SCAN 0 MATCH add:* COUNT 200
SCAN 0 MATCH multiply:* COUNT 200
SCAN 0 MATCH divide:* COUNT 200
```

Read one cached value:

```bash
GET "add:{\"a\": 5, \"b\": 7}"
```

### 6) Safe cleanup vs destructive cleanup

Delete only tool cache keys (safe):

```bash
EVAL "for _,k in ipairs(redis.call('KEYS','add:*')) do redis.call('DEL',k) end; for _,k in ipairs(redis.call('KEYS','multiply:*')) do redis.call('DEL',k) end; for _,k in ipairs(redis.call('KEYS','divide:*')) do redis.call('DEL',k) end; return 'ok'" 0
```

Delete everything in current DB (destructive):

```bash
FLUSHDB
```

### 7) Troubleshooting

If `FT._LIST` or `FT.SEARCH` returns unknown command, your Redis does not have RediSearch enabled.
Use Redis Stack or enable the RediSearch module.