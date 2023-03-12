
# Start Services

- [x] 💀 `docker` / `docker compose` must be installed on the system
- [x] 💀 nvidia gpu must be present


```bash
make up
```

# Stop services

```bash
make down
```

# Flow Chart of the system




# ⚡Triton server PORT details

|**Service** | **Port** |
|:------|:-------|
|GRPC InferenceService|0.0.0.0:8001|
|HTTPS Service|0.0.0.0:8000|
|Metrics Service|0.0.0.0:8002|

