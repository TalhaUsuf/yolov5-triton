- [How to clone](#how-to-clone)
- [Start Services](#start-services)
- [Stop services](#stop-services)
- [Flow Chart of the system](#flow-chart-of-the-system)
- [⚡Triton server PORT details](#triton-server-port-details)
- [Some useful requests](#some-useful-requests)
- [☢️ Important](#️-important)

# How to clone

☠️ Make sure that submodule is also cloned, otherwise do it manually `converter/yolov5`

```bash
git clone --recursive https://github.com/TalhaUsuf/yolov5-triton.git
```



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


# Some useful requests

 - [x] 🟩 Get **yolov5** model config:
`<host>` is the url where the triton server is running
`<version>` is the version of the model

```bash
curl --location --request GET 'http://<host>:8000/v2/models/yolov5/versions/<version>/config'
```

- [x] 🟩 Get **inference ready** models:
`<host>` is the url where the triton server is running

```bash
curl --location --request POST 'http://<host>:8000/v2/repository/index' \
--header 'Content-Type: application/json' \
--data-raw '{
    "ready" : true
}'
```

# ☢️ Important

 - model version mode is set to latest 1 so only the latest model will be loaded
 - ensemble model will use the latest version of the model for inference

