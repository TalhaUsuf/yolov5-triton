- [How to clone](#how-to-clone)
- [Start Services](#start-services)
- [Stop services](#stop-services)
- [Testing API](#testing-api)
- [Perf\_analyzer 📈 results](#perf_analyzer--results)
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
docker-compose up -d  --build
```
App will be running 🏃‍♂️on `http://localhost:8005/`


# Stop services

```bash
docker-compose down
```

# Testing API

**if following model is used for testing then class names don't need to be specified**
 - `http://localhost:8005/`
 - Sample Images:
   - [test image 1](testing/people.jpg)
   - [test image 2](testing/sd-fp16-two_smuggl-1-8004.png)
 - Sample Model:
   - [test model](testing/yolov5n.pt)



# Perf_analyzer 📈 results

 - [x] [yolo-postprocessing-B1](perf_analysis/yolo_postprocess_b1_perf.csv)
 - [x] [yolo-postprocessing-B2](perf_analysis/yolo_postprocess_b2_perf.csv)
 - [x] [yolov5-model-B1](perf_analysis/yolov5_b1_perf.csv)
 - [x] [yolov5-model-B2](perf_analysis/yolov5_b2_perf.csv)


|  ensemble-part  |   Concurrency |   Inferences/Second |   Client Send |   Network+Server Send/Recv |   Server Queue |   Server Compute Input |   Server Compute Infer |   Server Compute Output |   Client Recv |   p50 latency |   p90 latency |   p95 latency |   p99 latency |
|---:|--------------:|--------------------:|--------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
|  Yolo-postprocess-B1 |             1 |              7.4983 |          3649 |                       4738 |            348 |                   1275 |                 122293 |                     182 |             7 |        143654 |        158760 |        163193 |        172620 |
|  Yolo-postprocess-B2 |             1 |              6.7764 |          7379 |                      15042 |            358 |                   2204 |                 272046 |                    1183 |            10 |        310661 |        354668 |        357366 |        374222 |
|  Yolo-detection-B1 |             1 |             20.8297 |          1642 |                       3318 |             91 |                    659 |                  34114 |                    1857 |          6253 |         38402 |        100416 |        101235 |        124060 |
|  Yolo-detection-B2 |             1 |             25.8853 |          3685 |                       6090 |            123 |                   1313 |                  42735 |                    9961 |         13117 |         67164 |        108679 |        130117 |        149820 |






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

