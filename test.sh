#!/bin/bash

tritonserver \
    --model-repository=/repository/models \
    --http-port=8000 \
    --model-control-mode=explicit \
    --load-model=universal_model \
    --load-model=json_model \
    --log-verbose=1