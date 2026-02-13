#!/bin/bash

set -e
docker build -t dev-triton-server .

echo "빌드 완료: dev-triton-server"