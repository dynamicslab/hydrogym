#!/bin/bash

cd ..
docker build -t hydrogym:latest -f docker/Dockerfile .
docker build -t hydrogym:complex -f docker/Dockerfile.complex .
cd docker