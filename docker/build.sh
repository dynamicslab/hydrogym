#!/bin/bash

cd ..
docker build -t hydrogym:latest -f docker/Dockerfile .
cd docker