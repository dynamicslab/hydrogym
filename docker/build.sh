#!/bin/bash

cd ..
docker build -t cfd_gym:latest -f docker/Dockerfile .
cd docker