#!/bin/bash

cd ..
docker run --shm-size=1g -ti -v $PWD:/home/fenics/shared cfd_gym:latest
