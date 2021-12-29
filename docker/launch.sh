#!/bin/bash

cd ..
docker run --shm-size=1g -ti -v $PWD:/home/fenics/cfd_gym cfd_gym:latest