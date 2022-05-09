#!/bin/bash

cd ..
# docker run -p 8080:8888 --shm-size=1g -it --rm \
#     -v $PWD:/home/fenics/hydrogym hydrogym:latest --entrypoint "jupyter lab --ip=0.0.0.0 --no-browser --allow-root"

# docker run -p 8080:8888 --shm-size=1g -it --rm -v $PWD:/home/hydrogym hydrogym:latest
docker run --shm-size=1g -it --rm -v $PWD:/home/hydrogym hydrogym:latest