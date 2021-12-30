## Docker image with FEniCS and pyadjoint

1. Build with ```docker build -t hydrogym:latest```
2. Launch image with ```docker run --shm-size=1g -ti -v $PWD:/home/fenics/shared hydrogym:latest```
