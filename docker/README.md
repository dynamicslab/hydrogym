## Docker image with FEniCS and pyadjoint

1. Build with ```docker build -t hydrogym:latest```
2. Launch image with ```docker run --shm-size=1g -ti -v $PWD:/home/fenics/shared hydrogym:latest```
3. Activate virtualenv `. $VENV/bin/activate`

Alternatively, can use the build and launch scripts (but make sure to activate the venv before running anything)