## Docker image with FEniCS and pyadjoint

1. Build with ```docker build -t hydrogym:latest```
2. Launch image with ```docker run --shm-size=1g -ti -v $PWD:/home/fenics/shared hydrogym:latest```
3. Activate virtualenv `. $VENV/bin/activate`

Alternatively, can use the build and launch scripts (but make sure to activate the venv before running anything).

If you're using an ARM processor you may get a warning about the image platform not matching host platform. Rather than launching with `./launch.sh`, try running the longer `run` command with the flag `--platform linux/amd64` (see [this discussion](https://stackoverflow.com/questions/66662820/m1-docker-preview-and-keycloak-images-platform-linux-amd64-does-not-match-th)). It is noted that `./launch.sh` is preferred due to the fact that using the longer `run` command only allows the user to acces the core `hydrogym` file. This is sufficient for development purposes, but without the additional files such as `examples`, `notebooks`, etc being immediately accesible in your workspace, you may have a more difficult time taking advantage of the full set of features that this package has to offer.
