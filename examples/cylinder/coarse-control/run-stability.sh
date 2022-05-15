mkdir global-modes

mkdir tmp
NPROC=12
$VENV_COMPLEX/bin/mpiexec -np ${NPROC} $VENV_COMPLEX/bin/python stability.py
$VENV/bin/mpiexec -np ${NPROC} $VENV/bin/python gather.py
cp tmp/*.vtu tmp/*.pvtu tmp/*.pvd global-modes/
rm -r tmp