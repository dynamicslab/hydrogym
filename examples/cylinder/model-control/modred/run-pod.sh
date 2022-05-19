mpiexec -np 4 python unsteady.py
python to_arrays.py
mpiexec -np 4 python pod.py
# python to_functions.py