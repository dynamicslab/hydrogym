"""MPMD smoke test for `hydrogym.core_external` split strategies (audit Task 3.5).

Validates both world-split strategies live under mpirun, with no solver
binaries required. Run from the repo root:

  # Nek rank-color protocol: single-app world of 2 (rank 0 controller,
  # rank 1 worker), inter-communicator handshake + one ping-pong message:
  mpirun --allow-run-as-root -np 2 python test/mpmd_smoke_split.py nek

  # MAIA APPNUM protocol: true MPMD, two 1-rank applications:
  mpirun --allow-run-as-root -np 1 python test/mpmd_smoke_split.py appnum : -np 1 python test/mpmd_smoke_split.py appnum

  # Nek protocol error path: wrong world size for the declared nproc
  # (expects a clean RuntimeError; run serially):
  python test/mpmd_smoke_split.py nek-error

Each participant asserts its side of the split and, for `nek`, exchanges a
ping-pong message over the resulting inter-communicator. Any failed check
exits nonzero, which fails the mpirun invocation.
"""

import sys

import numpy as np
from mpi4py import MPI

from hydrogym.core_external import mpi_split, split_comm_by_appnum

world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()
failures = []


def check(cond, msg):
    if not cond:
        failures.append(f"rank {rank}: {msg}")


def check_nek_protocol():
    nproc = size - 1
    sub = mpi_split(world, nproc=nproc)
    check(sub is not None, "mpi_split returned None")
    check(sub.Get_remote_size() == nproc, f"remote size {sub.Get_remote_size()} != {nproc}")

    # Ping-pong over the inter-communicator: controller (world rank 0)
    # sends to remote rank 0, worker echoes it back.
    if rank == 0:
        sub.send(np.array([42.0], dtype="d"), dest=0, tag=7)
        echo = sub.recv(source=0, tag=8)
        check(echo[0] == 43.0, f"controller got {echo[0]!r}, expected 43.0")
    else:
        msg = sub.recv(source=0, tag=7)
        check(msg[0] == 42.0, f"worker got {msg[0]!r}, expected 42.0")
        sub.send(np.array([43.0], dtype="d"), dest=0, tag=8)

    # nproc validation: declaring one more worker than exist must fail
    try:
        mpi_split(world, nproc=nproc + 1)
        check(False, "nproc mismatch did not raise")
    except RuntimeError as e:
        check("world size mismatch" in str(e), f"nproc mismatch raised unexpected message: {e}")


def check_nek_error_path():
    """Serial (non-MPMD) invocation: mpi_split must refuse a 1-rank world."""
    try:
        mpi_split(world)
        print("FAIL: single-rank mpi_split did not raise", flush=True)
        sys.exit(1)
    except RuntimeError as e:
        assert "world size must be >= 2" in str(e)
        print("OK: single-rank world rejected with MPMD launch hint", flush=True)


def check_appnum_protocol():
    appnum, app_comm, app_rank, app_no_ranks, app_group, app_root_in_world, remote_root = split_comm_by_appnum(world)
    check(appnum in (0, 1), f"unexpected APPNUM {appnum}")
    check(app_comm.Get_size() == 1, "each app should hold exactly 1 rank")
    check(app_rank == 0, "sole rank of its app should be app rank 0")
    check(app_root_in_world == rank, "app root in world should be this rank itself")
    check(remote_root == 1 - rank, f"remote root {remote_root} should be the other app's sole rank")

    # Each app's root announces itself over COMM_WORLD using the discovered
    # remote root -- proves the pair (appRootInWorld, remoteRoot) is usable
    # for the MAIA tag protocol.
    world.send(np.array([100 + appnum], dtype="d"), dest=remote_root, tag=15)
    peer = world.recv(source=remote_root, tag=15)
    check(peer[0] == 100 + (1 - appnum), f"got {peer[0]!r} from remote app, expected {100 + (1 - appnum)}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "nek-error":
        check_nek_error_path()
    elif mode == "nek":
        check_nek_protocol()
    elif mode == "appnum":
        check_appnum_protocol()
    else:
        print(f"FAIL: unknown mode {mode!r}", flush=True)
        sys.exit(2)

    if failures:
        for f in failures:
            print(f"FAIL: {f}", flush=True)
        sys.exit(1)
    print(f"OK: rank {rank} mode {mode} all checks passed", flush=True)
