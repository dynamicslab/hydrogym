"""Multi-rank round-trip test for NekEnv's batched state/action RPC exchange.

Exercises the real ``NekEnv._send_action`` / ``NekEnv._get_state`` paths over
a real MPI intercommunicator, with world ranks 1..N-1 standing in for the
Nek solver side (no Nek binary needed). World rank 0 drives the environment
methods; the other ranks play the solver exactly the way the Fortran side
does: plain blocking point-to-point ``Send``/``Recv`` with the same
(dest/tag) pairs as the production protocol (see ``tag_dict`` in
``hydrogym.nek.env``). This pins the batching refactor's protocol contract:
same tags, same message contents, same per-source ordering, same buffer
layout.

Run under mpirun (world rank 0 = controller, one fake solver node per
additional rank), e.g.::

    mpirun --allow-run-as-root -np 3 python -m pytest test/test_nek_rpc_batching.py -v

Skipped automatically when mpi4py/pandas are absent or when the process was
not launched with at least 2 MPI ranks (so a plain ``pytest .`` run is a
no-op skip, not a failure or a hang). When launched under mpirun, guard it
with a timeout: a broken exchange deadlocks rather than fails (both sides
end up in a blocking receive -- same as any MPMD test, see
``test/mpmd_smoke_split.py``).
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("pandas")

from mpi4py import MPI  # noqa: E402

from hydrogym.nek.env import NekEnv, tag_dict  # noqa: E402

WORLD = MPI.COMM_WORLD
WORLD_SIZE = WORLD.Get_size()

if WORLD_SIZE < 2:
    pytest.skip(
        "needs mpirun -np N (1 controller rank + N-1 fake solver ranks)",
        allow_module_level=True,
    )

# Topology: one fake solver "node" per non-controller rank, 3 actuators each.
# NID_LIST must match the intercomm's remote group size -- the destination of
# each action Isend is a remote-group rank, so impersonating more nodes than
# there are solver ranks would be an invalid MPI destination.
NID_LIST = list(range(WORLD_SIZE - 1))
TOTCTRL = 4  # buffer slots per node (production sends the full TOTCTRL width)
NFLDC = 2  # state fields per actuator (obs_per_actuator)
ACTUATORS_PER_NODE = 3
ACT_NID = np.array([nid for nid in NID_LIST for _ in range(ACTUATORS_PER_NODE)], dtype=np.int32)
N_ACT = len(ACT_NID)

# Wire values (float64 on the wire, exactly like the real protocol)
STATE_PER_NODE = {
    nid: np.array(
        [
            [1.0 + 10 * nid, 2.0 + 10 * nid, 3.0 + 10 * nid, 4.0 + 10 * nid],
            [5.0 + 10 * nid, 6.0 + 10 * nid, 7.0 + 10 * nid, 8.0 + 10 * nid],
        ]
    )  # (NFLDC, TOTCTRL)
    for nid in NID_LIST
}
ACTION_FLAT = np.arange(1, N_ACT + 1, dtype=np.float32) / 10.0
CURRENT_TIME = 42.5

# Expected on-wire action values: the wire is float64, but values arrive via
# the float32 action space, so they are the float32-round-tripped ones.
ACTION_PER_NODE = {
    nid: ACTION_FLAT[il * ACTUATORS_PER_NODE : (il + 1) * ACTUATORS_PER_NODE].astype(np.float64)
    for il, nid in enumerate(NID_LIST)
}

# Expected observation from the production flatten order
# (state_buffer[il, :, jl] for each actuator jl of node nid, in order).
EXPECTED_OBS = np.concatenate(
    [STATE_PER_NODE[nid][:, jl] for nid in NID_LIST for jl in range(ACTUATORS_PER_NODE)]
).astype(np.float32)


def _make_intercomm():
    """Build the controller<->solver intercommunicator for this multi-rank job."""
    rank = WORLD.Get_rank()
    color = 0 if rank == 0 else 1
    local = WORLD.Split(color, rank)
    if rank == 0:
        remote_leader = 1  # world rank of the solver side's leader
    else:
        remote_leader = 0  # world rank of the controller
    return local.Create_intercomm(local_leader=0, peer_comm=WORLD, remote_leader=remote_leader, tag=12345)


def _make_bare_env(comm):
    """Bare NekEnv instance with only the attributes the RPC paths touch."""
    env = NekEnv.__new__(NekEnv)
    env.sub_comm = comm
    env.TOTCTRL = TOTCTRL
    env.obs_per_actuator = NFLDC
    env.uniqID = np.array(NID_LIST)
    env.nNID = len(NID_LIST)
    env.n_actuators = N_ACT
    env.actuator_info = {"NID": ACT_NID}
    env.normalize_input = "None"
    env.znmf_avg = 0  # ZNMF "done by Nek" -> action passed through unchanged
    return env


def _fake_solver(comm):
    """World ranks 1..N-1: answer the RPCs exactly like the Fortran side."""
    nid = WORLD.Get_rank() - 1  # this rank's node id (remote-group index on the controller side)

    # -- _send_action: the CNTRL command goes to the solver leader only, then
    # one TOTCTRL-double action message per node
    if nid == 0:
        cmd = bytearray(5)
        comm.Recv([cmd, MPI.CHARACTER], source=0, tag=tag_dict["COMMAND"]["tag"])
        assert bytes(cmd) == b"CNTRL"
    buf = np.empty(TOTCTRL, dtype=np.float64)
    comm.Recv([buf, MPI.DOUBLE], source=0, tag=nid + tag_dict["ACTION"]["tag"])
    np.testing.assert_array_equal(buf[: len(ACTION_PER_NODE[nid])], ACTION_PER_NODE[nid])
    # NOTE: padding slots beyond a node's actuator count are NOT asserted to
    # be zero -- production allocates them with np.ndarray (uninitialized)
    # and always has; unchanged here.

    # -- _get_state: the STATE command also goes to the leader only; then the
    # leader sends the current time and every node sends its state grid
    if nid == 0:
        comm.Recv([cmd, MPI.CHARACTER], source=0, tag=tag_dict["COMMAND"]["tag"])
        assert bytes(cmd) == b"STATE"
        comm.Send([np.array([CURRENT_TIME], dtype=np.float64), MPI.DOUBLE], dest=0, tag=1998)
    for t in range(NFLDC):
        # Same tag formula as the environment's receive side; for nid=0 every
        # field shares one tag (order carried by MPI non-overtaking)
        comm.Send(
            [np.ascontiguousarray(STATE_PER_NODE[nid][t]), MPI.DOUBLE],
            dest=0,
            tag=nid * (t + 1) + tag_dict["STATE"]["tag"],
        )


def test_nek_rpc_batching_roundtrip():
    rank = WORLD.Get_rank()
    comm = _make_intercomm()

    if rank > 0:
        _fake_solver(comm)
        comm.Disconnect()
        return

    env = _make_bare_env(comm)

    # Action path (world rank 0; the fake solvers assert the received bytes)
    env._send_action(ACTION_FLAT.copy())

    # State path
    current_time, observation = env._get_state()

    assert current_time == CURRENT_TIME
    np.testing.assert_array_equal(observation, EXPECTED_OBS)

    comm.Disconnect()
