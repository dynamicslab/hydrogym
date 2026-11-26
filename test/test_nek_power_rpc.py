"""Round trip of the Nek5000 actuation-power request (POWER) over a real MPI intercommunicator.

World rank 0 drives ``NekEnv._recv_power`` / ``_power_info``; ranks 1..N-1 play the Fortran side of
``drl/drl_power.f`` (header from the leader with tag 1995, per-node end and mean buffers with tags 110000+NID and
120000+NID). Checks the C_P normalisation (0.5 rho U^3 A of ``power.reference``, or the power area), the
per-actuator order (as the rewards), the sampling choice with the fallback to ``end``, that a zero power weight
leaves the reward unchanged and that a solver which ignores the request raises instead of blocking.

    mpirun -np 3 python -m pytest test/test_nek_power_rpc.py -v

Skipped without mpirun (fewer than 2 ranks).
"""

import numpy as np
import pytest

pytest.importorskip("mpi4py")
pytest.importorskip("omegaconf")
pytest.importorskip("pandas")

from mpi4py import MPI  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from hydrogym.nek.env import NekEnv, tag_dict  # noqa: E402

WORLD = MPI.COMM_WORLD
if WORLD.Get_size() < 2:
    pytest.skip("needs mpirun -np N (1 controller rank + N-1 fake solver ranks)", allow_module_level=True)

NID_LIST = list(range(WORLD.Get_size() - 1))
TOTCTRL = 5
PER_NODE = 3
ACT_NID = np.array([nid for nid in NID_LIST for _ in range(PER_NODE)], dtype=np.int32)
AREA = 0.2
HDR = np.array([1e-4, 3e-4, 5e-4, 1.1e-4, AREA, 2e-4, 4e-4, 6e-4, 2.1e-4, AREA, 1.0, 31.0, 31 * 4e-5, 28.5, 1e-6, 2e-6])


def _node_end(nid):
    return np.arange(TOTCTRL, dtype=np.float64) + 10.0 * nid


def _node_mean(nid):
    return -(np.arange(TOTCTRL, dtype=np.float64) + 10.0 * nid)


def _intercomm():
    rank = WORLD.Get_rank()
    local = WORLD.Split(0 if rank == 0 else 1, rank)
    return local.Create_intercomm(0, WORLD, 1 if rank == 0 else 0, tag=777)


def _fake_solver(comm, hdr):
    nid = WORLD.Get_rank() - 1
    if nid == 0:
        cmd = bytearray(5)
        comm.Recv([cmd, MPI.CHARACTER], source=0, tag=tag_dict["COMMAND"]["tag"])
        assert bytes(cmd) == b"POWER"
        comm.Send([hdr, MPI.DOUBLE], dest=0, tag=tag_dict["POWER"]["tag"])
    comm.Send([_node_end(nid), MPI.DOUBLE], dest=0, tag=nid + tag_dict["POWER_END"]["tag"])
    comm.Send([_node_mean(nid), MPI.DOUBLE], dest=0, tag=nid + tag_dict["POWER_MEAN"]["tag"])


def _env(comm, power):
    env = NekEnv.__new__(NekEnv)
    env.sub_comm = comm
    env.TOTCTRL = TOTCTRL
    env.uniqID = np.array(NID_LIST)
    env.nNID = len(NID_LIST)
    env.n_actuators = len(ACT_NID)
    env.actuator_info = {"NID": ACT_NID}
    env.conf = OmegaConf.create({"simulation": {"density": 1.0}, **({"power": power} if power is not None else {})})
    env._setup_power()
    return env


def test_power_rpc():
    rank = WORLD.Get_rank()
    comm = _intercomm()
    hdr_nomean = HDR.copy()
    hdr_nomean[10] = 0.0
    if rank > 0:
        for h in (HDR, HDR, hdr_nomean):
            _fake_solver(comm, h)
        comm.Disconnect()
        return

    # no power section: disabled, p096 = 0
    off = _env(comm, None)
    assert not off.power_enabled and off.conf.simulation.power_mode == 0

    # reference {velocity, length, span}: C_P = P / (0.5 U^3 c span)
    env = _env(comm, {"reference": {"velocity": 1.0, "length": 1.0, "span": 0.1}})
    assert env.power_enabled and env.power_sampling == "mean" and env.conf.simulation.power_mode == 1
    ref = 0.5 * 1.0 * 0.1
    info = env._power_info(env._recv_power(), "mean")
    assert info["power_sampling"] == "mean"
    np.testing.assert_allclose(info["actuation_power"], HDR[5] / ref)
    np.testing.assert_allclose(info["actuation_power_positive"], HDR[6] / ref)
    np.testing.assert_allclose(info["actuation_power_abs"], HDR[7] / ref)
    np.testing.assert_allclose(info["actuation_power_full_pressure"], HDR[8] / ref)
    expect = np.concatenate([_node_mean(nid)[:PER_NODE] for nid in NID_LIST]) / ref
    np.testing.assert_array_equal(info["actuation_power_per_actuator"], expect)
    assert env.power_config.apply(0.123, {"signed": 1.0, "positive": 2.0, "abs": 3.0}) == 0.123

    info_end = env._power_info(env._recv_power(), "end")
    np.testing.assert_allclose(info_end["actuation_power"], HDR[0] / ref)
    expect = np.concatenate([_node_end(nid)[:PER_NODE] for nid in NID_LIST]) / ref
    np.testing.assert_array_equal(info_end["actuation_power_per_actuator"], expect)

    # no per-step samples (p096 != 1): mean falls back to end; default reference = power area
    env2 = _env(comm, {"power_sampling": "mean"})
    with pytest.warns(UserWarning, match="no per-step samples"):
        info2 = env2._power_info(env2._recv_power(), "mean")
    assert info2["power_sampling"] == "end"
    np.testing.assert_allclose(info2["actuation_power"], HDR[0] / (0.5 * AREA))

    comm.Disconnect()


def test_power_no_reply_raises():
    """An executable without drl_power.f reads the request and never answers: the first POWER times out."""
    rank = WORLD.Get_rank()
    comm = _intercomm()
    if rank > 0:
        if rank == 1:
            cmd = bytearray(5)
            comm.Recv([cmd, MPI.CHARACTER], source=0, tag=tag_dict["COMMAND"]["tag"])
            assert bytes(cmd) == b"POWER"
        comm.Barrier()
        comm.Disconnect()
        return
    env = _env(comm, {"power_sampling": "end", "reply_timeout": 0.5})
    with pytest.raises(RuntimeError, match="did not answer the request POWER"):
        env._recv_power()
    comm.Barrier()
    comm.Disconnect()


def test_power_end_mode_and_weight():
    env = NekEnv.__new__(NekEnv)
    env.conf = OmegaConf.create({"simulation": {}, "power": {"power_sampling": "end", "power_weight": 0.5}})
    env._setup_power()
    assert env.conf.simulation.power_mode == 2 and env.power_sampling == "end"
    assert env.power_config.apply(1.0, {"signed": 0.1, "positive": 0.2, "abs": 0.4}) == pytest.approx(0.8)
    env = NekEnv.__new__(NekEnv)
    env.conf = OmegaConf.create({"simulation": {}, "power": {"enabled": False, "power_weight": 0.5}})
    with pytest.raises(Exception, match="power_weight"):
        env._setup_power()
