"""
Configuration of runner and NEK
@yuningw
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf


@dataclass
class Runner:
    RL_algorithm: str = "DDPG"  # Policy to use

    # Essential Parameter
    # Number of "interactions" which is just better for understanding stepping
    nb_interactions: int = 3000
    nb_episodes: int = (
        100  # The total amount of Episodes is n_episode =  nb_interactions * nb_episodes
    )
    nb_warmup_episodes: int = (
        1  # Only for transfer learning, the number of episodes to warm up the critic
    )
    train_steps: int = (
        300  # Off-policy only:  Steps to train the model, N-train = n_episode // train_steps
    )
    ckpt_int: int = 1  # Frequency of saving policy

    # Common: For Roll-out/Replay buffer
    custom_policy: bool = True
    policy_file: str = "conf/default_custom_policy.yml"
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 25  # Setup of Pol
    buffer_size: int = 10_000_000

    # if using PPO, it must be TRUE
    rescale_actions: bool = True
    target_kl: float = 0.02
    likhood_clipping: float = 0.5
    gamma: float = 0.99
    ent_coef: float = 0.000
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5

    # only for DDPG
    action_noise: float = 0.006388353
    # For each training step, how many times we extract the off-policy history
    # from the history buffer with batch size?
    gradient_steps: int = 64
    size_history: int = 10
    tau: float = 0.005
    learning_starts: int = 100
    # for TD3, Note: This is default in SB3
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5

    # Random Seed
    seed: int = -5000  # If -5000 -> None

    ## The replay buffer
    custom_buffer: bool = False
    keep_frac: float = 0.25
    buffer_mode: str = "rotate"

    # Agent loading/resuming options
    random_init: int = (
        2  # -1==No Shuffle, use the No.init = RANK; -2==NOT Cover the current rs8
    )
    agent_run_name: int = 0
    load_agent: bool = False
    rewrite_input_files: bool = False
    evaluation: bool = False
    rank: int = 0  # NAME of test case
    policy: str = ""
    learnt_policy: bool = False
    vars_record: bool = True
    vars_record_freq: int = 1
    # On Interaction
    rew_mode: str = "Homo"  # We let all subdomain share the same reward
    normalize_input: str = "utau"  # Between None, utau, std
    ctrl_min_amp: float = -0.06388353
    ctrl_max_amp: float = 0.06388353
    ctrl_array_size: int = 1
    npl_state: int = 2  # Should be consistent with SIZE

    u_tau: float = 0.047
    dUdy: float = 12.875


@dataclass
class Simulation:
    CASENAME: str = "phill"
    solver_version: str = "v19"
    # I/O
    exeName: str = "nek5000"
    nzs: int = 1
    nxs: int = 1
    out_log: str = "logfile"
    host: str = "localhost"
    hostfile: str = ""
    compile_path: str = "01_compile"
    restart_folder: str = "restart-files"
    # VERY IMPORTANT
    # --------------------------------
    nproc: int = 14  # ranks for running
    TOTCTRL: int = 10  # Be consistent with SIZE
    ndrl: int = 3
    znmf_avg: int = 1  # 1==Open
    target_cfl: float = 0.5
    y_sensing: float = 15.0
    retau: float = 180.0
    # ---- Body-Force Damping config, it will be used only if BDFD is on ----
    ys_bdf: float = 20.0  # Volume for Body-Force
    amp_bdf: float = 5.0  # Amplitude for Body-Force
    # Scale for Body-Force, it depends on the uncontrolled channel,
    # but the drl sensing plane depends on the body-force channel
    ret_bdf: float = 207.0
    # --------------------------------

    # slurm specific
    num_nodes_srun: int = 4
    oversubscribe: bool = False
    # ------------------------
    # SIZE
    # ------------------------
    lx1: int = 6  # Polynomial Order
    Lx: float = 2.67
    Ly: float = 1.0
    Lz: float = 0.8  # Size of domain
    Nx: int = 4
    Ny: int = 16
    Nz: int = 4  # No. Spectral Elements
    tmax: float = 1400
    # --------------------------
    # .par file
    # --------------------------
    ### General
    stopAt: str = "numSteps"
    numSteps: int = 10
    dt: float = -2.0e-03
    timeStepper: str = "bdf3"
    variableDt: str = "no"
    writeControl: str = "timeStep"  # runTime
    writeInterval: int = 5
    dealiasing: str = "yes"
    filtering: str = "none"
    filterWeight: float = 0.01
    filterCutoffRatio: float = 0.9
    # [PROBLEMTYPE]
    stressFormulation: str = "no"
    variableProperties: str = "no"
    # [PRESSURE]
    p_residualTol: float = 1e-8
    p_residualProj: str = "no"
    # [VELOCITY]
    v_residualTol: float = 1e-8
    v_residualProj: str = "no"
    density: float = 1.0
    viscosity: int = -2800
    advection: str = "yes"

    # [_RUNPAR]               # Runtime parameter section for rprm module
    PARFWRITE: str = "no"  # Do we write runtime parameter file
    PARFNAME: str = (
        "outparfile"  # Runtime parameter file name for output (without .par)
    )

    # [_MONITOR]              # Runtime parameter section for monitor module
    LOGLEVEL: int = 1  # Logging threshold for toolboxes
    WALLTIME: str = "23:45"  # Simulation wall time

    # [_CHKPOINT]             # Runtime paramere section for checkpoint module
    READCHKPT: str = "yes"  # Restat from checkpoint
    CHKPFNUMBER: int = 1  # Restart file number
    CHKPINTERVAL: int = 5000  # Checkpiont saving frequency (number of time steps)

    # [_STAT]             # Runtime paramere section for statistics module
    AVSTEP: int = 10
    IOSTEP: int = 10000

    # [_TSRS]             # Runtime paramere section for time series module
    SMPSTEP: int = 23


@dataclass
class Logging:
    run_name: int = int(time.time())
    group: Optional[str] = None
    notes: Optional[str] = None
    save_dir: str = "./runs"


@dataclass
class Config:
    simulation: Simulation = field(default_factory=Simulation)
    runner: Runner = field(default_factory=Runner)
    logging: Logging = field(default_factory=Logging)


def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("conf", help="Configuration")

    subparser.add_argument(
        "files_or_overrides",
        type=str,
        metavar="arg",
        nargs="*",
        help="Config YAML files e.g. `conf.yaml` or overrides e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=parse_cli)


def parse_cli(files_or_overrides: List[str], **ignored_kwargs):
    files = []
    overrides = []
    for x in files_or_overrides:
        if "=" in x:
            overrides.append(x)
        elif x.endswith((".yaml", ".yml")):
            files.append(x)
        else:
            raise ValueError(f"Unrecognized: {x}")
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        *map(OmegaConf.load, files),
        OmegaConf.from_dotlist(overrides),
    )
    print(OmegaConf.to_yaml(conf, resolve=True))


if __name__ == "__main__":
    pass
