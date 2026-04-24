---
sidebar_label: configs
title: hydrogym.nek.configs
---

Configuration of runner and NEK
@yuningw

## Runner Objects

```python
@dataclass
class Runner()
```

#### RL\_algorithm

Policy to use

#### nb\_episodes

The total amount of Episodes is n_episode =  nb_interactions * nb_episodes

#### nb\_warmup\_episodes

Only for transfer learning, the number of episodes to warm up the critic

#### train\_steps

Off-policy only:  Steps to train the model, N-train = n_episode // train_steps

#### ckpt\_int

Frequency of saving policy

#### n\_epochs

Setup of Pol

#### seed

If -5000 -&gt; None

#### random\_init

-1==No Shuffle, use the No.init = RANK; -2==NOT Cover the current rs8

#### rank

NAME of test case

#### rew\_mode

We let all subdomain share the same reward

#### normalize\_input

Between None, utau, std

#### npl\_state

Should be consistent with SIZE

## Simulation Objects

```python
@dataclass
class Simulation()
```

#### nproc

ranks for running

#### TOTCTRL

Be consistent with SIZE

#### znmf\_avg

1==Open

#### ys\_bdf

Volume for Body-Force

#### amp\_bdf

Amplitude for Body-Force

#### lx1

Polynomial Order

#### Lz

Size of domain

#### Nz

No. Spectral Elements

#### writeControl

runTime

#### PARFWRITE

Do we write runtime parameter file

#### PARFNAME

Runtime parameter file name for output (without .par)

#### LOGLEVEL

Logging threshold for toolboxes

#### WALLTIME

Simulation wall time

#### READCHKPT

Restat from checkpoint

#### CHKPFNUMBER

Restart file number

#### CHKPINTERVAL

Checkpiont saving frequency (number of time steps)

