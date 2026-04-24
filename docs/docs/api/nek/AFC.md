---
sidebar_label: AFC
title: hydrogym.nek.AFC
---

The classical approaches for wing control
Matching the structure from drl_repo_TEST/src/lib/AFC.py

## AFC Objects

```python
class AFC()
```

A general object for classical AFC approaches

#### policy

```python
def policy(observation)
```

The control policy depends on the observation

#### predict

```python
def predict(observations, state, episode_start, deterministic)
```

Return the action to ENV based on the policy

## OppoCtrl Objects

```python
class OppoCtrl(AFC)
```

Opposition control implementation

#### policy

```python
def policy(observation)
```

v = -alpha * (v - &lt;v&gt;)

## BLCtrl Objects

```python
class BLCtrl(AFC)
```

Steady uniform blowing/suction

#### policy

```python
def policy(observation)
```

v = Psi

## SinWave Objects

```python
class SinWave()
```

Imposing Sinusoidal wave to check the mean

#### load\_node\_info

```python
def load_node_info(Node_Info)
```

Get the node info to impose the Sinusoidal Wave

#### policy

```python
def policy(observation)
```

We define the V-Vel &lt;==&gt; 1

#### nameAgent

```python
@staticmethod
def nameAgent(nid, gllid, iface, ix, iy, iz)
```

Name the agent based on the GRID information

## ZeroCtrl Objects

```python
class ZeroCtrl(AFC)
```

Zero action controller (no control)

#### make\_afc\_controller

```python
def make_afc_controller(env, ctrl_type="AFC")
```

Factory function to create an AFC controller compatible with integrate().

The controller adapts between array-based (NekEnv) and dict-based
(NekParallelEnv, NekPettingZooEnv) formats.

To use an SB3 model, you can pass it directly to integrate():
from stable_baselines3 import PPO
loaded_model = PPO.load(&quot;path/to/model&quot;)
hgym.integrate(env, ..., controller=loaded_model)

**Arguments**:

- `env` - Environment instance (NekEnv, NekParallelEnv, or NekPettingZooEnv)
- `ctrl_type` - Algorithm name (&quot;AFC&quot;, &quot;OC&quot;, &quot;BL&quot;, &quot;SIN&quot;, &quot;ZERO&quot;, or SB3 algorithm name)
  

**Returns**:

- `controller` - Controller object with .predict() method, or None if not an AFC algorithm

