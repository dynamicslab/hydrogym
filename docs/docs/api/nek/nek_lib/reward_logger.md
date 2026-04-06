---
sidebar_label: reward_logger
title: hydrogym.nek.nek_lib.reward_logger
---

Real-time Reward Logger for NEK5000 DRL Environment
@yuningw

## RewardLogger Objects

```python
class RewardLogger()
```

A real-time reward logger that writes rewards to CSV files for monitoring.
Supports both per-agent and aggregated reward logging.

#### \_\_init\_\_

```python
def __init__(log_dir: Union[str, Path],
             log_name: str = "rewards",
             log_per_agent: bool = True,
             log_aggregated: bool = True,
             flush_frequency: int = 10)
```

Initialize the reward logger.

**Arguments**:

- `log_dir` - Directory to save log files
- `log_name` - Base name for log files
- `log_per_agent` - Whether to log individual agent rewards
- `log_aggregated` - Whether to log aggregated rewards (mean, std, etc.)
- `flush_frequency` - How often to flush data to disk (every N steps)

#### log\_rewards

```python
def log_rewards(rewards: Dict[str, float],
                dUdy_raw: Optional[Dict[str, float]] = None,
                episode: Optional[int] = None,
                step: Optional[int] = None)
```

Log rewards for all agents.

**Arguments**:

- `rewards` - Dictionary mapping agent names to reward values
- `dUdy_raw` - Dictionary mapping agent names to raw dUdy values (optional)
- `episode` - Episode number (optional)
- `step` - Step number within episode (optional)

#### log\_episode\_summary

```python
def log_episode_summary(episode: Optional[int] = None)
```

Log summary statistics for the current episode.

**Arguments**:

- `episode` - Episode number (optional)

#### flush

```python
def flush()
```

Flush all open files.

#### close

```python
def close()
```

Close all open files.

#### get\_latest\_stats

```python
def get_latest_stats() -> Dict[str, float]
```

Get the latest reward statistics.

#### get\_agent\_stats

```python
def get_agent_stats() -> Dict[str, Dict[str, float]]
```

Get statistics for each agent.

## SimpleRewardLogger Objects

```python
class SimpleRewardLogger()
```

A simplified reward logger for basic CSV logging.

#### \_\_init\_\_

```python
def __init__(log_file: Union[str, Path])
```

Initialize simple reward logger.

**Arguments**:

- `log_file` - Path to the CSV log file

#### log\_reward

```python
def log_reward(agent_name: str,
               reward: float,
               dUdy_raw: float = 0.0,
               episode: Optional[int] = None,
               step: Optional[int] = None)
```

Log a single reward.

**Arguments**:

- `agent_name` - Name of the agent
- `reward` - Reward value
- `dUdy_raw` - Raw dUdy value
- `episode` - Episode number
- `step` - Step number

#### new\_episode

```python
def new_episode()
```

Mark the start of a new episode.

#### flush

```python
def flush()
```

Flush the file.

#### close

```python
def close()
```

Close the file.

