---
sidebar_label: io
title: hydrogym.jax.utils.io
---

#### create\_animation

```python
def create_animation(trajectory, gif_name, frame_interval_factor)
```

Produces an animation of the trajectory.

**Arguments**:

- `trajectory` - fft vorticity trajectory (array, or the path of a numpy file)
- `gif_name` - file name of gif file that will be saved
- `frame_interval_factor` - frame interval as related to the length of the trajectory.

