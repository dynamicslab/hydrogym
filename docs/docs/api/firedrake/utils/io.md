---
sidebar_label: io
title: hydrogym.firedrake.utils.io
---

## SnapshotCallback Objects

```python
class SnapshotCallback(CallbackBase)
```

#### \_\_init\_\_

```python
def __init__(interval: Optional[int] = 1,
             filename: Optional[str] = "snapshots")
```

Save snapshots as checkpoints for modal analysis

Note that this slows down the simulation

