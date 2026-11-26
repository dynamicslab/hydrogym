"""Print environment.dt of a wing YAML (following its `base:` files), or nothing if it has none (run.sh)."""

import os
import sys

import yaml


def conf_dt(path):
    with open(path) as f:
        c = yaml.safe_load(f) or {}
    v = (c.get("environment") or {}).get("dt")
    if v is None and c.get("base"):
        return conf_dt(os.path.join(os.path.dirname(path), c["base"]))
    return v


if __name__ == "__main__":
    v = conf_dt(sys.argv[1])
    print("" if v is None else repr(float(v)))
