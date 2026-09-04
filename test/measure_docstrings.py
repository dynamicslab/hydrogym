"""Measure docstring coverage (audit Task 7.1) — dev-only helper, not a test.

Method mirrors the audit: ast.get_docstring over every module-level and
nested class/function definition. Run inside a container:

    python test/measure_docstrings.py [file1.py file2.py ...]

With no args, measures the audit's target set.
"""

import ast
import os
import sys

DEFAULT_TARGETS = [
    "hydrogym/core.py",
    "hydrogym/firedrake/flow.py",
    "hydrogym/firedrake/actuator.py",
    "hydrogym/firedrake/solvers/base.py",
    "hydrogym/firedrake/solvers/bdf_ext.py",
    "hydrogym/firedrake/solvers/integrate.py",
    "hydrogym/firedrake/solvers/stabilization.py",
    "hydrogym/firedrake/envs/cavity/flow.py",
    "hydrogym/firedrake/envs/cylinder/flow.py",
    "hydrogym/firedrake/envs/pinball/flow.py",
    "hydrogym/firedrake/envs/step/flow.py",
    "hydrogym/jax/env_core.py",
    "hydrogym/jax/equation.py",
    "hydrogym/jax/solvers/base.py",
    "hydrogym/jax/envs/channel.py",
    "hydrogym/jax/envs/kolmogorov.py",
    # Additional files the audit's Finding-4 table measured but that were
    # omitted from the original target list. (hydrogym/nek/integrate.py is
    # deliberately omitted — under concurrent edit at measurement time.)
    "hydrogym/data_manager.py",
    "hydrogym/hf_env_mixin.py",
    "hydrogym/core_external.py",
    "hydrogym/nek/env.py",
    "hydrogym/maia/env_core.py",
    "hydrogym/jaxfluids/env_core.py",
]


def measure(path):
    tree = ast.parse(open(path).read())
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    total = len(nodes)
    have = sum(1 for n in nodes if ast.get_docstring(n))
    return have, total


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # --fail-under <pct> sets a CI exit-code gate on the TOTAL coverage;
    # everything else is treated as explicit file targets.
    fail_under = None
    args = list(sys.argv[1:])
    if "--fail-under" in args:
        i = args.index("--fail-under")
        fail_under = float(args[i + 1])
        del args[i : i + 2]
    targets = args or DEFAULT_TARGETS
    grand_have = grand_total = 0
    for t in targets:
        p = t if os.path.isabs(t) else os.path.join(root, t)
        if not os.path.exists(p):
            print(f"{t:50s} MISSING")
            continue
        have, total = measure(p)
        grand_have += have
        grand_total += total
        pct = 100 * have / total if total else 100.0
        print(f"{t:50s} {have:3d}/{total:<3d} {pct:6.1f}%")
    if not grand_total:
        return
    total_pct = 100 * grand_have / grand_total
    print(f"{'TOTAL':50s} {grand_have:3d}/{grand_total:<3d} {total_pct:6.1f}%")
    if fail_under is not None:
        if total_pct < fail_under:
            print(
                f"FAIL: docstring coverage {total_pct:.1f}% is below the "
                f"{fail_under:.1f}% floor (audit Task 7.2)"
            )
            sys.exit(1)
        print(f"OK: docstring coverage at or above the {fail_under:.1f}% floor")


if __name__ == "__main__":
    main()
