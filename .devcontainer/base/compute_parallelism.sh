#!/usr/bin/env bash
# Prints a safe `make -j` value for the machine this runs on: the minimum of
# available CPU cores and available_RAM_GB / MEM_PER_JOB_GB.
#
# Why not just `nproc`: some translation units in these builds (heaviest
# case: MAIA's CFD template instantiations, e.g.
# fvcartesiansolverxd_inst_3d_*.cpp) peak at ~4.2GB RSS each - see
# .devcontainer/README.md. A fixed `-j16` over- or
# under-subscribes most machines (too few jobs on a large one, compiler
# OOM kills on a small one).
#
# Usage:
#   JOBS="$(bash compute_parallelism.sh)"        # standalone
#   source compute_parallelism.sh; make -j"$(compute_parallel_jobs)"

MEM_PER_JOB_GB="${MEM_PER_JOB_GB:-5}"  # 4.2GB observed peak + safety margin

compute_parallel_jobs() {
    local cores mem_avail_kb mem_avail_gb jobs

    cores="$(nproc)"

    mem_avail_kb=""
    if [[ -r /proc/meminfo ]]; then
        mem_avail_kb="$(awk '/MemAvailable/{print $2; exit}' /proc/meminfo)"
    fi
    if [[ -z "${mem_avail_kb}" ]]; then
        # No /proc/meminfo (shouldn't happen in a Linux container/image
        # build) - fall back to core count alone rather than fail the build.
        echo "${cores}"
        return
    fi

    mem_avail_gb=$(( mem_avail_kb / 1024 / 1024 ))
    jobs=$(( mem_avail_gb / MEM_PER_JOB_GB ))
    [[ ${jobs} -lt 1 ]] && jobs=1
    [[ ${jobs} -gt ${cores} ]] && jobs=${cores}

    echo "${jobs}"
}

# Only auto-print when executed directly (e.g. `$(compute_parallelism.sh)`),
# not when sourced to pull in the function.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    compute_parallel_jobs
fi
