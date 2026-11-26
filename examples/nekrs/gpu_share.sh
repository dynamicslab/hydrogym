#!/usr/bin/env bash
# Run more NekRS ranks per node than the node has GPUs: give each rank one GPU, round robin, and start NekRS on it.
#
# NekRS selects device = node-local rank (DEVICE NUMBER = LOCAL-RANK) and does not wrap around, so with more ranks
# than GPUs on a node the extra ranks find no device. Used as a prefix of the NekRS command in an MPMD launch,
#
#   mpirun -n 1 python deploy.py ... : -n 8 gpu_share.sh nekrs-mpmd --setup case.par --backend CUDA
#
# this script makes GPU (k mod nGPU) the only visible device of NekRS rank k on the node and appends
# `--device-id 0` to the command, so that NekRS uses that device. With 8 ranks on a node with 4 GPUs, 2 ranks share
# each GPU. In examples/nekrs/wing/run.sh it is switched on with GPU_SHARE=1.
#
# Node-local rank: OMPI_COMM_WORLD_LOCAL_RANK (Open MPI), SLURM_LOCALID (srun), MPI_LOCALRANKID (MPICH / Intel MPI)
# or PMIX_RANK (single-node PMIx launch). k = node-local rank minus the controller ranks started first on the node
# (HG_CONTROLLER_RANKS, default 1, as in bind_numa.sh; under srun only on node 0), so the GPU of rank k belongs to the
# NUMA domain that bind_numa.sh selects for it.
#
# GPUs: the list in CUDA_VISIBLE_DEVICES (NVIDIA) or ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES (AMD) when it is set,
# e.g. by Slurm; otherwise 0..n-1 with n from HG_NGPU or from `nvidia-smi -L` / `rocm-smi`. The vendor follows
# HG_GPU_VENDOR (nvidia | amd), else NEKRS_BACKEND (CUDA | HIP), else the tool that is installed.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: gpu_share.sh <nekrs command> [arguments]" >&2
    exit 2
fi

local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${MPI_LOCALRANKID:-${PMIX_RANK:-}}}}
if [[ -z "$local_rank" ]]; then
    echo "[gpu_share] no node-local rank in the environment, running the command unchanged" >&2
    exec "$@"
fi

if [[ -z "${OMPI_COMM_WORLD_LOCAL_RANK:-}" && -n "${SLURM_LOCALID:-}" && "${SLURM_NODEID:-0}" != 0 ]]; then
    ctrl=0
else
    ctrl=${HG_CONTROLLER_RANKS:-1}
fi
k=$((local_rank - ctrl))
((k >= 0)) || k=$local_rank

vendor=${HG_GPU_VENDOR:-}
if [[ -z "$vendor" ]]; then
    case "${NEKRS_BACKEND:-}" in
        CUDA | cuda) vendor=nvidia ;;
        HIP | hip) vendor=amd ;;
        *)
            if command -v nvidia-smi > /dev/null 2>&1; then
                vendor=nvidia
            elif command -v rocm-smi > /dev/null 2>&1; then
                vendor=amd
            else
                vendor=nvidia
            fi
            ;;
    esac
fi

if [[ "$vendor" == amd ]]; then
    visible=${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}
else
    visible=${CUDA_VISIBLE_DEVICES:-}
fi

gpus=()
if [[ -n "$visible" ]]; then
    IFS=, read -ra gpus <<< "$visible"
else
    n=${HG_NGPU:-}
    if [[ -z "$n" ]]; then
        if [[ "$vendor" == amd ]]; then
            n=$(rocm-smi --showid --csv 2> /dev/null | grep -c '^card' || true)
        else
            n=$(nvidia-smi -L 2> /dev/null | grep -c '^GPU' || true)
        fi
    fi
    if [[ -z "$n" || "$n" -lt 1 ]]; then
        echo "[gpu_share] no GPU found (set HG_NGPU or the visible-device list)" >&2
        exit 1
    fi
    for ((i = 0; i < n; i++)); do gpus+=("$i"); done
fi

gpu=${gpus[$((k % ${#gpus[@]}))]}
if [[ "$vendor" == amd ]]; then
    export ROCR_VISIBLE_DEVICES=$gpu HIP_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$gpu
fi
echo "[gpu_share] local rank $local_rank (NekRS rank $k on this node): GPU $gpu of ${#gpus[@]}" >&2
exec "$@" --device-id 0
