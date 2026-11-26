#!/usr/bin/env bash
# Run a NekRS rank on the cores and memory of its GPU's NUMA domain (MI300A: one APU per domain), as prefix of the
# NekRS command in an Open MPI MPMD launch:  mpirun -n 1 python ... : -n NP bind_numa.sh nekrs-mpmd ...
# NekRS takes device = node-local rank among the NekRS ranks (DEVICE NUMBER = LOCAL-RANK), i.e. the Open MPI local
# rank minus the HG_CONTROLLER_RANKS controller ranks launched first on the node (default 1). Without binding the
# ranks float over all domains and reach their memory across the sockets (up to 6x slower steps on MI300A nodes).
# HG_BIND=0 disables the binding; without hwloc-bind or with a single CPU NUMA domain the command runs unchanged.
# Under srun (no Open MPI launcher variables) SLURM_LOCALID is the node-local rank.
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-$SLURM_LOCALID}
if [ "${HG_BIND:-1}" != 0 ] && command -v hwloc-bind > /dev/null && [ -n "$LOCAL_RANK" ]; then
  # NUMA domains of the CPU sockets: the first domain of every distinct cpuset. Domains without CPUs of their own
  # (GH200: the GPU memory, listed under the socket of its GPU, so with that socket's cpuset) are skipped; binding a
  # rank's memory to them fails (errno 22) once there are more ranks than sockets, e.g. 2 ranks per GPU.
  n=$(hwloc-calc --number-of numa machine:0 2> /dev/null || echo 1)
  np=$(hwloc-calc --number-of package machine:0 2> /dev/null || echo 1)
  all=$(hwloc-calc machine:0 2> /dev/null)
  doms=() seen=" "
  for ((i = 0; i < n; i++)); do
    c=$(hwloc-calc numa:$i 2> /dev/null) || continue
    case "$seen" in *" $c "*) continue ;; esac
    [ "$np" -gt 1 ] && [ "$c" = "$all" ] && continue   # attached to the whole machine, not to one socket
    seen="$seen$c "
    doms+=("$i")
  done
  if [ "${#doms[@]}" -gt 1 ]; then
    # controller ranks on this node: Open MPI launch 1 (first on the node); srun: 1 on node 0 only
    if [ -z "$OMPI_COMM_WORLD_LOCAL_RANK" ] && [ "${SLURM_NODEID:-0}" != 0 ]; then ctrl=0; else ctrl=${HG_CONTROLLER_RANKS:-1}; fi
    r=${doms[$(( (LOCAL_RANK - ctrl) % ${#doms[@]} ))]}
    echo "[bind_numa] local rank $LOCAL_RANK on $(hostname): NUMA domain $r (of ${#doms[@]} with CPUs: ${doms[*]})" >&2
    exec hwloc-bind --cpubind numa:$r --membind numa:$r -- "$@"
  fi
fi
exec "$@"
