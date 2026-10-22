# TODO (later): multi-simulation training (several NekRS channels, one shared policy)

Status 2026-10-22: deferred; first reproduce the paper's results with NekRS.

Idea: N channel simulations (one GPU each, different initial conditions from the snapshot library) feed one
learner. Same 5760 agents (shared actor/critic) in every simulation -> N x 5760 transitions per action at about
the same wall time per action, and less correlated samples than one flow (the current TD3 episodes from the
same setup vary from -0.03 to 0.10 in last-third reward).

Why not one simulation on several GPUs: the channel mesh (720 el, N=7, ~370k points) is too small;
NP4 was 0.370 s/action vs NP1 0.358 s/action (P2P is broken on this host, halo exchange via host).

Implementation plan
1. Launch: one mpirun, learner rank + N NekRS instances with NEKRS_MPMD_COLOR = 1..N (mpmd/driver.cpp already
   splits MPI_COMM_WORLD by this color; no solver change). Verify the GPU binding per instance in nekrs.sh and
   build the udf once before the N instances start (cache race).
2. interface.py: controller splits with color 0 and creates one intercommunicator per instance (remote leader =
   first world rank of each color). Add an async request (send ACTUA + STEPS to all, then collect) so the
   simulations advance concurrently.
3. Env: wrapper holding N NekRSFlowEnv, exposed as an SB3 VecEnv with N x 5760 sub-envs (extend
   multiagent.ParallelToVecEnv). Each instance draws its own snapshot index; build the library once and
   load it from disk in all instances (SNAPF; same partition, NP1 each).
4. Training: scale gradient steps per update with N (or report the changed update-to-data ratio).
5. Also worth doing first: move the per-step CFL and wall-gradient monitor to the device (currently host-side
   copies of the velocity every solver step in hydrogym/nekrs/udf/hydrogym_nekrs.hpp).
