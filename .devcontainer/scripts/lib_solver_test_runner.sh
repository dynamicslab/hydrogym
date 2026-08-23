#!/usr/bin/env bash
# Shared test-runner driver for test_gpu_solvers.sh / test_cpu_solvers.sh -
# split out of a single test_all_solvers.sh so each stack can be tested (or
# CI'd, or just iterated on) independently, without needing the other
# stack's container running at all. Not meant to be run directly: source it
# after declaring TESTS/CMD/LOG_DIR/TIMEOUT_SECS, then call
# run_solver_tests "$cid".
#
# This is a SMOKE TEST driver, not a full-run harness: the goal is just
# "does the env actually reset() and step() through the real solver", not a
# complete episode. Every test is capped at TIMEOUT_SECS regardless of which
# command it runs - this matters because an MPMD pairing (maia/nek) where
# the Python side crashes does NOT automatically kill the paired solver
# ranks, which then hang forever waiting for a handshake that will never
# come (hit this for real with nek5000 - see conversation history). Without
# a uniform timeout, one broken test silently blocks the whole run.

container_for() {
  # Prints the first running container ID labeled devcontainer.config_variant=$1
  docker ps --filter "label=devcontainer.config_variant=$1" --format '{{.ID}}' | head -1
}

# Runs ${TESTS} (space-separated names) against the single container id in
# $1, using the CMD[name]=script associative array the caller already
# declared. Writes each test's log to ${LOG_DIR}/<name>.log, prints a
# summary table, and returns 1 if anything FAILed (TIMEOUT/SKIPPED alone
# don't fail the run - see each script's header for why).
run_solver_tests() {
  local cid="$1"
  local -a results=()

  for name in ${TESTS}; do
    local log_file="${LOG_DIR}/${name}.log"

    echo ""
    echo "==================================================================="
    echo "Testing '${name}'  (timeout ${TIMEOUT_SECS}s)"
    echo "Log: ${log_file}"
    echo "==================================================================="

    if [[ -z "${cid}" ]]; then
      echo "SKIPPED: no container available for '${name}'"
      results+=("${name}: SKIPPED (container not running)")
      continue
    fi

    local cmd="${CMD[${name}]:-}"
    if [[ -z "${cmd}" ]]; then
      echo "SKIPPED: unknown test '${name}'"
      results+=("${name}: SKIPPED (unknown test name)")
      continue
    fi

    local step_start step_elapsed timing rc evidence
    step_start=$(date +%s)
    docker exec "${cid}" timeout -k 10 "${TIMEOUT_SECS}" bash -c "${cmd}" 2>&1 | tee "${log_file}"
    rc=${PIPESTATUS[0]}
    step_elapsed=$(( $(date +%s) - step_start ))
    timing="[$(( step_elapsed / 60 ))m $(( step_elapsed % 60 ))s]"

    # Regardless of clean exit, check whether the log actually shows the
    # thing we care about: the env stepped through the real solver at least
    # once. Deliberately specific, not just "step"/"reward" - those words
    # also show up in unrelated boilerplate (e.g. prepare_workspace.py's own
    # "Next steps:" printout, which previously false-positived this check
    # on a run that had actually crashed before ever reaching the solver).
    if grep -qiE "resetting environment|env[._]step[ =]|sim_step=|episode 1/|reward *[:=] *-?[0-9]|total time for [0-9]+ steps" "${log_file}"; then
      evidence="(saw step/reset output)"
    else
      evidence="(no step/reset output seen)"
    fi

    if [[ ${rc} -eq 0 ]]; then
      results+=("${name}: PASS  ${evidence}  ${timing}")
    elif [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
      results+=("${name}: TIMEOUT ${evidence}  ${timing}")
    else
      results+=("${name}: FAIL (exit ${rc}) ${evidence}  ${timing}")
    fi
  done

  echo ""
  echo "==================================================================="
  echo "Solver test summary"
  echo "==================================================================="
  for r in "${results[@]}"; do
    echo "  ${r}"
  done

  for r in "${results[@]}"; do
    [[ "${r}" == *": FAIL"* ]] && return 1
  done
  return 0
}
