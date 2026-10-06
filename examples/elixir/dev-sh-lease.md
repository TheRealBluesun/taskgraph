# Elixir: replacing dev.sh's simulator lock with a `taskgraph lease`

`dev.sh` currently guards the one iOS Simulator with an `mkdir` lock dir (`SIM_LOCK=/tmp/elixir-sim.lock`)
plus pid/staleness heuristics and a priority queue. taskgraph replaces exactly that machinery with a
cross-process lease (SPEC §5): a slot is held **iff** a process holds `flock(LOCK_EX|LOCK_NB)` on
`$TASKGRAPH_STATE/leases/simulator/slot-0`, and the kernel drops the lock the instant that process exits,
crashes or is `kill -9`ed. There is no stale state to detect, so all the `ps`-grepping, holder-pid and
queueing code disappears.

Two files change: the project's `taskgraph.toml` (resources + deny, already in
[`taskgraph.toml`](taskgraph.toml)) and `dev.sh`. **No taskgraph change is needed.**

## 1. Config

```toml
[resources]
simulator = 1                       # the one simulator; capacity 2+ would allow two holders

[agent]
deny = ["xcodebuild", "xcrun"]      # agents must not run these directly
```

`taskgraph run` installs a shim for each `deny` entry first on every agent's `PATH`. Without a lease the
shim prints `xcodebuild is disabled for agents: run it through the project's lease wrapper (e.g. ./dev.sh) …`
and exits 64; with `TASKGRAPH_LEASE` set it `exec`s the real binary. Agents are therefore funnelled through
`./dev.sh`, which takes the lease — the lease is the only way to the simulator.

## 2. Delete the lock, add one helper

```zsh
# --- before: /tmp/elixir-sim.lock, taken with mkdir, kept alive by pid checks -------------
SIM_LOCK=/tmp/elixir-sim.lock
BUILD_LOCK=/tmp/elixir-build.lock
HELD_LOCKS=()
lock_tier()      { … }                 # priority by subcommand
higher_waiting() { … }                 # live-waiter comparison
take_lock()      { … mkdir "$lock" … ps -o command= -p "$holder" | grep -q 'dev\.sh' … }
release_lock()   { rm -rf "$1"; … }
sim_lock()       { take_lock "$SIM_LOCK" simulator; }
trap 'for l in "${HELD_LOCKS[@]}"; do release_lock "$l"; done; …' EXIT INT TERM
```

```zsh
# --- after --------------------------------------------------------------------------------
BUILD_LOCK=/tmp/elixir-build.lock      # unchanged (see "The build lock" below)
HELD_LOCKS=()

# Hold the one simulator for the rest of this run. The wrapper re-runs dev.sh (same subcommand)
# with TASKGRAPH_LEASE=simulator exported and holds the slot until that child exits, so the body
# below never has to lock or release anything; any signal or crash releases it in the kernel.
exec_sim() {  # exec_sim <args…> — re-exec dev.sh under `taskgraph lease simulator`
  [[ -n "${TASKGRAPH_LEASE:-}" ]] && return 0          # already inside the lease: continue
  exec taskgraph lease simulator --task "${TASKGRAPH_TASK:-dev.sh}" -- "$0" "$@"
}
trap 'rm -f "${QUEUE_FILES[@]}"' EXIT INT TERM          # no sim lock to release
```

Each simulator-using branch calls `exec_sim` *after* the work that does not need the simulator, and drops
its `sim_lock` call. `test` keeps compiling outside the lease, exactly as before:

```zsh
  test)
    … routing to host_classes, only=(…) unchanged …
    # Compile first (build lock only), then hold the simulator just for the run.
    if [[ -z "${TASKGRAPH_LEASE:-}" ]]; then
      build_locked -scheme $SCHEME -destination "$DEST" -derivedDataPath build "${only[@]}" build-for-testing -quiet
      exec_sim test ${2:+"$2"}          # ← was `sim_lock`
    fi
    # ---- leased: the simulator is ours until this process exits ----
    xcrun simctl boot "$DEVICE" 2>/dev/null || true
    xcrun simctl bootstatus "$DEVICE" -b >/dev/null
    … xcodebuild … test-without-building … unchanged …
```

```zsh
  run)
    build_locked -scheme $SCHEME -destination "$DEST" -derivedDataPath build build -quiet
    exec_sim run                                   # ← was `sim_lock`
    xcrun simctl boot "$DEVICE" 2>/dev/null || true
    …
  shot)
    exec_sim shot                                  # ← was `sim_lock` (boot not even needed)
    xcrun simctl io "$DEVICE" screenshot screenshot.png
    ;;
  sim-reboot)
    exec_sim sim-reboot                            # ← was `sim_lock`
    xcrun simctl shutdown "$DEVICE" 2>/dev/null || true
    …
```

Notes on the pattern:

- **Recursion guard.** `TASKGRAPH_LEASE=simulator` in the re-exec'd child makes `exec_sim` a no-op there,
  and it is also what makes the `xcodebuild`/`xcrun` shims exec the real binaries.
- **One lease per run, one run per lease.** Keep the `exec_sim` call inside each branch (not at the top of
  the script): the lease is taken once per simulator section, and `build`/`test-host` never touch it.
- **`--task`** is an optional audit label (`taskgraph leases`, the audit log, stall diagnostics). taskgraph
  does not export a task id, so pass `TASKGRAPH_TASK` if your agent shell sets one; otherwise omit it, or
  hard-code `dev.sh` as above.
- **`xcodegen generate`** runs in both the outer and the leased invocation (it is idempotent). Guard it with
  the same `TASKGRAPH_LEASE` test if you want to skip it in the child.

## 3. The build lock (optional)

The build lock is a *different* resource and can stay as it is, or migrate the same way:

```toml
[resources]
simulator = 1
build = 1
```

```zsh
build_locked() { taskgraph lease build --task "${TASKGRAPH_TASK:-dev.sh}" -- "$@"; }
```

## 4. Verify

```sh
# two agents (or two terminals) at once: the second waits, printing one line immediately and every 30 s
./dev.sh shot & ./dev.sh shot
#   waiting for simulator (held by dev.sh for 3 s)
taskgraph leases                       # RESOURCE/SLOT/PID/TASK/AGE/COMMAND of every live holder
tail -F "$TASKGRAPH_STATE/leases/audit.log"   # acquire/release lines
```

Then kill the holder: `kill -9 <wrapper pid>` — the next waiter acquires **immediately** (flock dies with
the process), which is the case the old pid-based lock could not get right. SIGTERM/SIGINT to the wrapper
are forwarded to the leased `dev.sh`, which is allowed to finish its `trap` (kill the test guard process,
write logs) before the slot is released.

While an agent sits in `taskgraph lease`, it is *not* idle: the wrapper announces itself
(`$TASKGRAPH_STATE/procs/`), so the scheduler's stall watchdog exempts that agent instead of killing it
(SPEC §6). Waits longer than `agent.max_lease_secs` (default 1800) are logged as `anomaly` events rather
than kills.
