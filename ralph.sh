#!/bin/zsh
# Ralph loop for taskgraph: one omp run per PLAN.md task; unit tests gate; one commit per task for review.
#   ./ralph.sh [N]   run up to N tasks (default: all), stopping on the first failure
set -uo pipefail
cd "${0:A:h}"
source ./tools/run-omp.sh
mkdir -p logs
N=${1:-100}
for i in $(seq 1 $N); do
  task=$(grep -m1 '^- \[ \]' PLAN.md | sed -E 's/^- \[ \] ([A-Z]+[0-9]+[a-z]?).*/\1/')
  if [[ -z "$task" ]]; then echo "All tasks done."; exit 0; fi
  log="logs/$task-$(date +%H%M%S).log"
  echo "$(date +%T) === start $task (log: $log)"
  run_omp "$task" "$log"
  [[ $OMP_REASON != done ]] && echo "$(date +%T) !!! omp $OMP_REASON on $task"
  if grep -q "^- \[ \] $task" PLAN.md; then echo "$(date +%T) !!! blocked $task: not marked done — review"; exit 1; fi
  if ! python3 -m unittest discover -s tests -q >> "$log" 2>&1; then echo "$(date +%T) !!! blocked $task: tests fail — review"; exit 1; fi
  title=$(grep -m1 "^- \[x\] $task" PLAN.md | sed -E 's/^- \[x\] [A-Z]+[0-9]+[a-z]? //' | tr -d '`' | cut -c1-60)
  git add -A && git commit -q -m "$task: $title

Implemented by omp (ralph loop); planned and reviewed with Claude.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017nt17ajhN2xY9pBawuHqhz"
  echo "$(date +%T) --- merged $task ($(git log -1 --format=%h))"
done
