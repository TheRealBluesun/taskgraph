# Shared omp runner for ralph loops (sourced). Provides: run_omp <task> <budget> <log>  → sets OMP_RC, OMP_REASON
# Traces: full JSON event stream in <log>; one line per run appended to logs/timing.csv.
MODEL=${RALPH_MODEL:-deepseek/deepseek-v4-flash}
FALLBACK_MODEL=${RALPH_FALLBACK_MODEL:-}
STALL_SECS=${RALPH_STALL_SECS:-480}   # no events for 8 min = stuck
SAFETY_CAP=${RALPH_SAFETY_CAP:-120m}   # runaway guard only, not a budget

_omp_once() {   # model task log
  local model=$1 task=$2 log=$3 start=$(date +%s) shown=0
  omp -p --mode json --auto-approve --no-session --model "$model" --thinking high --max-time "$SAFETY_CAP" --config "$PWD/tools/omp-agent.yml" \
      "@${RALPH_PROMPT:-PROMPT.md}" < /dev/null > "$log" 2>&1 &
  local pid=$! last_size=-1 last_change=$(date +%s)
  OMP_REASON=done
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    # Only before the first tool call and only at the top of the trace: an agent reading this file puts the
    # phrase into its own trace (that killed a working FL8 agent once).
    if (( shown == 0 )) && head -n 20 "$log" | grep -q '^Still starting after'; then kill $pid; OMP_REASON=startup-hang; break; fi
    local calls=$(grep -c '"type":"tool_execution_start"' "$log")
    if (( calls > shown )); then
      grep '"type":"tool_execution_start"' "$log" | tail -n $((calls - shown)) | python3 -c "
import sys,json,time
for l in sys.stdin:
    d=json.loads(l); a=d.get('args') or {}
    s=a.get('command') or a.get('path') or a.get('file_path') or json.dumps(a)
    print(time.strftime('%H:%M:%S'),'  ▶',d.get('toolName'),str(s).replace(chr(10),' ')[:140], flush=True)"
      shown=$calls
    fi
    local size=$(stat -f %z "$log" 2>/dev/null || echo 0) now=$(date +%s)
    if (( size != last_size )); then last_size=$size; last_change=$now
    elif (( now - last_change > STALL_SECS )); then kill $pid; OMP_REASON=stalled; break; fi
  done
  wait $pid; OMP_RC=$?
  [[ $OMP_REASON == done && $OMP_RC != 0 ]] && OMP_REASON="exit-$OMP_RC"
  if grep -qiE '"(error|errorMessage)".{0,200}(quota|insufficient|rate.?limit|429|usage limit|exceeded)' "$log"; then OMP_REASON=quota; fi
  local secs=$(( $(date +%s) - start ))
  local tools=$(grep -c '"type":"tool_execution_start"' "$log")
  mkdir -p logs; [[ -f logs/timing.csv ]] || echo "date,task,model,seconds,tool_calls,reason" > logs/timing.csv
  echo "$(date +%F\ %T),$task,$model,$secs,$tools,$OMP_REASON" >> logs/timing.csv
  echo "    omp[$model] $OMP_REASON in ${secs}s, $tools tool calls"
}

run_omp() {     # task log
  _omp_once "$MODEL" "$1" "$2"
  if [[ $OMP_REASON == quota && "$FALLBACK_MODEL" != "$MODEL" ]]; then
    echo "    quota exhausted on $MODEL — falling back to $FALLBACK_MODEL"
    _omp_once "$FALLBACK_MODEL" "$1" "${2%.log}-fallback.log"
  fi
}
