#!/usr/bin/env python3
"""Call gym create_execution (parallel or sequential).
Known issue: with --parallel > 1, the gym server often returns only ONE HTTP response;
the other connections never get a response. Use --parallel 1 (default) until the
gym server fixes concurrent create_execution response handling.
Usage:
  python3 scripts/call_create_execution.py [--gym-url URL] [--gym-id ID] [--parallel N]
"""
import argparse
import json
import ssl
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_one(base: str, gym_id: str, task_id: str, timeout: int, index: int) -> dict:
    url = f"{base}/api/v1/{gym_id}/tasks/{task_id}"
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        out = json.loads(r.read().decode())
    out["_index"] = index
    out["_task_id"] = task_id
    return out

def main() -> None:
    p = argparse.ArgumentParser(description="Parallel create_execution")
    p.add_argument("--gym-url", default="http://127.0.0.1:5010", help="Gym base URL")
    p.add_argument("--gym-id", default="umetrip", help="Gym ID")
    p.add_argument("--parallel", type=int, default=1, help="Number of parallel creations (default: 1; use 1 until gym server fixes concurrent response)")
    p.add_argument("--timeout", type=int, default=600, help="Timeout per request (seconds)")
    args = p.parse_args()

    base = args.gym_url.rstrip("/")
    ctx = ssl.create_default_context()

    # Get task list, take first N task_ids
    req = urllib.request.Request(f"{base}/api/v1/{args.gym_id}/tasks", method="GET")
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        data = json.loads(r.read().decode())
    tasks = data.get("tasks") or []
    task_ids = []
    for t in tasks:
        tid = t.get("task_id") or t.get("id") or (t.get("meta") or {}).get("id")
        if tid:
            task_ids.append(tid)
    if len(task_ids) < args.parallel:
        print(f"Only {len(task_ids)} tasks available, using {len(task_ids)} parallel", file=sys.stderr, flush=True)
        task_ids = (task_ids * (args.parallel // len(task_ids) + 1))[:args.parallel]
    task_ids = task_ids[: args.parallel]

    if args.parallel > 1:
        print("WARNING: parallel > 1 often hits a gym server bug (only one response returned). Use --parallel 1 if you see only one [i] done.", file=sys.stderr, flush=True)
    print(f"Creating {len(task_ids)} executions (parallel={len(task_ids)}). Timeout={args.timeout}s per request.", file=sys.stderr, flush=True)
    results = []
    done_count = [0]  # list so inner fn can mutate
    total = len(task_ids)

    def heartbeat():
        n = done_count[0]
        if n < total:
            print(f"  (still waiting for {total - n} more ...)", file=sys.stderr, flush=True)
        t = threading.Timer(30.0, heartbeat)
        t.daemon = True
        t.start()

    with ThreadPoolExecutor(max_workers=len(task_ids)) as ex:
        futs = {
            ex.submit(create_one, base, args.gym_id, tid, args.timeout, i): i
            for i, tid in enumerate(task_ids)
        }
        heartbeat()  # start first heartbeat in 30s
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results.append(fut.result())
                done_count[0] += 1
                print(f"  [{i}] done: execution_id={results[-1].get('execution_id')}", file=sys.stderr, flush=True)
            except Exception as e:
                done_count[0] += 1
                print(f"  [{i}] failed: {e}", file=sys.stderr, flush=True)
                results.append({"_index": i, "_error": str(e)})

    # Sort by index and print summary + full results
    results.sort(key=lambda x: x.get("_index", 0))
    print("\n--- summary ---", file=sys.stderr, flush=True)
    for r in results:
        if "_error" in r:
            print(f"  [{r['_index']}] ERROR: {r['_error']}", file=sys.stderr, flush=True)
        else:
            eid = r.get("execution_id", "")
            dev = r.get("device") or {}
            serial = dev.get("device_serial", "")
            print(f"  [{r['_index']}] execution_id={eid} device_serial={serial}", file=sys.stderr, flush=True)

    print("\n--- all responses (JSON) ---")
    out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
