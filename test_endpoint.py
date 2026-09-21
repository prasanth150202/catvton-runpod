"""Send a try-on job to the RunPod endpoint (or poll an existing one) and save result.png.

Usage:
    export RUNPOD_API_KEY=...
    python3 test_endpoint.py                 # submit test_input.json and wait
    python3 test_endpoint.py <job-id>        # wait for an already-submitted job
"""
import base64
import json
import os
import sys
import time
import urllib.request

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "ia71h1ng6xyqig")
API = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}


def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, headers=HEADERS, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


if len(sys.argv) > 1:
    job_id = sys.argv[1]
else:
    with open("test_input.json") as f:
        job_id = call("/run", json.load(f))["id"]
print("job:", job_id)

start = time.time()
while True:
    r = call(f"/status/{job_id}")
    status = r.get("status")
    print(f"[{int(time.time() - start):>4}s] {status}")
    if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        break
    time.sleep(10)

out = r.get("output") or {}
if status == "COMPLETED" and "image" in out:
    with open("result.png", "wb") as f:
        f.write(base64.b64decode(out["image"]))
    print("saved result.png (seed %s)" % out.get("seed"))
else:
    print(json.dumps(r, indent=2)[:3000])
    sys.exit(1)
