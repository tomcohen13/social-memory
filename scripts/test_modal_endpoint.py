"""Smoke-test the deployed InternVideo2 Modal endpoint.

Usage:
    export API_KEY=<your key>
    python scripts/test_modal_endpoint.py
    python scripts/test_modal_endpoint.py --url <endpoint> --video <mp4 url> --text "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = (
    "https://cyborgpsychologylab--internvideo2-stage2-1b-internvideo2-bba4fc.modal.run"
)
DEFAULT_VIDEO = "https://download.samplelib.com/mp4/sample-5s.mp4"
DEFAULT_TEXT = "a car driving"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--video", default=DEFAULT_VIDEO)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()

    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("ERROR: API_KEY env var not set. Run: export API_KEY=<your key>")
        return 2

    body = json.dumps({"video_url": args.video, "text": args.text}).encode()
    req = urllib.request.Request(
        args.url,
        data=body,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print(f"POST {args.url}")
    print(f"  video: {args.video}")
    print(f"  text:  {args.text!r}")
    print(f"  key:   len={len(api_key)} (first 4 = {api_key[:4]}…)")
    print("waiting for response (cold start can take ~60s)…")

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            payload = json.loads(resp.read())
            elapsed = time.time() - t0
            print(f"OK in {elapsed:.1f}s")
            for k, v in payload.items():
                if isinstance(v, list):
                    norm = sum(x * x for x in v) ** 0.5
                    print(f"  {k}: dim={len(v)}, norm={norm:.4f}, head={v[:3]}")
                else:
                    print(f"  {k}: {v!r}")
            return 0
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        print(f"HTTP {e.code} in {elapsed:.1f}s")
        print(e.read().decode(errors="replace"))
        return 1
    except urllib.error.URLError as e:
        elapsed = time.time() - t0
        print(f"URLError in {elapsed:.1f}s: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
