#!/usr/bin/env python3
"""SnowTest CI helper: runs / jobs / log / artifacts / download."""
import json
import sys
import urllib.request
import urllib.error
import zipfile
import io
import os

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response, ensure_allowed_url

API = "https://api.github.com"
ALLOWED = ["api.github.com"]
OWNER, REPO = "a-user-of-python", "snowtest"


def api(method, path):
    url = API + path
    ensure_allowed_url(url, ALLOWED)
    req = urllib.request.Request(url, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    add_surrogate_to_request(req, "custom.github", allowed_hosts=ALLOWED)
    return read_json_response(urllib.request.urlopen(req))


def fetch_logs(kind, ident):
    url = f"{API}/repos/{OWNER}/{REPO}/actions/{kind}/{ident}/logs"
    ensure_allowed_url(url, ALLOWED)
    req = urllib.request.Request(url, method="GET")
    add_surrogate_to_request(req, "custom.github", allowed_hosts=ALLOWED)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(req)
        raise SystemExit("expected redirect from logs endpoint")
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
        return urllib.request.urlopen(loc).read()


def main():
    cmd = sys.argv[1]
    if cmd == "runs":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        r = api("GET", f"/repos/{OWNER}/{REPO}/actions/runs?per_page={n}")
        for run in r["workflow_runs"]:
            print(f"{run['id']}  {run['status']:12} {run.get('conclusion') or '-':10} "
                  f"{run['head_branch']}  {run['created_at']}  {run['name']}")
    elif cmd == "jobs":
        r = api("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{sys.argv[2]}/jobs")
        for j in r["jobs"]:
            print(f"{j['id']}  {j['name']:12} {j['status']:12} {j.get('conclusion') or '-'}")
            for s in j.get("steps", []):
                print(f"    - {s['name']:44} {s['status']:12} {s.get('conclusion') or '-'}")
    elif cmd == "log":
        tail = int(sys.argv[3]) if len(sys.argv) > 3 else 80
        data = fetch_logs("jobs", sys.argv[2]).decode("utf-8", "replace")
        lines = data.splitlines()
        print(f"--- {len(lines)} lines total, last {tail} ---")
        print("\n".join(lines[-tail:]))
    elif cmd == "errors":
        data = fetch_logs("jobs", sys.argv[2]).decode("utf-8", "replace")
        lines = data.splitlines()
        seen = set()
        for i, l in enumerate(lines):
            if "error:" in l or "BUILD FAILED" in l or "Undefined symbol" in l or "ld:" in l:
                key = l[26:140] if len(l) > 26 else l
                if key not in seen:
                    seen.add(key)
                    print(i, l[:300])
    elif cmd == "artifacts":
        r = api("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{sys.argv[2]}/artifacts")
        for a in r["artifacts"]:
            print(f"{a['id']}  {a['name']}  {a['size_in_bytes']} bytes  expired={a['expired']}")
    elif cmd == "download":
        run_id, name, dest = sys.argv[2], sys.argv[3], sys.argv[4]
        r = api("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{run_id}/artifacts")
        art = next((a for a in r["artifacts"] if a["name"] == name), None)
        if not art:
            sys.exit(f"artifact {name} not found")
        url = f"{API}/repos/{OWNER}/{REPO}/actions/artifacts/{art['id']}/zip"
        ensure_allowed_url(url, ALLOWED)
        req = urllib.request.Request(url, method="GET")
        add_surrogate_to_request(req, "custom.github", allowed_hosts=ALLOWED)

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        opener = urllib.request.build_opener(NoRedirect)
        try:
            opener.open(req)
            sys.exit("expected redirect")
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location")
            if not loc:
                raise
            blob = urllib.request.urlopen(loc).read()
        os.makedirs(dest, exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(blob))
        z.extractall(dest)
        for info in z.infolist():
            print(f"{info.filename}  {info.file_size} bytes")
        print(f"extracted to {dest}")
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
