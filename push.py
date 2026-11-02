#!/usr/bin/env python3
"""Surgical tree-push for snowtest (correct PATCH /git/refs/ endpoint).
Usage: push.py <message> <local-root> <path1> [path2 ...]
A path starting with "!" deletes that path."""
import base64
import json
import os
import sys
import urllib.request

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response, ensure_allowed_url

API = "https://api.github.com"
ALLOWED = ["api.github.com"]
OWNER, REPO, BRANCH = "a-user-of-python", "snowtest", "main"


def api(method, path, data=None):
    url = API + path
    ensure_allowed_url(url, ALLOWED)
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body:
        req.add_header("Content-Type", "application/json")
    add_surrogate_to_request(req, "custom.github", allowed_hosts=ALLOWED)
    return read_json_response(urllib.request.urlopen(req))


def main():
    message, root = sys.argv[1], sys.argv[2]
    paths = sys.argv[3:]
    ref = api("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    head = ref["object"]["sha"]
    print(f"Remote {BRANCH}: {head[:12]}")
    head_tree = api("GET", f"/repos/{OWNER}/{REPO}/git/commits/{head}")["tree"]["sha"]

    entries = []
    for p in paths:
        if p.startswith("!"):
            entries.append({"path": p[1:], "mode": "100644", "type": "blob", "sha": None})
            print(f"  delete {p[1:]}")
            continue
        with open(os.path.join(root, p), "rb") as f:
            content = f.read()
        blob = api("POST", f"/repos/{OWNER}/{REPO}/git/blobs",
                   {"content": base64.b64encode(content).decode(), "encoding": "base64"})
        entries.append({"path": p, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print(f"  blob {p}: {blob['sha'][:12]} ({len(content)} bytes)")

    tree = api("POST", f"/repos/{OWNER}/{REPO}/git/trees",
               {"base_tree": head_tree, "tree": entries})
    print(f"Tree: {tree['sha'][:12]}")
    commit = api("POST", f"/repos/{OWNER}/{REPO}/git/commits",
                 {"message": message, "tree": tree["sha"], "parents": [head]})
    print(f"Commit: {commit['sha'][:12]}")

    ref2 = api("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    if ref2["object"]["sha"] != head:
        sys.exit(f"REF MOVED during push; refusing to PATCH.")
    api("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}", {"sha": commit["sha"]})
    print(f"Pushed {BRANCH} -> {commit['sha'][:12]}")

    v = api("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    vc = api("GET", f"/repos/{OWNER}/{REPO}/git/commits/{v['object']['sha']}")
    assert vc["tree"]["sha"] == tree["sha"], "TREE MISMATCH after push!"
    print(f"Verified: remote tree == pushed tree ({tree['sha'][:12]})")


if __name__ == "__main__":
    main()
