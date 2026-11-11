#!/usr/bin/env python3
"""
Automatically parse .github/ISSUES.md and create all issues on GitHub.
Supports execution via GitHub CLI (gh) or direct GitHub REST API with a token.
"""

import os
import re
import sys
import json
import subprocess
import urllib.request
import urllib.error
from typing import List, Dict, Any

REPO = "EpitechPGE3-2026/G-CNA-501-MAR-5-1-mytorch-1"
GH_CLI_PATH = r"C:\Users\Esteban\AppData\Local\Programs\gh\bin\gh.exe"


def get_gh_executable() -> str:
    if os.path.exists(GH_CLI_PATH):
        return GH_CLI_PATH
    return "gh"


def parse_issues_markdown(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    issues = []
    current_milestone = None

    # Split by lines
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for Milestone header: ## 📌 Milestone 1 : ...
        if line.startswith("## ") and "Milestone" in line:
            current_milestone = line.replace("## ", "").replace("📌 ", "").strip()
            i += 1
            continue

        # Check for Issue header: ### Issue #1 : [CORE] ...
        if line.startswith("### Issue"):
            title_match = re.search(r"### Issue #\d+ :\s*(.*)", line)
            title = title_match.group(1).strip() if title_match else line

            labels = []
            description_lines = []
            tasks_lines = []
            criteria_lines = []
            current_section = None

            i += 1
            while i < len(lines):
                cur_line = lines[i]
                stripped = cur_line.strip()

                if stripped.startswith("### Issue") or (stripped.startswith("## ") and "Milestone" in stripped):
                    break

                if stripped.startswith("- **Labels** :"):
                    lbl_match = re.search(r"\*\*(?:Labels|Label)\*\*\s*:\s*(.*)", stripped)
                    if lbl_match:
                        raw_labels = lbl_match.group(1).replace("`", "")
                        labels = [l.strip() for l in raw_labels.split(",") if l.strip()]
                    current_section = None
                elif stripped.startswith("- **Description** :"):
                    current_section = "desc"
                elif stripped.startswith("- **Tâches** :"):
                    current_section = "tasks"
                elif stripped.startswith("- **Critères d'acceptation** :"):
                    current_section = "criteria"
                elif stripped.startswith("---"):
                    pass
                else:
                    if current_section == "desc":
                        description_lines.append(cur_line)
                    elif current_section == "tasks":
                        tasks_lines.append(cur_line)
                    elif current_section == "criteria":
                        criteria_lines.append(cur_line)
                i += 1

            # Format body
            body_parts = []
            if current_milestone:
                body_parts.append(f"**Milestone** : {current_milestone}\n")
            if description_lines:
                body_parts.append("### Description\n" + "\n".join(description_lines).strip() + "\n")
            if tasks_lines:
                body_parts.append("### Tâches techniques\n" + "\n".join(tasks_lines).strip() + "\n")
            if criteria_lines:
                body_parts.append("### Critères d'acceptation\n" + "\n".join(criteria_lines).strip() + "\n")

            body = "\n".join(body_parts).strip()

            issues.append({
                "title": title,
                "labels": labels,
                "milestone": current_milestone,
                "body": body,
            })
            continue

        i += 1

    return issues


def create_via_gh_cli(issues: List[Dict[str, Any]]) -> bool:
    gh = get_gh_executable()

    # Check auth
    res = subprocess.run([gh, "auth", "status"], capture_output=True, text=True)
    if res.returncode != 0:
        print("[!] GitHub CLI is not authenticated. Please run 'gh auth login' or provide a token.")
        return False

    print(f"Creating {len(issues)} issues via GitHub CLI on {REPO}...")
    for idx, iss in enumerate(issues, 1):
        cmd = [
            gh, "issue", "create",
            "--repo", REPO,
            "--title", iss["title"],
            "--body", iss["body"],
        ]
        for lbl in iss["labels"]:
            cmd.extend(["--label", lbl])

        print(f"[{idx}/{len(issues)}] Creating: {iss['title']}...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  -> {r.stdout.strip()}")
        else:
            print(f"  [ERROR] {r.stderr.strip()}")
    return True


def create_via_api(issues: List[Dict[str, Any]], token: str):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "MyTorch-Issue-Creator",
        "Content-Type": "application/json",
    }

    # Fetch existing issues to avoid duplicates
    existing_titles = set()
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/issues?state=all&per_page=100", headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for it in data:
                existing_titles.add(it.get("title", ""))
    except Exception as e:
        print(f"Warning: could not fetch existing issues ({e}). Proceeding...")

    print(f"Creating {len(issues)} issues on {REPO} via GitHub API...")
    for idx, iss in enumerate(issues, 1):
        if iss["title"] in existing_titles:
            print(f"[{idx}/{len(issues)}] Skipped (already exists): {iss['title']}")
            continue

        payload = {
            "title": iss["title"],
            "body": iss["body"],
            "labels": iss["labels"],
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/issues", data=data_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                print(f"[{idx}/{len(issues)}] Created #{res_data.get('number')}: {iss['title']} ({res_data.get('html_url')})")
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            print(f"[{idx}/{len(issues)}] Failed to create '{iss['title']}': HTTP {e.code} - {err_msg}")
        except Exception as e:
            print(f"[{idx}/{len(issues)}] Failed to create '{iss['title']}': {e}")


def main():
    issues_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".github", "ISSUES.md")
    if not os.path.exists(issues_file):
        print(f"Error: {issues_file} not found.")
        sys.exit(1)

    issues = parse_issues_markdown(issues_file)
    print(f"Parsed {len(issues)} issues from {issues_file}.")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if len(sys.argv) > 1:
        token = sys.argv[1]

    if token:
        create_via_api(issues, token)
    else:
        success = create_via_gh_cli(issues)
        if not success:
            print("\nUsage options to create issues:")
            print("1. Authenticate with GitHub CLI:")
            print("   gh auth login")
            print("   python tools/create_github_issues.py")
            print("2. Or pass a GitHub Personal Access Token (PAT):")
            print("   python tools/create_github_issues.py <YOUR_GITHUB_TOKEN>")
            print("   (ou définir la variable d'environnement GITHUB_TOKEN)")


if __name__ == "__main__":
    main()
