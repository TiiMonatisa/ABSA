#!/usr/bin/env python3
import os
import sys
import csv
import time
import argparse
from collections import defaultdict
from typing import Dict, Iterable, Optional, Set
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv, find_dotenv

API_TIMEOUT = 30
DEFAULT_PAGE_SIZE = 1000        # user/permission/search window
PROJECT_PAGE_SIZE = 50
GROUP_PAGE_SIZE = 100

INDICATOR = "Y"  # what to write when user has a role

def backoff_sleep(resp, attempt):
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            sleep_s = int(retry_after)
        except ValueError:
            sleep_s = 2 ** attempt
    else:
        sleep_s = 2 ** attempt
    time.sleep(min(sleep_s, 60))

class JiraClient:
    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(email, api_token)
        self.session.headers.update({"Accept": "application/json"})
        # cache
        self._group_members_cache: Dict[str, Set[str]] = {}

    def _get(self, path_or_url: str, params: Optional[Dict] = None):
        # Accept either absolute URLs (role "self") or relative paths
        if path_or_url.startswith("http"):
            url = path_or_url
        else:
            url = urljoin(self.base_url + "/", path_or_url.lstrip("/"))
        for attempt in range(6):
            resp = self.session.get(url, params=params, timeout=API_TIMEOUT)
            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                backoff_sleep(resp, attempt)
                continue
            resp.raise_for_status()
            if resp.text and resp.headers.get("Content-Type", "").startswith("application/json"):
                return resp.json()
            return None
        raise RuntimeError(f"GET {url} failed after retries")

    # ----- Projects -----
    def iter_projects(self) -> Iterable[Dict]:
        """Paginated project search; returns projects visible to the caller."""
        start_at = 0
        while True:
            data = self._get("/rest/api/3/project/search",
                             params={"startAt": start_at, "maxResults": PROJECT_PAGE_SIZE})
            values = (data or {}).get("values", [])
            for proj in values:
                yield proj
            if not values or data.get("isLast", True):
                break
            start_at += data.get("maxResults", PROJECT_PAGE_SIZE)

    # ----- Users with access (BROWSE_PROJECTS) -----
    def iter_users_with_browse(self, project_key: str, include_inactive: bool = False) -> Iterable[Dict]:
        """
        Lists users who can browse the project (BROWSE_PROJECTS).
        EXCLUDES add-on/app users (accountType == 'app').
        Note: Jira user-search endpoints only return results within the first 1000 users window.
        """
        start_at = 0
        page_size = DEFAULT_PAGE_SIZE
        while start_at < 1000:
            users = self._get(
                "/rest/api/3/user/permission/search",
                params={
                    "projectKey": project_key,
                    "permissions": "BROWSE_PROJECTS",
                    "startAt": start_at,
                    "maxResults": page_size,
                },
            )
            if not users:
                break
            count = 0
            for u in users:
                count += 1
                # Skip add-on/app users
                if (u.get("accountType") or "").lower() == "app":
                    continue
                if include_inactive or u.get("active", True):
                    yield u
            if count < page_size:
                break
            start_at += page_size

    # ----- Role discovery & expansion -----
    def get_project_roles_map(self, project_key: str) -> Dict[str, str]:
        """Returns {role_name: role_url} for a project."""
        data = self._get(f"/rest/api/3/project/{project_key}/role")
        return {name: url for name, url in (data or {}).items()}

    def _expand_group_members(self, group_id: Optional[str], group_name: Optional[str],
                              include_inactive: bool) -> Set[str]:
        """
        Returns a set of accountIds in the group (cached). Tries groupId first, then groupname.
        """
        cache_key = f"id:{group_id}" if group_id else f"name:{group_name}"
        if cache_key in self._group_members_cache:
            return self._group_members_cache[cache_key]

        params = {"startAt": 0, "maxResults": GROUP_PAGE_SIZE, "includeInactiveUsers": str(include_inactive).lower()}
        if group_id:
            params["groupId"] = group_id
        elif group_name:
            params["groupname"] = group_name

        members: Set[str] = set()
        try:
            while True:
                data = self._get("/rest/api/3/group/member", params=params)
                if not data:
                    break
                for m in data.get("values", []):
                    acc = m.get("accountId")
                    if acc:
                        members.add(acc)
                start_at = data.get("startAt", 0)
                max_results = data.get("maxResults", GROUP_PAGE_SIZE)
                total = data.get("total", start_at + len(data.get("values", [])))
                if start_at + max_results >= total:
                    break
                params["startAt"] = start_at + max_results
        except requests.HTTPError as e:
            print(f"[WARN] Could not expand group members for {cache_key}: {e}", file=sys.stderr)

        self._group_members_cache[cache_key] = members
        return members

    def get_project_role_members(self, project_key: str, include_inactive: bool = False) -> Dict[str, Set[str]]:
        """
        For a project, returns {role_name: set(accountIds)} including user actors and expanded group actors.
        """
        out: Dict[str, Set[str]] = defaultdict(set)
        role_map = self.get_project_roles_map(project_key)
        for role_name, role_url in role_map.items():
            try:
                role = self._get(role_url)
            except requests.HTTPError as e:
                print(f"[WARN] Failed to fetch role '{role_name}' for {project_key}: {e}", file=sys.stderr)
                continue
            for actor in role.get("actors", []):
                atype = actor.get("type")
                if atype == "atlassian-user-role-actor":
                    acc = (actor.get("actorUser") or {}).get("accountId")
                    if acc:
                        out[role_name].add(acc)
                elif atype == "atlassian-group-role-actor":
                    grp = actor.get("actorGroup") or {}
                    gid = grp.get("groupId")
                    gname = grp.get("name") or actor.get("name")
                    members = self._expand_group_members(gid, gname, include_inactive)
                    out[role_name].update(members)
                # ignore add-on role actors etc.
        return out

    # ----- Email lookup (optional; may require approval) -----
    def get_user_email_via_account(self, account_id: str) -> Optional[str]:
        try:
            data = self._get("/rest/api/3/user/email", params={"accountId": account_id})
            if isinstance(data, dict) and "email" in data:
                return data["email"]
        except requests.HTTPError:
            pass
        return None

def str2bool(v: Optional[str], default: bool = False) -> bool:
    if v is None: return default
    return str(v).strip().lower() in {"1","true","t","yes","y","on"}

def export_access_with_roles(base_url: str, email: str, token: str, out_csv: str,
                             include_inactive: bool = False, try_email_lookup: bool = False):
    jc = JiraClient(base_url, email, token)

    # First pass: gather roles per project (fully expanded) and the global set of role names
    print("[INFO] Discovering roles & members per project...", file=sys.stderr)
    project_keys: list[str] = []
    roles_by_project: Dict[str, Dict[str, Set[str]]] = {}
    all_role_names: Set[str] = set()

    for proj in jc.iter_projects():
        pkey = proj.get("key") or proj.get("id")
        if not pkey:
            continue
        project_keys.append(pkey)
        role_members = jc.get_project_role_members(pkey, include_inactive=include_inactive)
        roles_by_project[pkey] = role_members
        all_role_names.update(role_members.keys())

    role_columns = sorted(all_role_names)

    # Prepare email cache for optional lookup
    email_cache: Dict[str, Optional[str]] = {}

    # Second pass: export users-with-access + role indicators
    print("[INFO] Writing CSV...", file=sys.stderr)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["project key", "User name", "email"] + role_columns
        writer.writerow(header)

        for pkey in project_keys:
            role_members = roles_by_project.get(pkey, {})
            seen_accounts: Set[str] = set()

            for user in jc.iter_users_with_browse(pkey, include_inactive=include_inactive):
                acc_id = user.get("accountId")
                if not acc_id or acc_id in seen_accounts:
                    continue
                seen_accounts.add(acc_id)

                display_name = user.get("displayName") or user.get("name") or ""
                email_address = user.get("emailAddress") or ""

                if not email_address and try_email_lookup and acc_id:
                    if acc_id not in email_cache:
                        email_cache[acc_id] = jc.get_user_email_via_account(acc_id)
                    email_address = email_cache.get(acc_id) or ""

                row = [pkey, display_name, email_address]
                for role_name in role_columns:
                    row.append(INDICATOR if acc_id in role_members.get(role_name, set()) else "")
                writer.writerow(row)

def main():
    load_dotenv(find_dotenv(usecwd=True))
    parser = argparse.ArgumentParser(description="Export Jira Cloud project access + roles to CSV (.env supported).")
    parser.add_argument("--base-url", default=os.environ.get("JIRA_BASE_URL"))
    parser.add_argument("--email", default=os.environ.get("JIRA_EMAIL"))
    parser.add_argument("--api-token", default=os.environ.get("JIRA_API_TOKEN"))
    parser.add_argument("-o", "--out", default=os.environ.get("OUT", "jira_project_access_with_roles.csv"))
    parser.add_argument("--include-inactive", action="store_true",
                        default=str2bool(os.environ.get("INCLUDE_INACTIVE", "false")))
    parser.add_argument("--try-email-lookup", action="store_true",
                        default=str2bool(os.environ.get("TRY_EMAIL_LOOKUP", "false")))
    args = parser.parse_args()

    missing = [k for k, v in {
        "JIRA_BASE_URL": args.base_url,
        "JIRA_EMAIL": args.email,
        "JIRA_API_TOKEN": args.api_token,
    }.items() if not v]
    if missing:
        print("Missing required configuration: " + ", ".join(missing), file=sys.stderr)
        print("Tip: set them in .env or pass as CLI flags.", file=sys.stderr)
        sys.exit(2)

    try:
        export_access_with_roles(args.base_url, args.email, args.api_token, args.out,
                                 include_inactive=args.include_inactive,
                                 try_email_lookup=args.try_email_lookup)
        print(f"Done. Wrote: {args.out}")
    except requests.HTTPError as e:
        print(f"HTTP error: {e} – response: {getattr(e, 'response', None) and e.response.text}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
