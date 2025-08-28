#!/usr/bin/env python3
import os
import sys
import csv
import time
import argparse
from typing import Dict, Iterable, Optional, Set, Tuple, List
from urllib.parse import urljoin
from datetime import datetime, timezone
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv, find_dotenv

try:
    from tqdm import tqdm  # progress bars
except Exception:
    tqdm = None

API_TIMEOUT = 30
DEFAULT_PAGE_SIZE = 50
PROJECT_PAGE_SIZE = 50
GROUP_PAGE_SIZE = 100

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

def parse_jira_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
        elif len(s) > 5 and (s[-5] in ["+", "-"]) and s[-2:] != ":":
            s2 = s[:-2] + ":" + s[-2:]
        else:
            s2 = s
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def dt_to_iso(dt: Optional[datetime]) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else ""

def is_http_400(err: Exception) -> bool:
    return isinstance(err, requests.HTTPError) and getattr(err.response, "status_code", None) == 400

def quote_aaid(account_id: str) -> str:
    # Some sites require quoting AAIDs when used in JQL user fields (esp. with `was`).
    return f'"{account_id}"'

class JiraClient:
    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(email, api_token)
        self.session.headers.update({"Accept": "application/json"})
        # caches
        self._group_members_cache: Dict[str, Set[str]] = {}
        self._role_map_cache: Dict[str, Dict[str, str]] = {}
        self._role_members_cache: Dict[Tuple[str, bool], Dict[str, Set[str]]] = {}

    def _get(self, path_or_url: str, params: Optional[Dict] = None):
        url = path_or_url if path_or_url.startswith("http") else urljoin(self.base_url + "/", path_or_url.lstrip("/"))
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
        start_at = 0
        while True:
            data = self._get("/rest/api/3/project/search", params={"startAt": start_at, "maxResults": PROJECT_PAGE_SIZE})
            values = (data or {}).get("values", [])
            for proj in values:
                yield proj
            if not values or data.get("isLast", True):
                break
            start_at += data.get("maxResults", PROJECT_PAGE_SIZE)

    # ----- Users with BROWSE_PROJECTS -----
    def iter_users_with_browse(self, project_key: str, include_inactive: bool = False) -> Iterable[Dict]:
        start_at = 0
        page_size = 1000
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
                if (u.get("accountType") or "").lower() == "app":
                    continue
                if include_inactive or u.get("active", True):
                    yield u
            if count < page_size:
                break
            start_at += page_size

    # ----- Search issues -----
    def search_issues(self, jql: str, start_at: int = 0, max_results: int = DEFAULT_PAGE_SIZE,
                      fields: Optional[Iterable[str]] = None, expand: Optional[Iterable[str]] = None) -> Dict:
        params = {"jql": jql, "startAt": start_at, "maxResults": max_results}
        if fields is not None:
            params["fields"] = ",".join(fields)
        if expand is not None:
            params["expand"] = ",".join(expand)
        return self._get("/rest/api/3/search", params=params) or {}

    # ----- Roles: map + members (expand groups) -----
    def get_project_roles_map(self, project_key: str) -> Dict[str, str]:
        if project_key in self._role_map_cache:
            return self._role_map_cache[project_key]
        data = self._get(f"/rest/api/3/project/{project_key}/role") or {}
        role_map = {name: url for name, url in data.items()}
        self._role_map_cache[project_key] = role_map
        return role_map

    def _expand_group_members(self, group_id: Optional[str], group_name: Optional[str], include_inactive: bool) -> Set[str]:
        cache_key = f"id:{group_id}" if group_id else f"name:{group_name}"
        if cache_key in self._group_members_cache:
            return self._group_members_cache[cache_key]

        params = {
            "startAt": 0,
            "maxResults": GROUP_PAGE_SIZE,
            "includeInactiveUsers": str(include_inactive).lower()
        }
        if group_id:
            params["groupId"] = group_id
        elif group_name:
            params["groupname"] = group_name

        members: Set[str] = set()
        try:
            while True:
                data = self._get("/rest/api/3/group/member", params=params) or {}
                vals = data.get("values", []) or []
                for m in vals:
                    acc = m.get("accountId")
                    if acc:
                        members.add(acc)
                start_at = data.get("startAt", 0)
                max_results = data.get("maxResults", GROUP_PAGE_SIZE)
                total = data.get("total", start_at + len(vals))
                if start_at + max_results >= total:
                    break
                params["startAt"] = start_at + max_results
        except requests.HTTPError as e:
            print(f"[WARN] Could not expand group members for {cache_key}: {e}", file=sys.stderr)

        self._group_members_cache[cache_key] = members
        return members

    def get_project_role_members(self, project_key: str, include_inactive: bool = False) -> Dict[str, Set[str]]:
        cache_key = (project_key, include_inactive)
        if cache_key in self._role_members_cache:
            return self._role_members_cache[cache_key]

        out: Dict[str, Set[str]] = {}
        role_map = self.get_project_roles_map(project_key)
        for role_name, role_url in role_map.items():
            try:
                role = self._get(role_url) or {}
            except requests.HTTPError as e:
                print(f"[WARN] Failed to fetch role '{role_name}' for {project_key}: {e}", file=sys.stderr)
                continue

            members: Set[str] = set()
            for actor in role.get("actors", []) or []:
                atype = actor.get("type")
                if atype == "atlassian-user-role-actor":
                    acc = (actor.get("actorUser") or {}).get("accountId")
                    if acc:
                        members.add(acc)
                elif atype == "atlassian-group-role-actor":
                    grp = actor.get("actorGroup") or {}
                    gid = grp.get("groupId")
                    gname = grp.get("name") or actor.get("name")
                    members |= self._expand_group_members(gid, gname, include_inactive)
            out[role_name] = members

        self._role_members_cache[cache_key] = out
        return out

class AtlassianAdminClient:
    """
    Organization Admin API for last-active dates (per product).
    Requires an org-level API key and the correct org ID.
    """
    def __init__(self, org_id: Optional[str], api_key: Optional[str]):
        self.org_id = (org_id or "").strip()
        self.enabled = bool(self.org_id and api_key)
        self.base_url = "https://api.atlassian.com/admin/v1"
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            })
        self._cache: Dict[str, Optional[datetime]] = {}

    def _get(self, path: str, params: Optional[Dict] = None):
        url = self.base_url + path
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

    def get_last_active_any_product(self, account_id: str) -> Optional[datetime]:
        if not self.enabled:
            return None
        if account_id in self._cache:
            return self._cache[account_id]
        try:
            data = self._get(f"/orgs/{self.org_id}/directory/users/{account_id}/last-active-dates") or {}
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 403:
                print(
                    f"[WARN] last-active lookup failed for {account_id}: 403 Forbidden. "
                    f"Check that the API key belongs to an Org Admin, org ID is correct, "
                    f"and the user is a managed account in that org.", file=sys.stderr
                )
            else:
                print(f"[WARN] last-active lookup failed for {account_id}: {e}", file=sys.stderr)
            self._cache[account_id] = None
            return None

        product_access = ((data.get("data") or {}).get("product_access")) or []
        best = None
        for pa in product_access:
            ts = pa.get("last_active_timestamp") or pa.get("last_active")
            dt = parse_jira_time(ts)
            if dt and (best is None or dt > best):
                best = dt
        self._cache[account_id] = best
        return best

def str2bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1","true","t","yes","y","on"}

def get_latest_user_activity_in_project(jc: JiraClient, project_key: str, account_id: str,
                                        max_issue_scan: int = 500) -> Optional[datetime]:
    """
    Returns the latest UTC timestamp where the user either created an issue in the project
    or authored an update on an issue they were ever assignee/reporter on.
    If a 400 occurs in any JQL call, we log a WARN and SKIP this user (return None).
    """
    aaid = quote_aaid(account_id)

    # 1) newest issue CREATED by the user
    created_dt = None
    try:
        jql_created = f'project = "{project_key}" AND creator = {aaid} ORDER BY created DESC'
        data_created = jc.search_issues(jql_created, start_at=0, max_results=1, fields=["created"])
        issues_c = data_created.get("issues", [])
        if issues_c:
            created_dt = parse_jira_time(((issues_c[0] or {}).get("fields") or {}).get("created"))
    except requests.HTTPError as e:
        if is_http_400(e):
            print(f"[WARN] 400 on created-scan; skipping user {account_id} in {project_key}", file=sys.stderr)
            return None
        raise

    # 2) latest UPDATE authored by the user on issues they were ever assignee/reporter
    jql_updated = (
        f'project = "{project_key}" '
        f'AND (assignee was {aaid} OR reporter was {aaid}) '
        f'ORDER BY updated DESC'
    )

    scanned = 0
    start_at = 0
    best_dt = None

    try:
        while scanned < max_issue_scan:
            data = jc.search_issues(jql_updated, start_at=start_at, max_results=DEFAULT_PAGE_SIZE,
                                    fields=["updated"], expand=["changelog"])
            issues = data.get("issues", [])
            if not issues:
                break
            for issue in issues:
                scanned += 1
                histories = (((issue or {}).get("changelog") or {}).get("histories")) or []
                latest_by_user = None
                for h in histories:
                    if (h.get("author") or {}).get("accountId") == account_id:
                        dt = parse_jira_time(h.get("created"))
                        if dt and (latest_by_user is None or dt > latest_by_user):
                            latest_by_user = dt
                if latest_by_user and (best_dt is None or latest_by_user > best_dt):
                    best_dt = latest_by_user
            if len(issues) < DEFAULT_PAGE_SIZE:
                break
            start_at += DEFAULT_PAGE_SIZE
    except requests.HTTPError as e:
        if is_http_400(e):
            print(f"[WARN] 400 on update-scan; skipping user {account_id} in {project_key}", file=sys.stderr)
            return None
        raise

    # 3) pick later of created vs updated
    if created_dt and (not best_dt or created_dt >= best_dt):
        return created_dt
    return best_dt

def export_contributors(base_url: str, email: str, token: str, out_csv: str,
                        include_inactive: bool = False, try_email_lookup: bool = False,
                        max_issue_scan: int = 500, show_progress: bool = True,
                        org_id: Optional[str] = None, admin_api_key: Optional[str] = None):
    jc = JiraClient(base_url, email, token)
    admin = AtlassianAdminClient(org_id, admin_api_key)

    if not admin.enabled:
        print("[WARN] Org admin API is not configured; 'last active (UTC)' will be blank. "
              "Set ATLASSIAN_ORG_ID and ATLASSIAN_API_KEY in .env", file=sys.stderr)

    projects = list(jc.iter_projects())
    proj_iter = tqdm(projects, desc="Projects", unit="proj") if show_progress and tqdm else projects

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["project name", "project key", "user name", "email",
                  "last worked (UTC)", "last active (UTC)", "Roles"]
        writer.writerow(header)

        for proj in proj_iter:
            pkey = proj.get("key") or proj.get("id")
            pname = proj.get("name") or pkey
            if not pkey:
                continue

            # Preload project role membership (expands groups)
            try:
                role_members = jc.get_project_role_members(pkey, include_inactive=include_inactive)
            except requests.HTTPError as e:
                print(f"[WARN] Failed to load roles for {pkey}: {e}", file=sys.stderr)
                role_members = {}

            users = list(jc.iter_users_with_browse(pkey, include_inactive=include_inactive))
            user_iter = tqdm(users, desc=f"{pkey} users", leave=False, unit="user") if show_progress and tqdm else users

            email_cache: Dict[str, Optional[str]] = {}

            for user in user_iter:
                acc_id = user.get("accountId")
                if not acc_id:
                    continue
                display_name = user.get("displayName") or user.get("name") or ""
                email_address = user.get("emailAddress") or ""
                if not email_address and try_email_lookup:
                    if acc_id not in email_cache:
                        try:
                            email_cache[acc_id] = jc._get("/rest/api/3/user/email", params={"accountId": acc_id}).get("email")
                        except Exception:
                            email_cache[acc_id] = None
                    email_address = email_cache.get(acc_id) or ""

                last_worked_dt = get_latest_user_activity_in_project(
                    jc, pkey, acc_id, max_issue_scan=max_issue_scan
                )
                # if 400 occurred, function returns None; we still write the row

                last_active_dt = admin.get_last_active_any_product(acc_id)

                user_roles = sorted([rname for rname, members in (role_members or {}).items() if acc_id in members])
                roles_str = "; ".join(user_roles)

                writer.writerow([
                    pname, pkey, display_name, email_address,
                    dt_to_iso(last_worked_dt), dt_to_iso(last_active_dt), roles_str
                ])

def main():
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(description="Export Jira Cloud contributors with latest activity, last login, and project roles (.env supported).")
    parser.add_argument("--base-url", default=os.environ.get("JIRA_BASE_URL"))
    parser.add_argument("--email", default=os.environ.get("JIRA_EMAIL"))
    parser.add_argument("--api-token", default=os.environ.get("JIRA_API_TOKEN"))
    parser.add_argument("-o", "--out", default=os.environ.get("OUTPUT_CONTRIBUTORS", os.environ.get("OUT", "jira_contributors.csv")))
    parser.add_argument("--include-inactive", action="store_true", default=str2bool(os.environ.get("INCLUDE_INACTIVE", "false")))
    parser.add_argument("--try-email-lookup", action="store_true", default=str2bool(os.environ.get("TRY_EMAIL_LOOKUP", "false")))
    parser.add_argument("--max-issue-scan", type=int, default=int(os.environ.get("MAX_ISSUE_SCAN", "500")))
    parser.add_argument("--no-progress", action="store_true", default=str2bool(os.environ.get("NO_PROGRESS", "false")))
    parser.add_argument("--org-id", default=os.environ.get("ATLASSIAN_ORG_ID"))
    parser.add_argument("--org-api-key", default=os.environ.get("ATLASSIAN_API_KEY"))
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

    show_progress = (not args.no_progress)
    if tqdm is None and show_progress:
        print("[WARN] tqdm not installed; progress bars disabled. Install with: pip install tqdm", file=sys.stderr)
        show_progress = False

    try:
        export_contributors(args.base_url, args.email, args.api_token, args.out,
                            include_inactive=args.include_inactive, try_email_lookup=args.try_email_lookup,
                            max_issue_scan=args.max_issue_scan, show_progress=show_progress,
                            org_id=args.org_id, admin_api_key=args.org_api_key)
        print(f"Done. Wrote: {args.out}")
    except requests.HTTPError as e:
        print(f"HTTP error: {e} – response: {getattr(e, 'response', None) and e.response.text}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
