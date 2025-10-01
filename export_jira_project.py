#!/usr/bin/env python3
import os
import sys
import csv
import time
import argparse
from typing import Dict, Iterable, Optional, Set, Tuple
from urllib.parse import urljoin
from datetime import datetime, timezone
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv, find_dotenv

try:
    from tqdm import tqdm  # progress bars
except Exception:
    tqdm = None

# -----------------------
# Helpers
# -----------------------

def str2bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}

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
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def dt_to_iso(dt: Optional[datetime]) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else ""

def quote_aaid(account_id: str) -> str:
    # Account IDs can contain ':'; safe for path segments
    return requests.utils.requote_uri(account_id)

# -----------------------
# Jira Client
# -----------------------

PROJECT_PAGE_SIZE = 50
GROUP_PAGE_SIZE = 100

class JiraClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        token: str,
        verbose: bool = False,
        ca_bundle: Optional[str] = None,
        client_cert: Optional[str] = None,
        client_key: Optional[str] = None,
        insecure: bool = False,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(email, token)
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "export-jira-contributors/1.0"
        })
        # ---- TLS / cert handling ----
        # verify can be: True/False or path to CA bundle (PEM)
        self.session.verify = False if insecure else (ca_bundle if ca_bundle else True)
        # client cert: tuple(cert, key) or single file (combined PEM)
        if client_cert and client_key:
            self.session.cert = (client_cert, client_key)
        elif client_cert:
            self.session.cert = client_cert
        # ------------------------------
        self.verbose = verbose

    def _get(self, path: str, params: Optional[dict] = None):
        url = urljoin(self.base_url, path.strip("/"))
        if self.verbose:
            print(f"[DEBUG] GET {url} params={params}")
        resp = self.session.get(url, params=params)
        if resp.status_code == 429:
            attempt = 1
            while resp.status_code == 429 and attempt <= 6:
                if self.verbose:
                    print(f"[DEBUG] 429 from {url}, backing off (attempt {attempt})")
                backoff_sleep(resp, attempt)
                resp = self.session.get(url, params=params)
                attempt += 1
        # Helpful debug before raising
        if self.verbose:
            try:
                print(f"[DEBUG] {url} -> {resp.status_code} {resp.text[:200]}...")
            except Exception:
                pass
        resp.raise_for_status()
        return resp.json()

    def iter_projects(self) -> Iterable[dict]:
        start_at = 0
        while True:
            data = self._get("/rest/api/3/project/search",
                             params={"startAt": start_at, "maxResults": PROJECT_PAGE_SIZE})
            values = data.get("values") or []
            if not values:
                break
            for p in values:
                yield p
            if len(values) < PROJECT_PAGE_SIZE:
                break
            start_at += PROJECT_PAGE_SIZE

    def get_project_role_members(self, project_key: str) -> Dict[str, Set[str]]:
        roles = self._get(f"/rest/api/3/project/{project_key}/role")
        out: Dict[str, Set[str]] = {}
        for role_name, role_url in roles.items():
            try:
                r = self.session.get(role_url)
                if r.status_code == 429:
                    attempt = 1
                    while r.status_code == 429 and attempt <= 6:
                        backoff_sleep(r, attempt)
                        r = self.session.get(role_url)
                        attempt += 1
                r.raise_for_status()
                actors = r.json().get("actors") or []
                ids: Set[str] = set()
                for m in actors:
                    if m.get("type") == "atlassian-user-role-actor":
                        if m.get("actorUser") and m["actorUser"].get("accountId"):
                            ids.add(m["actorUser"]["accountId"])
                out[role_name] = ids
            except requests.HTTPError as e:
                print(f"[WARN] Failed to load role {role_name} in {project_key}: {e}")
                continue
        return out

    def iter_users_with_browse(self, project_key: str, include_inactive: bool = False) -> Iterable[dict]:
        # This enumerates site users; substitute with a more precise per-project API if available in your plan.
        start_at = 0
        while True:
            data = self._get("/rest/api/3/users/search",
                             params={"startAt": start_at, "maxResults": GROUP_PAGE_SIZE})
            if not isinstance(data, list) or not data:
                break
            for u in data:
                if include_inactive or u.get("active", True):
                    yield u
            if len(data) < GROUP_PAGE_SIZE:
                break
            start_at += GROUP_PAGE_SIZE

# -----------------------
# Atlassian Admin API client (last active)
# -----------------------

class AtlassianAdminClient:
    def __init__(
        self,
        org_id: Optional[str],
        api_key: Optional[str],
        verbose: bool = False,
        ca_bundle: Optional[str] = None,
        client_cert: Optional[str] = None,
        client_key: Optional[str] = None,
        insecure: bool = False,
    ):
        self.org_id = org_id
        self.api_key = api_key
        self.enabled = bool(org_id and api_key)
        self.verbose = verbose
        self.session = requests.Session()
        self._cache: Dict[str, Optional[datetime]] = {}
        # ---- TLS / cert handling ----
        self.session.verify = False if insecure else (ca_bundle if ca_bundle else True)
        if client_cert and client_key:
            self.session.cert = (client_cert, client_key)
        elif client_cert:
            self.session.cert = client_cert
        # ------------------------------
        if self.enabled:
            self.session.headers.update({
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "export-jira-contributors/1.0"
            })

    def get_last_active_any_product(self, account_id: str) -> Optional[datetime]:
        if not self.enabled:
            return None
        if account_id in self._cache:
            return self._cache[account_id]

        url = f"https://api.atlassian.com/admin/v1/orgs/{self.org_id}/directory/users/{quote_aaid(account_id)}/last-active-dates"
        if self.verbose:
            print(f"[DEBUG] GET {url} (Admin API)")
        r = self.session.get(url)
        if r.status_code == 429:
            attempt = 1
            while r.status_code == 429 and attempt <= 6:
                if self.verbose:
                    print(f"[DEBUG] Admin API 429, backoff (attempt {attempt})")
                backoff_sleep(r, attempt)
                r = self.session.get(url)
                attempt += 1
        if self.verbose:
            try:
                print(f"[DEBUG] Admin API -> {r.status_code} {r.text[:200]}...")
            except Exception:
                pass
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            print(f"[WARN] Admin API failed for {account_id}: {e}")
            self._cache[account_id] = None
            return None

        data = r.json() or {}
        # Endpoint typically returns an object with product keys -> dates; choose the max
        # Fallback if a list or single value is returned.
        last_dt: Optional[datetime] = None
        try:
            if isinstance(data, dict):
                for _, v in data.items():
                    if isinstance(v, str):
                        dt = parse_jira_time(v)
                        if dt and (last_dt is None or dt > last_dt):
                            last_dt = dt
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                dt = parse_jira_time(item)
                                if dt and (last_dt is None or dt > last_dt):
                                    last_dt = dt
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, str):
                                dt = parse_jira_time(vv)
                                if dt and (last_dt is None or dt > last_dt):
                                    last_dt = dt
            elif isinstance(data, list):
                for v in data:
                    if isinstance(v, str):
                        dt = parse_jira_time(v)
                        if dt and (last_dt is None or dt > last_dt):
                            last_dt = dt
        except Exception:
            pass

        self._cache[account_id] = last_dt
        return last_dt

# -----------------------
# Resume helpers
# -----------------------

def load_processed_keys(out_csv: str) -> Tuple[Set[Tuple[str, str]], bool]:
    """
    Returns (processed_set, has_account_id_column).
    processed_set contains (project_key, account_id) tuples already in the CSV if possible,
    otherwise falls back to (project_key, display_name).
    """
    processed: Set[Tuple[str, str]] = set()
    has_acc = False
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return processed, has_acc

    with open(out_csv, "r", newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf)
        header = next(reader, None) or []
        lower = [h.lower() for h in header]
        try:
            pkey_i = lower.index("project key")
        except ValueError:
            return processed, has_acc

        acc_i = lower.index("account id") if "account id" in lower else -1
        if acc_i >= 0:
            has_acc = True
            for row in reader:
                if len(row) > max(pkey_i, acc_i):
                    processed.add((row[pkey_i], row[acc_i]))
        else:
            # fallback to display name
            try:
                uname_i = lower.index("user name")
                for row in reader:
                    if len(row) > max(pkey_i, uname_i):
                        processed.add((row[pkey_i], row[uname_i]))
            except ValueError:
                pass
    return processed, has_acc

# -----------------------
# Activity
# -----------------------

def get_issue_latest_activity(jc: JiraClient, issue_key: str, verbose: bool = False) -> Optional[datetime]:
    try:
        data = jc._get(f"/rest/api/3/issue/{issue_key}",
                       params={"expand": "renderedFields,changelog"})
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            return None
        raise
    fields = data.get("fields") or {}
    updated = parse_jira_time(fields.get("updated"))
    latest = updated
    ch = data.get("changelog") or {}
    for h in ch.get("histories") or []:
        created_dt = parse_jira_time(h.get("created"))
        if created_dt and (latest is None or created_dt > latest):
            latest = created_dt
    if verbose:
        print(f"[DEBUG] Issue {issue_key} latest activity: {latest}")
    return latest

def get_latest_user_activity_in_project(jc: JiraClient, project_key: str, account_id: str,
                                        max_issue_scan: int = 500, verbose: bool = False) -> Optional[datetime]:
    jql = f"project={project_key} AND assignee={account_id}"
    start_at = 0
    best_dt: Optional[datetime] = None
    scanned = 0
    while scanned < max_issue_scan:
        try:
            resp = jc._get("/rest/api/3/search/jql",
                           params={"jql": jql, "startAt": start_at,
                                   "maxResults": 50, "expand": "changelog"})
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                break
            raise
        issues = resp.get("issues") or []
        if not issues:
            break
        for issue in issues:
            scanned += 1
            latest = get_issue_latest_activity(jc, issue.get("key"), verbose)
            if latest and (best_dt is None or latest > best_dt):
                best_dt = latest
            if scanned >= max_issue_scan:
                break
        if len(issues) < 50:
            break
        start_at += 50
    if verbose:
        print(f"[DEBUG] User {account_id} in {project_key} latest: {best_dt}")
    return best_dt

# -----------------------
# Export
# -----------------------

def export_contributors(base_url: str, email: str, token: str, out_csv: str,
                        include_inactive: bool = False, max_issue_scan: int = 500,
                        show_progress: bool = True, resume: bool = False,
                        verbose: bool = False,
                        org_id: Optional[str] = None, org_api_key: Optional[str] = None,
                        ca_bundle: Optional[str] = None, client_cert: Optional[str] = None,
                        client_key: Optional[str] = None, insecure: bool = False):
    jc = JiraClient(base_url, email, token, verbose,
                    ca_bundle=ca_bundle, client_cert=client_cert,
                    client_key=client_key, insecure=insecure)
    admin = AtlassianAdminClient(org_id, org_api_key, verbose,
                                 ca_bundle=ca_bundle, client_cert=client_cert,
                                 client_key=client_key, insecure=insecure)

    projects = list(jc.iter_projects())
    if verbose:
        print(f"[INFO] Found {len(projects)} projects")

    proj_iter = tqdm(projects, desc="Projects", unit="proj") if show_progress and tqdm else projects

    append_mode = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0
    mode = "a" if append_mode else "w"

    processed_keys, has_acc = (set(), False)
    if resume and append_mode:
        processed_keys, has_acc = load_processed_keys(out_csv)
        if verbose:
            print(f"[INFO] Resume loaded {len(processed_keys)} existing rows (has account id column: {has_acc})")

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not append_mode:
            writer.writerow(["project name", "project key", "user name",
                             "email", "last worked (UTC)", "last active (UTC)",
                             "Roles", "account id"])

        for proj in proj_iter:
            pkey = proj.get("key")
            pname = proj.get("name") or pkey
            if verbose:
                print(f"[INFO] Processing project {pkey} - {pname}")
            try:
                role_members = jc.get_project_role_members(pkey)
            except Exception as e:
                print(f"[WARN] Roles failed for {pkey}: {e}")
                role_members = {}

            users = jc.iter_users_with_browse(pkey, include_inactive=include_inactive)
            for user in users:
                acc_id = user.get("accountId")
                if not acc_id:
                    continue

                # 🚫 Skip plugin/app users
                if user.get("accountType") == "app":
                    if verbose:
                        print(f"[SKIP] Plugin user {user.get('displayName')} ({acc_id})")
                    continue

                display_name = user.get("displayName") or user.get("name") or ""
                email_address = user.get("emailAddress") or ""

                # Resume key: prefer (project_key, account_id)
                resume_key = (pkey, acc_id) if has_acc or not append_mode else (pkey, display_name)
                if resume and resume_key in processed_keys:
                    if verbose:
                        print(f"[SKIP] {pkey}-{display_name} (resume)")
                    continue

                last_worked_dt = get_latest_user_activity_in_project(
                    jc, pkey, acc_id, max_issue_scan=max_issue_scan, verbose=verbose)

                last_active_dt = admin.get_last_active_any_product(acc_id)

                user_roles = sorted([r for r, members in role_members.items() if acc_id in members])
                roles_str = "; ".join(user_roles)

                row = [pname, pkey, display_name, email_address,
                       dt_to_iso(last_worked_dt), dt_to_iso(last_active_dt), roles_str, acc_id]
                writer.writerow(row)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
                if verbose:
                    print(f"[WRITE] {pkey}-{display_name}")

                if resume:
                    processed_keys.add(resume_key)

# -----------------------
# CLI
# -----------------------

def main():
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(description="Export Jira contributors with resume + logging + Admin 'last active' (skips plugin users)")
    parser.add_argument("--base-url", default=os.environ.get("JIRA_BASE_URL"))
    parser.add_argument("--email", default=os.environ.get("JIRA_EMAIL"))
    parser.add_argument("--api-token", default=os.environ.get("JIRA_API_TOKEN"))
    parser.add_argument("-o", "--out", default=os.environ.get("OUT_CONTRIBUTORS", "jira_contributors.csv"))
    parser.add_argument("--include-inactive", action="store_true", default=str2bool(os.environ.get("INCLUDE_INACTIVE", "false")))
    parser.add_argument("--max-issue-scan", type=int, default=int(os.environ.get("MAX_ISSUE_SCAN", "500")))
    parser.add_argument("--no-progress", action="store_true", default=str2bool(os.environ.get("NO_PROGRESS", "false")))
    parser.add_argument("--resume", action="store_true", default=str2bool(os.environ.get("RESUME", "false")))
    parser.add_argument("--verbose", action="store_true", default=str2bool(os.environ.get("VERBOSE", "false")))
    parser.add_argument("--org-id", default=os.environ.get("ATLASSIAN_ORG_ID"))
    parser.add_argument("--org-api-key", default=os.environ.get("ATLASSIAN_API_KEY"))

    # ---- NEW: TLS / certificate options (env-driven, with CLI override) ----
    parser.add_argument("--ca-bundle", default=os.environ.get("CA_BUNDLE"),
                        help="Path to custom CA bundle PEM used to verify server certificates")
    parser.add_argument("--client-cert", default=os.environ.get("CLIENT_CERT"),
                        help="Path to client certificate PEM (or combined cert+key PEM)")
    parser.add_argument("--client-key", default=os.environ.get("CLIENT_KEY"),
                        help="Path to client private key PEM (if separate)")
    parser.add_argument("--insecure", action="store_true",
                        default=str2bool(os.environ.get("INSECURE", "false")),
                        help="Disable TLS verification (NOT recommended)")
    # -----------------------------------------------------------------------

    args = parser.parse_args()

    missing = [k for k, v in {
        "JIRA_BASE_URL": args.base_url,
        "JIRA_EMAIL": args.email,
        "JIRA_API_TOKEN": args.api_token,
    }.items() if not v]
    if missing:
        print("Missing required settings: " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    show_progress = not args.no_progress

    export_contributors(
        args.base_url, args.email, args.api_token, args.out,
        include_inactive=args.include_inactive, max_issue_scan=args.max_issue_scan,
        show_progress=show_progress, resume=args.resume, verbose=args.verbose,
        org_id=args.org_id, org_api_key=args.org_api_key,
        ca_bundle=args.ca_bundle, client_cert=args.client_cert,
        client_key=args.client_key, insecure=args.insecure
    )

if __name__ == "__main__":
    main()
