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
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm  # progress bars
except Exception:
    tqdm = None

# -----------------------
# Helpers
# -----------------------

def str2bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def backoff_sleep(resp, attempt: int):
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


def is_probably_html(text: str) -> bool:
    t = (text or "").strip().lower()
    # quick-and-dirty detection for login/proxy pages
    return ("<html" in t) or ("<!doctype html" in t) or ("<title" in t)


def raise_non_json_200(url: str, resp: requests.Response, verbose: bool):
    ctype = resp.headers.get("Content-Type", "")
    snippet = ""
    try:
        snippet = resp.text[:800]
    except Exception:
        snippet = "<non-text body>"
    msg = [
        f"Expected JSON but got non-JSON response (HTTP 200) from: {url}",
        f"Content-Type: {ctype or '<none>'}",
    ]
    if is_probably_html(snippet):
        msg.append("Body looks like HTML (often an SSO/proxy login page or block page).")
    if verbose:
        msg.append("Body snippet:\n" + snippet)
    msg.append(
        "Hints: ensure you are calling the correct Jira Cloud URL, your API token is valid and belongs to the same account, "
        "and any corporate proxy/SSO is not intercepting this API path. If a proxy rewrites TLS, set CA_BUNDLE to your corp root (PEM)."
    )
    raise RuntimeError("\n".join(msg))


# -----------------------
# Jira Client
# -----------------------

PROJECT_PAGE_SIZE = 50


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
        self.session.verify = False if insecure else (ca_bundle if ca_bundle else True)
        if client_cert and client_key:
            self.session.cert = (client_cert, client_key)
        elif client_cert:
            self.session.cert = client_cert
        # ------------------------------
        self.verbose = verbose
        # cache for user objects
        self._user_cache: Dict[str, Optional[dict]] = {}

    def _get(self, path: str, params: Optional[dict] = None):
        url = urljoin(self.base_url, path.strip("/"))
        if self.verbose:
            print(f"[DEBUG] GET {url} params={params}")
            if self.session.verify is True:
                print("[DEBUG] TLS verify: system CAs")
            elif self.session.verify is False:
                print("[DEBUG] TLS verify: DISABLED (insecure)")
            else:
                print(f"[DEBUG] TLS verify: CA bundle -> {self.session.verify}")
            if getattr(self.session, "cert", None):
                print(f"[DEBUG] mTLS client cert configured: {self.session.cert}")

        resp = self.session.get(url, params=params, allow_redirects=True)
        if resp.status_code == 429:
            attempt = 1
            while resp.status_code == 429 and attempt <= 6:
                if self.verbose:
                    print(f"[DEBUG] 429 from {url}, backing off (attempt {attempt})")
                backoff_sleep(resp, attempt)
                resp = self.session.get(url, params=params, allow_redirects=True)
                attempt += 1

        if self.verbose:
            try:
                print(f"[DEBUG] {url} -> {resp.status_code} CT={resp.headers.get('Content-Type','')}")
            except Exception:
                pass

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            if self.verbose:
                try:
                    print(f"[DEBUG] Body: {resp.text[:800]}")
                except Exception:
                    pass
            raise

        try:
            return resp.json()
        except ValueError:
            raise_non_json_200(url, resp, self.verbose)

    def iter_projects(self) -> Iterable[dict]:
        """
        Stream projects page-by-page so we can start processing immediately.
        """
        start_at = 0
        while True:
            data = self._get(
                "/rest/api/3/project/search",
                params={"startAt": start_at, "maxResults": PROJECT_PAGE_SIZE},
            )
            values = data.get("values") or []
            if not values:
                break
            for p in values:
                yield p
            if len(values) < PROJECT_PAGE_SIZE:
                break
            start_at += PROJECT_PAGE_SIZE

    def get_project_role_members(self, project_key: str) -> Dict[str, Set[str]]:
        """
        Return a dict: role_name -> set(accountId) for users in that project's roles.
        """
        roles = self._get(f"/rest/api/3/project/{project_key}/role")
        out: Dict[str, Set[str]] = {}
        for role_name, role_url in roles.items():
            try:
                r = self.session.get(role_url, allow_redirects=True)
                if r.status_code == 429:
                    attempt = 1
                    while r.status_code == 429 and attempt <= 6:
                        backoff_sleep(r, attempt)
                        r = self.session.get(role_url, allow_redirects=True)
                        attempt += 1
                r.raise_for_status()
                try:
                    payload = r.json()
                except ValueError:
                    raise_non_json_200(role_url, r, self.verbose)
                actors = payload.get("actors") or []
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

    def get_user(self, account_id: str) -> Optional[dict]:
        """
        Fetch and cache a Jira user by accountId.
        Returns None if user cannot be loaded.
        """
        if account_id in self._user_cache:
            return self._user_cache[account_id]

        try:
            user = self._get("/rest/api/3/user", params={"accountId": account_id})
        except requests.HTTPError as e:
            if self.verbose:
                print(f"[WARN] Failed to fetch user {account_id}: {e}")
            self._user_cache[account_id] = None
            return None

        self._user_cache[account_id] = user
        return user


# -----------------------
# Atlassian Admin API client (kept for future use)
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
        self.session.verify = False if insecure else (ca_bundle if ca_bundle else True)
        if client_cert and client_key:
            self.session.cert = (client_cert, client_key)
        elif client_cert:
            self.session.cert = client_cert
        if self.enabled:
            self.session.headers.update({
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "export-jira-contributors/1.0"
            })

    def get_last_active_any_product(self, account_id: str) -> Optional[datetime]:
        # Not used in current CSVs but left here if you want to add a column later.
        if not self.enabled:
            return None
        if account_id in self._cache:
            return self._cache[account_id]

        url = f"https://api.atlassian.com/admin/v1/orgs/{self.org_id}/directory/users/{quote_aaid(account_id)}/last-active-dates"
        if self.verbose:
            print(f"[DEBUG] GET {url} (Admin API)")
        r = self.session.get(url, allow_redirects=True)
        if r.status_code == 429:
            attempt = 1
            while r.status_code == 429 and attempt <= 6:
                if self.verbose:
                    print(f"[DEBUG] Admin API 429, backoff (attempt {attempt})")
                backoff_sleep(r, attempt)
                r = self.session.get(url, allow_redirects=True)
                attempt += 1
        if self.verbose:
            try:
                print(f"[DEBUG] Admin API -> {r.status_code} CT={r.headers.get('Content-Type','')}")
            except Exception:
                pass
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            print(f"[WARN] Admin API failed for {account_id}: {e}")
            self._cache[account_id] = None
            return None

        try:
            data = r.json() or {}
        except ValueError:
            try:
                raise_non_json_200(url, r, self.verbose)
            except RuntimeError as ex:
                print(f"[WARN] {ex}")
                self._cache[account_id] = None
                return None

        last_dt: Optional[datetime] = None
        try:
            if isinstance(data, dict):
                for _, v in data.items():
                    if isinstance(v, str):
                        dt = parse_jira_time(v)
                        last_dt = max(last_dt, dt) if (last_dt and dt) else (dt or last_dt)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                dt = parse_jira_time(item)
                                last_dt = max(last_dt, dt) if (last_dt and dt) else (dt or last_dt)
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, str):
                                dt = parse_jira_time(vv)
                                last_dt = max(last_dt, dt) if (last_dt and dt) else (dt or last_dt)
            elif isinstance(data, list):
                for v in data:
                    if isinstance(v, str):
                        dt = parse_jira_time(v)
                        last_dt = max(last_dt, dt) if (last_dt and dt) else (dt or last_dt)
        except Exception:
            pass

        self._cache[account_id] = last_dt
        return last_dt


# -----------------------
# Resume helpers
# -----------------------

def load_processed_keys(out_csv: str) -> Set[Tuple[str, str]]:
    """
    Returns processed_set of (project_key, account_id) tuples already in the projects CSV.
    """
    processed: Set[Tuple[str, str]] = set()
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return processed

    with open(out_csv, "r", newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf)
        header = next(reader, None) or []
        lower = [h.lower() for h in header]
        try:
            pkey_i = lower.index("project key")
            acc_i = lower.index("account id")
        except ValueError:
            return processed

        for row in reader:
            if len(row) > max(pkey_i, acc_i):
                processed.add((row[pkey_i], row[acc_i]))
    return processed


def load_user_last_work(users_csv: str) -> Dict[str, Tuple[str, Optional[datetime]]]:
    """
    Load existing per-user 'last worked' data from the users CSV (if present),
    so we can merge with new data when using --resume.
    Returns: { account_id: (user_name, last_worked_dt) }
    """
    data: Dict[str, Tuple[str, Optional[datetime]]] = {}
    if not os.path.exists(users_csv) or os.path.getsize(users_csv) == 0:
        return data

    with open(users_csv, "r", newline="", encoding="utf-8") as rf:
        reader = csv.reader(rf)
        header = next(reader, None) or []
        lower = [h.lower() for h in header]

        try:
            acc_i = lower.index("account id")
            uname_i = lower.index("user name")
        except ValueError:
            return data

        last_idx = None
        for i, h in enumerate(lower):
            if "last worked" in h:
                last_idx = i
                break

        for row in reader:
            if len(row) <= max(acc_i, uname_i):
                continue
            acc_id = row[acc_i]
            uname = row[uname_i]
            last_dt: Optional[datetime] = None
            if last_idx is not None and len(row) > last_idx:
                last_dt = parse_jira_time(row[last_idx])
            if acc_id in data:
                existing_name, existing_dt = data[acc_id]
                if last_dt and (existing_dt is None or last_dt > existing_dt):
                    data[acc_id] = (uname or existing_name, last_dt)
            else:
                data[acc_id] = (uname, last_dt)
    return data


# -----------------------
# Activity
# -----------------------

def get_latest_user_activity_in_project(
    jc: JiraClient,
    project_key: str,
    account_id: str,
    max_issue_scan: int = 500,
    verbose: bool = False,
) -> Optional[datetime]:
    """
    Get the latest activity for a user in a project by scanning issues via /search
    with expand=changelog – avoids separate per-issue calls.
    """
    jql = f"project={project_key} AND assignee={account_id}"
    start_at = 0
    best_dt: Optional[datetime] = None
    scanned = 0

    while scanned < max_issue_scan:
        try:
            resp = jc._get(
                "/rest/api/3/search/jql",
                params={
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": 50,
                    "expand": "changelog",
                },
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                break
            raise

        issues = resp.get("issues") or []
        if not issues:
            break

        for issue in issues:
            scanned += 1
            fields = issue.get("fields") or {}
            updated = parse_jira_time(fields.get("updated"))
            latest = updated

            ch = issue.get("changelog") or {}
            for h in ch.get("histories") or []:
                created_dt = parse_jira_time(h.get("created"))
                if created_dt and (latest is None or created_dt > latest):
                    latest = created_dt

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
# Worker task (runs in threads)
# -----------------------

def worker_task(
    config: dict,
    project_key: str,
    project_name: str,
    acc_id: str,
    roles_str: str,
) -> Optional[Tuple[str, str, str, str, str, str, Optional[datetime]]]:
    """
    Worker: create its own JiraClient, fetch user, compute last_worked_dt.
    Returns: (pname, pkey, acc_id, user_name, email, roles_str, last_worked_dt) or None (skip).
    """
    jc = JiraClient(
        config["base_url"],
        config["email"],
        config["token"],
        verbose=config["verbose"],
        ca_bundle=config["ca_bundle"],
        client_cert=config["client_cert"],
        client_key=config["client_key"],
        insecure=config["insecure"],
    )

    user = jc.get_user(acc_id)
    if not user:
        return None

    if not config["include_inactive"] and not user.get("active", True):
        return None

    if user.get("accountType") == "app":
        return None

    display_name = user.get("displayName") or user.get("name") or ""
    email_address = user.get("emailAddress") or ""

    last_worked_dt = get_latest_user_activity_in_project(
        jc,
        project_key,
        acc_id,
        max_issue_scan=config["max_issue_scan"],
        verbose=config["verbose"],
    )

    return (
        project_name,
        project_key,
        acc_id,
        display_name,
        email_address,
        roles_str,
        last_worked_dt,
    )


# -----------------------
# Export
# -----------------------

def export_contributors(
    base_url: str,
    email: str,
    token: str,
    out_base: str,
    include_inactive: bool = False,
    max_issue_scan: int = 500,
    show_progress: bool = True,
    resume: bool = False,
    verbose: bool = False,
    org_id: Optional[str] = None,
    org_api_key: Optional[str] = None,
    ca_bundle: Optional[str] = None,
    client_cert: Optional[str] = None,
    client_key: Optional[str] = None,
    insecure: bool = False,
    workers: int = 4,
):
    """
    Writes two CSV files:

    1) <base>_projects.csv  (per-project membership)
       Columns: project name, project key, account id, user name, email, roles

    2) <base>_users.csv     (per-user last worked)
       Columns: account id, user name, last worked (UTC)
    """
    # Derive two output paths from the base
    root, ext = os.path.splitext(out_base)
    if not ext:
        ext = ".csv"
    projects_csv = f"{root}_projects{ext}"
    users_csv = f"{root}_users{ext}"

    jc_main = JiraClient(
        base_url,
        email,
        token,
        verbose,
        ca_bundle=ca_bundle,
        client_cert=client_cert,
        client_key=client_key,
        insecure=insecure,
    )
    _admin = AtlassianAdminClient(
        org_id,
        org_api_key,
        verbose,
        ca_bundle=ca_bundle,
        client_cert=client_cert,
        client_key=client_key,
        insecure=insecure,
    )

    projects_iter = jc_main.iter_projects()
    if show_progress and tqdm:
        proj_iter = tqdm(projects_iter, desc="Projects", unit="proj")
    else:
        proj_iter = projects_iter

    append_mode = os.path.exists(projects_csv) and os.path.getsize(projects_csv) > 0
    mode = "a" if append_mode else "w"

    processed_keys: Set[Tuple[str, str]] = set()
    if resume and append_mode:
        processed_keys = load_processed_keys(projects_csv)
        if verbose:
            print(f"[INFO] Resume loaded {len(processed_keys)} existing rows")

    user_last_work: Dict[str, Tuple[str, Optional[datetime]]] = {}
    if resume:
        user_last_work = load_user_last_work(users_csv)
        if verbose:
            print(f"[INFO] Loaded {len(user_last_work)} existing user last-worked records")

    worker_config = {
        "base_url": base_url,
        "email": email,
        "token": token,
        "include_inactive": include_inactive,
        "max_issue_scan": max_issue_scan,
        "verbose": verbose,
        "ca_bundle": ca_bundle,
        "client_cert": client_cert,
        "client_key": client_key,
        "insecure": insecure,
    }

    with open(projects_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not append_mode:
            writer.writerow([
                "project name",
                "project key",
                "account id",
                "user name",
                "email",
                "Roles",
            ])

        # If workers <= 1, just run sequentially (no thread pool)
        if workers <= 1:
            for proj in proj_iter:
                pkey = proj.get("key")
                pname = proj.get("name") or pkey
                if verbose:
                    print(f"[INFO] Processing project {pkey} - {pname}")

                try:
                    role_members = jc_main.get_project_role_members(pkey)
                except Exception as e:
                    print(f"[WARN] Roles failed for {pkey}: {e}")
                    role_members = {}

                project_user_ids: Set[str] = set()
                for _, members in role_members.items():
                    project_user_ids.update(members)

                if verbose:
                    print(f"[INFO] {pkey} has {len(project_user_ids)} users from roles")

                if not project_user_ids:
                    continue

                for acc_id in sorted(project_user_ids):

                    resume_key = (pkey, acc_id)
                    if resume and resume_key in processed_keys:
                        if verbose:
                            print(f"[SKIP] {pkey}-{acc_id} (resume)")
                        continue

                    result = worker_task(
                        worker_config,
                        project_key=pkey,
                        project_name=pname,
                        acc_id=acc_id,
                        roles_str="; ".join(
                            r for r, members in role_members.items() if acc_id in members
                        ),
                    )
                    if result is None:
                        if resume:
                            processed_keys.add(resume_key)
                        continue

                    pname2, pkey2, acc_id2, display_name, email_address, roles_str, last_worked_dt = result

                    # update per-user aggregated last-worked
                    if acc_id2 in user_last_work:
                        existing_name, existing_dt = user_last_work[acc_id2]
                        best_name = display_name or existing_name
                        if last_worked_dt and (existing_dt is None or last_worked_dt > existing_dt):
                            user_last_work[acc_id2] = (best_name, last_worked_dt)
                        else:
                            user_last_work[acc_id2] = (best_name, existing_dt)
                    else:
                        user_last_work[acc_id2] = (display_name, last_worked_dt)

                    row = [
                        pname2,
                        pkey2,
                        acc_id2,
                        display_name,
                        email_address,
                        roles_str,
                    ]
                    writer.writerow(row)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass

                    if verbose:
                        print(f"[WRITE] {pkey2}-{display_name} ({acc_id2})")

                    if resume:
                        processed_keys.add(resume_key)

        else:
            # Parallel mode with ThreadPoolExecutor
            from concurrent.futures import Future

            pending: Set[Future] = set()
            future_to_resume: Dict[Future, Tuple[str, str]] = {}

            def drain_completed(all_done: bool = False):
                """Write completed futures' results to CSV and update aggregates."""
                nonlocal pending
                done_list = list(pending) if all_done else [ft for ft in list(pending) if ft.done()]
                for ft in done_list:
                    pending.remove(ft)
                    resume_key = future_to_resume.pop(ft, None)
                    result = ft.result()
                    if result is None:
                        if resume and resume_key:
                            processed_keys.add(resume_key)
                        continue

                    pname2, pkey2, acc_id2, display_name, email_address, roles_str, last_worked_dt = result

                    # update per-user aggregated last-worked
                    if acc_id2 in user_last_work:
                        existing_name, existing_dt = user_last_work[acc_id2]
                        best_name = display_name or existing_name
                        if last_worked_dt and (existing_dt is None or last_worked_dt > existing_dt):
                            user_last_work[acc_id2] = (best_name, last_worked_dt)
                        else:
                            user_last_work[acc_id2] = (best_name, existing_dt)
                    else:
                        user_last_work[acc_id2] = (display_name, last_worked_dt)

                    row = [
                        pname2,
                        pkey2,
                        acc_id2,
                        display_name,
                        email_address,
                        roles_str,
                    ]
                    writer.writerow(row)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass

                    if verbose:
                        print(f"[WRITE] {pkey2}-{display_name} ({acc_id2})")

                    if resume and resume_key:
                        processed_keys.add(resume_key)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for proj in proj_iter:
                    pkey = proj.get("key")
                    pname = proj.get("name") or pkey
                    if verbose:
                        print(f"[INFO] Processing project {pkey} - {pname}")

                    try:
                        role_members = jc_main.get_project_role_members(pkey)
                    except Exception as e:
                        print(f"[WARN] Roles failed for {pkey}: {e}")
                        role_members = {}

                    project_user_ids: Set[str] = set()
                    for _, members in role_members.items():
                        project_user_ids.update(members)

                    if verbose:
                        print(f"[INFO] {pkey} has {len(project_user_ids)} users from roles")

                    if not project_user_ids:
                        continue

                    for acc_id in sorted(project_user_ids):
                        resume_key = (pkey, acc_id)
                        if resume and resume_key in processed_keys:
                            if verbose:
                                print(f"[SKIP] {pkey}-{acc_id} (resume)")
                            continue

                        roles_str = "; ".join(
                            r for r, members in role_members.items() if acc_id in members
                        )

                        ft = executor.submit(
                            worker_task,
                            worker_config,
                            pkey,
                            pname,
                            acc_id,
                            roles_str,
                        )
                        pending.add(ft)
                        future_to_resume[ft] = resume_key

                        # keep queue bounded so we start writing as tasks finish
                        if len(pending) >= workers * 10:
                            drain_completed(all_done=False)

                # after all tasks submitted, drain the rest
                drain_completed(all_done=True)

    # After processing all projects, write / rewrite the per-user sheet
    with open(users_csv, "w", newline="", encoding="utf-8") as uf:
        uwriter = csv.writer(uf)
        uwriter.writerow(["account id", "user name", "last worked (UTC)"])
        for acc_id in sorted(user_last_work.keys()):
            uname, last_dt = user_last_work[acc_id]
            uwriter.writerow([acc_id, uname, dt_to_iso(last_dt)])

    if verbose:
        print(f"[INFO] Wrote per-project data to {projects_csv}")
        print(f"[INFO] Wrote per-user last-worked data to {users_csv}")


# -----------------------
# CLI
# -----------------------

def main():
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(
        description=(
            "Export Jira contributors (by project roles) into two CSVs: "
            "1) per-project membership, 2) per-user last worked. "
            "Supports resume, TLS options, and parallel workers."
        )
    )
    parser.add_argument("--base-url", default=os.environ.get("JIRA_BASE_URL"))
    parser.add_argument("--email", default=os.environ.get("JIRA_EMAIL"))
    parser.add_argument("--api-token", default=os.environ.get("JIRA_API_TOKEN"))
    parser.add_argument(
        "-o",
        "--out",
        default=os.environ.get("OUT_CONTRIBUTORS", "jira_contributors.csv"),
        help="Base output path; will create <base>_projects.csv and <base>_users.csv",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        default=str2bool(os.environ.get("INCLUDE_INACTIVE", "false")),
    )
    parser.add_argument(
        "--max-issue-scan",
        type=int,
        default=int(os.environ.get("MAX_ISSUE_SCAN", "500")),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=str2bool(os.environ.get("NO_PROGRESS", "false")),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=str2bool(os.environ.get("RESUME", "false")),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=str2bool(os.environ.get("VERBOSE", "false")),
    )
    parser.add_argument("--org-id", default=os.environ.get("ATLASSIAN_ORG_ID"))
    parser.add_argument("--org-api-key", default=os.environ.get("ATLASSIAN_API_KEY"))

    # TLS / certificate options (env-driven, with CLI override)
    parser.add_argument(
        "--ca-bundle",
        default=os.environ.get("CA_BUNDLE"),
        help="Path to custom CA bundle .crt/.pem (PEM) used to verify server certificates",
    )
    parser.add_argument(
        "--client-cert",
        default=os.environ.get("CLIENT_CERT"),
        help="Path to client certificate PEM (or combined cert+key PEM) if proxy requires mTLS",
    )
    parser.add_argument(
        "--client-key",
        default=os.environ.get("CLIENT_KEY"),
        help="Path to client private key PEM (if separate)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=str2bool(os.environ.get("INSECURE", "false")),
        help="Disable TLS verification (NOT recommended)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "4")),
        help="Number of parallel workers for fetching user data (>=1, default 4).",
    )

    args = parser.parse_args()

    missing = [
        k
        for k, v in {
            "JIRA_BASE_URL": args.base_url,
            "JIRA_EMAIL": args.email,
            "JIRA_API_TOKEN": args.api_token,
        }.items()
        if not v
    ]
    if missing:
        print("Missing required settings: " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)

    show_progress = not args.no_progress

    export_contributors(
        args.base_url,
        args.email,
        args.api_token,
        args.out,
        include_inactive=args.include_inactive,
        max_issue_scan=args.max_issue_scan,
        show_progress=show_progress,
        resume=args.resume,
        verbose=args.verbose,
        org_id=args.org_id,
        org_api_key=args.org_api_key,
        ca_bundle=args.ca_bundle,
        client_cert=args.client_cert,
        client_key=args.client_key,
        insecure=args.insecure,
        workers=max(1, args.workers),
    )


if __name__ == "__main__":
    main()
