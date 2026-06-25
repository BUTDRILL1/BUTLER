import base64
import subprocess
from typing import Any
from pathlib import Path

import requests
from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext, ToolError

BASE_URL = "https://api.github.com"


def _get_headers(ctx: ToolContext) -> dict[str, str]:
    if not ctx.config.github_token:
        raise ToolError(
            "BUTLER_GITHUB_TOKEN is not set in the configuration. "
            "Please add it to your .env file."
        )
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {ctx.config.github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _make_request(ctx: ToolContext, method: str, endpoint: str, **kwargs: Any) -> Any:
    headers = _get_headers(ctx)
    url = f"{BASE_URL}{endpoint}"
    
    try:
        response = requests.request(method, url, headers=headers, timeout=15, **kwargs)
        response.raise_for_status()
        if response.status_code == 204:
            return None
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_msg = f"GitHub API error: {e}"
        try:
            error_details = e.response.json()
            error_msg += f"\nDetails: {error_details}"
        except Exception:
            pass
        raise ToolError(error_msg)
    except requests.exceptions.RequestException as e:
        raise ToolError(f"GitHub API request failed: {e}")


# --- search_repos ---
class SearchReposArgs(BaseModel):
    query: str = Field(description="Search query (e.g., 'language:python machine learning').")
    sort: str | None = Field(default=None, description="Sort field (stars, forks, help-wanted-issues, updated).")


def _search_repos(ctx: ToolContext, args: SearchReposArgs) -> dict[str, Any]:
    params = {"q": args.query, "per_page": 10}
    if args.sort:
        params["sort"] = args.sort
    result = _make_request(ctx, "GET", "/search/repositories", params=params)
    items = result.get("items", [])
    
    formatted = []
    for item in items:
        formatted.append({
            "full_name": item["full_name"],
            "description": item.get("description", ""),
            "html_url": item["html_url"],
            "stargazers_count": item["stargazers_count"],
            "language": item.get("language", ""),
        })
    return {"total_count": result.get("total_count", 0), "repositories": formatted}


# --- get_file ---
class GetFileArgs(BaseModel):
    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    path: str = Field(description="Path to the file in the repository.")
    ref: str | None = Field(default=None, description="Branch or commit sha (optional).")


def _get_file(ctx: ToolContext, args: GetFileArgs) -> dict[str, Any]:
    params = {}
    if args.ref:
        params["ref"] = args.ref
        
    result = _make_request(ctx, "GET", f"/repos/{args.owner}/{args.repo}/contents/{args.path}", params=params)
    
    if isinstance(result, list):
        raise ToolError(f"Path '{args.path}' is a directory, not a file.")
        
    if result.get("encoding") == "base64":
        content = base64.b64decode(result["content"]).decode("utf-8", errors="replace")
        return {
            "name": result["name"],
            "path": result["path"],
            "html_url": result["html_url"],
            "content": content
        }
    raise ToolError("Unsupported file encoding from GitHub.")


# --- list_issues ---
class ListIssuesArgs(BaseModel):
    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    state: str = Field(default="open", description="State of issues (open, closed, all).")
    per_page: int = Field(default=30, description="Number of issues to return.")


def _list_issues(ctx: ToolContext, args: ListIssuesArgs) -> dict[str, Any]:
    params = {"state": args.state, "per_page": args.per_page}
    result = _make_request(ctx, "GET", f"/repos/{args.owner}/{args.repo}/issues", params=params)
    
    formatted = []
    for item in result:
        formatted.append({
            "number": item["number"],
            "title": item["title"],
            "state": item["state"],
            "is_pull_request": "pull_request" in item,
            "user": item["user"]["login"],
            "html_url": item["html_url"],
        })
    return {"issues": formatted}


# --- create_issue ---
class CreateIssueArgs(BaseModel):
    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    title: str = Field(description="Title of the issue.")
    body: str | None = Field(default=None, description="Body of the issue.")
    assignees: list[str] | None = Field(default=None, description="List of usernames to assign.")


def _create_issue(ctx: ToolContext, args: CreateIssueArgs) -> dict[str, Any]:
    payload: dict[str, Any] = {"title": args.title}
    if args.body:
        payload["body"] = args.body
    if args.assignees:
        payload["assignees"] = args.assignees
        
    result = _make_request(ctx, "POST", f"/repos/{args.owner}/{args.repo}/issues", json=payload)
    return {
        "number": result["number"],
        "title": result["title"],
        "html_url": result["html_url"],
        "state": result["state"]
    }


# --- list_prs ---
class ListPrsArgs(BaseModel):
    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    state: str = Field(default="open", description="State of PRs (open, closed, all).")


def _list_prs(ctx: ToolContext, args: ListPrsArgs) -> dict[str, Any]:
    params = {"state": args.state, "per_page": 30}
    result = _make_request(ctx, "GET", f"/repos/{args.owner}/{args.repo}/pulls", params=params)
    
    formatted = []
    for item in result:
        formatted.append({
            "number": item["number"],
            "title": item["title"],
            "state": item["state"],
            "user": item["user"]["login"],
            "html_url": item["html_url"],
            "draft": item.get("draft", False),
        })
    return {"pull_requests": formatted}


# --- get_commits ---
class GetCommitsArgs(BaseModel):
    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    per_page: int = Field(default=10, description="Number of commits to return.")


def _get_commits(ctx: ToolContext, args: GetCommitsArgs) -> dict[str, Any]:
    params = {"per_page": args.per_page}
    result = _make_request(ctx, "GET", f"/repos/{args.owner}/{args.repo}/commits", params=params)
    
    formatted = []
    for item in result:
        commit_data = item["commit"]
        formatted.append({
            "sha": item["sha"],
            "message": commit_data["message"],
            "author": commit_data["author"]["name"],
            "date": commit_data["author"]["date"],
            "html_url": item["html_url"]
        })
    return {"commits": formatted}


# --- clone_repo ---
class CloneRepoArgs(BaseModel):
    owner: str = Field(description="Repository owner.")
    repo: str = Field(description="Repository name.")
    destination_path: str = Field(description="Local path to clone the repository into. Must be an absolute path.")


def _clone_repo(ctx: ToolContext, args: CloneRepoArgs) -> dict[str, Any]:
    if not ctx.config.github_token:
        raise ToolError("BUTLER_GITHUB_TOKEN is not set. Cannot clone authenticated repositories.")
        
    dest_path = Path(args.destination_path).resolve()
    
    # We construct the git URL with the oauth2 token for authentication
    token = ctx.config.github_token
    git_url = f"https://oauth2:{token}@github.com/{args.owner}/{args.repo}.git"
    
    try:
        # Run git clone. We capture output so the token isn't easily leaked to console logs if it fails,
        # but we also need to be careful with subprocess errors.
        process = subprocess.run(
            ["git", "clone", git_url, str(dest_path)],
            capture_output=True,
            text=True,
            check=False
        )
        
        if process.returncode != 0:
            error_out = process.stderr.replace(token, "***TOKEN***")
            raise ToolError(f"Git clone failed: {error_out}")
            
        return {
            "success": True,
            "message": f"Successfully cloned {args.owner}/{args.repo} to {dest_path}"
        }
    except FileNotFoundError:
        raise ToolError("Git is not installed or not available in PATH.")


# --- Tool Definitions ---

github_search_repos_tool = Tool(
    name="github.search_repos",
    description="Search GitHub repositories by query.",
    input_model=SearchReposArgs,
    handler=_search_repos,
)

github_get_file_tool = Tool(
    name="github.get_file",
    description="Get the contents of a specific file from a GitHub repository.",
    input_model=GetFileArgs,
    handler=_get_file,
)

github_list_issues_tool = Tool(
    name="github.list_issues",
    description="List issues (and PRs) for a GitHub repository.",
    input_model=ListIssuesArgs,
    handler=_list_issues,
)

github_create_issue_tool = Tool(
    name="github.create_issue",
    description="Create a new issue on a GitHub repository.",
    input_model=CreateIssueArgs,
    handler=_create_issue,
    side_effect=True,
)

github_list_prs_tool = Tool(
    name="github.list_prs",
    description="List pull requests for a GitHub repository.",
    input_model=ListPrsArgs,
    handler=_list_prs,
)

github_get_commits_tool = Tool(
    name="github.get_commits",
    description="Get the latest commits for a GitHub repository.",
    input_model=GetCommitsArgs,
    handler=_get_commits,
)

github_clone_repo_tool = Tool(
    name="github.clone_repo",
    description="Clone a GitHub repository to the local machine.",
    input_model=CloneRepoArgs,
    handler=_clone_repo,
    side_effect=True,
)

TOOLS = [
    github_search_repos_tool,
    github_get_file_tool,
    github_list_issues_tool,
    github_create_issue_tool,
    github_list_prs_tool,
    github_get_commits_tool,
    github_clone_repo_tool,
]
