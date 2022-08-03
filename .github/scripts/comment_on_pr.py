from expecttest import Any
from trymerge import gh_post_pr_comment
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
import os


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Comment on a PR")
    parser.add_argument("pr_num", type=int)
    parser.add_argument("action", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name(), debug=True)
    org, project = repo.gh_owner_and_name()
    sender = os.environ.get("GH_SENDER")
    run_url = os.environ.get("GH_RUN_URL")

    author_mention_prefix = f"@{sender}" if sender is not None else "The"
    job_link = f"[job]({run_url})" if run_url is not None else "job"
    msg = f"{author_mention_prefix} {args.action} {job_link} was canceled."

    gh_post_pr_comment(org, project, args.pr_num, msg)
    print(org, project, args.pr_num, msg)


if __name__ == "__main__":
    main()