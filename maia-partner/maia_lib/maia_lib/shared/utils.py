import datetime
import os.path

import pygit2


def get_commit_info(repo_path=os.path.dirname(__file__)):
    repository_git_path = pygit2.discover_repository(repo_path)
    repo = pygit2.Repository(repository_git_path)
    last = repo[repo.head.target]
    return {
        "date": datetime.datetime.utcfromtimestamp(last.commit_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "hex": last.hex,
        "message": last.message.strip(),
        "author": last.author.name,
        "author_email": last.author.email,
        "link": f"x/{last.hex}",
    }
