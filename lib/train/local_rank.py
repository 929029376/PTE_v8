import os


def resolve_local_rank(cli_local_rank, environ=None):
    if cli_local_rank != -1:
        return cli_local_rank
    environ = os.environ if environ is None else environ
    env_rank = environ.get("LOCAL_RANK")
    return int(env_rank) if env_rank is not None else -1