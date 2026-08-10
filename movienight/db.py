"""Lakebase connection management.

Lakebase authenticates with an OAuth token used as the Postgres password, and
that token expires after about an hour. A long-lived pool holding one password
therefore dies silently mid-session. The fix is a connection class that mints a
fresh token every time the pool opens a connection, plus a max_lifetime well
under the expiry so connections are recycled before their credential dies.
"""

import os
from contextlib import contextmanager

from databricks.sdk import WorkspaceClient
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SCHEMA = "movie_night"
INSTANCE = os.environ.get("LAKEBASE_INSTANCE", "bootcamp-support-db")
DATABASE = os.environ.get("LAKEBASE_DATABASE", "databricks_postgres")

_workspace = None
_pool = None


def _ws():
    global _workspace
    if _workspace is None:
        _workspace = WorkspaceClient()
    return _workspace


class _TokenConnection(Connection):
    """Mints a fresh OAuth token for each new physical connection."""

    @classmethod
    def connect(cls, conninfo="", **kwargs):
        w = _ws()
        instance = w.database.get_database_instance(name=INSTANCE)
        cred = w.database.generate_database_credential(
            request_id=os.urandom(8).hex(), instance_names=[INSTANCE]
        )
        kwargs.update(
            host=instance.read_write_dns,
            port=5432,
            dbname=DATABASE,
            user=w.current_user.me().user_name,
            password=cred.token,
            sslmode="require",
            options=f"-c search_path={SCHEMA},public",
        )
        return super().connect("", **kwargs)


def get_pool():
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo="",
            connection_class=_TokenConnection,
            kwargs={"row_factory": dict_row},
            min_size=1,
            max_size=4,
            max_lifetime=45 * 60,      # under the ~60 min token expiry
            open=True,
        )
    return _pool


@contextmanager
def connection():
    with get_pool().connection() as conn:
        yield conn
