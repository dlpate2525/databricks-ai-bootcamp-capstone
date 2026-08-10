"""Grant the deployed Databricks App's service principal access to the schema.

A Databricks App runs as its own service principal, not as you. Deploying the app
is therefore not enough: that principal needs a Postgres role on the Lakebase
instance, and then privileges on the `movie_night` schema. Without this the app
starts successfully and then fails on its first query.

Two steps, in order.

STEP 1 - create the Postgres role for the app's service principal (Databricks CLI,
not SQL; the role must exist before any GRANT can reference it):

    databricks apps get movie-night -p <profile> -o json     # read service_principal_client_id

    databricks database create-database-instance-role \
        bootcamp-support-db <service_principal_client_id> \
        --identity-type SERVICE_PRINCIPAL -p <profile>

STEP 2 - run this script to grant schema privileges:

    python scripts/grant_app_access.py <service_principal_client_id>

Re-running is safe; GRANT is idempotent.
"""

import sys

from movienight.db import connection


def main(service_principal_id):
    # The role name is the service principal's client id, quoted because it
    # contains hyphens. It is an identifier, not a value, so it cannot be a
    # bound parameter - which is why this script takes it as an argument you
    # supply rather than accepting it from anywhere untrusted.
    if not service_principal_id or '"' in service_principal_id:
        print("Refusing: pass the app's service principal client id as one argument.")
        return 2

    statements = [
        f'GRANT USAGE ON SCHEMA movie_night TO "{service_principal_id}"',
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA movie_night '
        f'TO "{service_principal_id}"',
        f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA movie_night '
        f'TO "{service_principal_id}"',
        # Tables created later (a re-run of bootstrap_db.py) should inherit the
        # same access, otherwise the app breaks again after the next migration.
        f'ALTER DEFAULT PRIVILEGES IN SCHEMA movie_night '
        f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "{service_principal_id}"',
    ]

    with connection() as conn:
        for statement in statements:
            conn.execute(statement)
            print("OK:", statement[:78])
        conn.commit()

        rows = conn.execute(
            """
            SELECT privilege_type, count(*) AS n
            FROM information_schema.role_table_grants
            WHERE table_schema = 'movie_night' AND grantee = %(sp)s
            GROUP BY privilege_type ORDER BY privilege_type
            """,
            {"sp": service_principal_id},
        ).fetchall()

    if not rows:
        print("\nNo grants visible - did STEP 1 (create-database-instance-role) run?")
        return 1

    print("\nverified:")
    for row in rows:
        print(f"  {row['privilege_type']:8s} on {row['n']} tables")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
