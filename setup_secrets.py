"""
One-time setup: store TMDB credentials in a Databricks secret scope.

Uses getpass, so the values are never echoed to the terminal, never land in
shell history, and never get written to a file in this repo.

    python setup_secrets.py

Run it once from a machine with the Databricks CLI authenticated. Re-running is
safe: the scope creation is tolerated if it already exists, and putting a secret
overwrites the previous value, which is how you rotate a leaked key.

WHAT GETS STORED
----------------
TMDB issues two credentials, and they are used differently:

  tmdb-api-key      the v3 API Key. Passed as an `api_key=` query parameter.
  tmdb-read-token   the v4 API Read Access Token. Passed as an
                    `Authorization: Bearer <token>` header.

Both are stored because TMDB's own docs use each in different places. The
application code should prefer the read token (header auth keeps the credential
out of URLs, which otherwise leak into proxy logs and browser history), and
falls back to the api key only where an endpoint requires it.

READING THEM BACK
-----------------
Application code never hardcodes these. It reads them at runtime:

    from databricks.sdk import WorkspaceClient
    token = WorkspaceClient().secrets.get_secret(
        scope="movie_night", key="tmdb-read-token"
    ).value                      # base64-encoded; decode before use

In a Databricks App, prefer a `valueFrom` entry in app.yaml so the platform
injects the secret as an environment variable and the app never calls the
secrets API itself.
"""

import base64
import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

SCOPE = "movie_night"

# (key, prompt, minimum plausible length) - the length check catches a truncated
# paste or an accidental Enter, which otherwise fails much later as a confusing
# 401 from TMDB.
SECRETS = [
    ("tmdb-api-key", "TMDB API Key (v3, ~32 chars)", 20),
    ("tmdb-read-token", "TMDB API Read Access Token (v4, a long JWT)", 100),
]


def ensure_scope(w):
    try:
        w.secrets.create_scope(scope=SCOPE)
        print(f"created secret scope {SCOPE!r}")
    except Exception as exc:
        # Almost always RESOURCE_ALREADY_EXISTS on a re-run, which is fine.
        print(f"scope {SCOPE!r} already exists or could not be created: {exc}")


def prompt_and_store(w, key, label, min_len):
    value = getpass.getpass(f"Paste {label} (input hidden): ").strip()

    if not value:
        print(f"  nothing entered - skipping {key!r}, existing value untouched")
        return False

    if len(value) < min_len:
        print(
            f"  refusing to store {key!r}: got {len(value)} characters, expected "
            f"at least {min_len}. Looks like a truncated paste - re-run."
        )
        return False

    w.secrets.put_secret(scope=SCOPE, key=key, string_value=value)
    print(f"  stored {SCOPE}/{key}")
    return True


def verify(w, key):
    """Confirm the round-trip without ever printing the value."""
    try:
        raw = w.secrets.get_secret(scope=SCOPE, key=key).value
        length = len(base64.b64decode(raw).decode())
        print(f"  {key}: readable, {length} characters")
    except Exception as exc:
        print(f"  {key}: could NOT be read back - {exc}")


def main():
    w = WorkspaceClient()
    print(f"workspace: {w.config.host}\n")

    ensure_scope(w)

    stored = [
        key
        for key, label, min_len in SECRETS
        if prompt_and_store(w, key, label, min_len)
    ]

    if not stored:
        print("\nNothing was stored.")
        return 1

    # The app runs as its own service principal and reads this scope at startup;
    # granting the users group READ keeps it working for anyone you demo it to.
    try:
        w.secrets.put_acl(
            scope=SCOPE, principal="users", permission=workspace.AclPermission.READ
        )
        print(f"\ngranted READ on {SCOPE} to 'users'")
    except Exception as exc:
        print(f"\ncould not set ACL on {SCOPE}: {exc}")

    print("\nVerifying round-trip:")
    for key in stored:
        verify(w, key)

    print(f"\nKeys now in scope {SCOPE!r}:")
    for item in w.secrets.list_secrets(scope=SCOPE):
        print(f"  {item.key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
