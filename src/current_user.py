"""The current-user seam — the ONE place the app decides "who is this request for?".

The whole app is being built as if every request is already authenticated to its own
data (see the multi-tenancy design): every user-owned query is scoped by a user_id.
Real login/authentication isn't built yet, so for now this resolves to a single default
user. When sessions/auth land, `current_user_id()` is the only function that changes —
it starts reading the logged-in user from the request/session instead of returning the
default.

Keeping this behind one function means routes, the tracker, and the pipeline can already
take and pass a real `user_id` today, and nothing else has to change when auth arrives.
"""

# The seeded default user (see db.ensure_default_user / the v8 migration). Everything
# that exists today is owned by this user; a fresh install gets it too.
DEFAULT_USER_ID = 1


def current_user_id() -> int:
    """Which user this request/operation belongs to.

    For now: always the default user (no auth yet). LATER: read the authenticated user
    from the Flask session here — this is the single insertion point, so callers that
    already pass the returned id keep working unchanged.
    """
    return DEFAULT_USER_ID
