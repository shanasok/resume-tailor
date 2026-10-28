#!/bin/sh
# Container entrypoint: fix ownership of the bind-mounted data dirs, then drop to the
# non-root appuser to run the app.
#
# Why this exists: resumes/, data/ and logs/ are bind-mounted from the host, so the
# HOST's ownership shadows the image's build-time `chown`. A dir the host created as
# root — e.g. the Docker daemon auto-creating a missing mount source — is then owned
# by root:0755 and UNWRITABLE by appuser (uid 1000), which crashed the app the first
# time it tried to write to logs/ (EACCES). Running this entrypoint as
# root, chowning the mounts, then exec'ing the app via `gosu appuser` keeps the app
# process non-root while GUARANTEEING it can always write its mounted dirs — regardless
# of how (or by whom) the host dirs were created. This is the durable fix for that bug.
set -e

if [ "$(id -u)" = "0" ]; then
    # logs/ is small — make the whole tree writable; resumes/ can be large, so only the
    # top-level mount point needs to be appuser-owned for new files/subdirs.
    chown -R appuser:appuser /app/logs 2>/dev/null || true
    chown appuser:appuser /app/resumes /app/data 2>/dev/null || true
    exec gosu appuser "$@"
fi

# Already non-root (e.g. run with an explicit --user): just exec the command.
exec "$@"
