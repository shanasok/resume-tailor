# Dockerfile — the resume-tailor web app in a container.
#
# The image needs three unusual things beyond the Python deps:
#   1. Chromium (via Playwright) — renders the résumé HTML to PDF.
#   2. Node.js + the `claude` CLI — resume tailoring runs `claude -p` as a subprocess
#      using your Claude Pro subscription (NOT the API). Inside the container the CLI
#      can't reach your Mac's keychain login, so auth comes from CLAUDE_CODE_OAUTH_TOKEN
#      in .env — generate it once on the host with:  claude setup-token
#   3. The Playwright system libraries (--with-deps installs them via apt).
#
# Installs are pinned via requirements.lock for reproducible builds (regenerate it per
# the header in requirements.txt). The app runs as a non-root user; Chromium therefore
# needs --no-sandbox (set in src/renderer.py — the only rendered content is our own).
#
# Build/run via docker-compose.yml (which also starts Postgres):  docker compose up

FROM python:3.12-slim

# Node.js (for the claude CLI). nodesource ships current Node for Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg gosu \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Browsers live in a fixed shared path (not root's home) so the non-root user below
# can use them.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Python deps first (own layer → cached unless the lock changes), then the Chromium
# browser + its system libraries for the HTML→PDF step.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# The app itself (see .dockerignore for what's excluded — secrets, personal data, venv).
COPY . .

# Run as a non-root user: a bug in a parsed job posting or a dependency shouldn't get
# root in the container. The volume-mounted dirs (resumes/, data/, logs/) are created
# here so first run doesn't fail on permissions when the host dirs don't exist yet.
#
# The container STARTS as root so docker-entrypoint.sh can chown the bind-mounted dirs
# (whose host ownership shadows this build-time chown), then drops to appuser via gosu.
# So the app process is still non-root — only the brief entrypoint chown runs as root.
RUN useradd --create-home appuser \
    && mkdir -p resumes data logs \
    && chown -R appuser:appuser /app /ms-playwright \
    && chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Listen on all interfaces (required to be reachable from outside the container) and
# turn off Flask's debug/reloader in the container.
ENV HOST=0.0.0.0 \
    FLASK_DEBUG=0 \
    PORT=5001

EXPOSE 5001

CMD ["python", "app.py"]
