# Security & Abuse-Prevention Plan — Public Multi-User Readiness

Plan for taking Resume Handler from a trusted **local single-user** app to a **public multi-user web app**.
Every item is grounded in the current code. Severity: **P0** = fix before any public exposure, **P1** = soon after, **P2** = hardening.

> Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## 1. Input size limits — the "huge resume" / huge-input problem
Oversized input is both a **DoS** and a **cost** attack: pasted job text and resume content flow straight into LLM prompts (paid per token) and into Playwright PDF rendering (CPU/memory).

- [ ] **P0** Global request cap: `app.config['MAX_CONTENT_LENGTH']` (~256 KB). `job_text` is currently unbounded.
- [ ] **P0** Per-field caps server-side: company/role ≤ 200, `job_text` ≤ ~20k chars, chat message ≤ 2k, `boards_limit` integer-bounded. Reject with a clean 400.
- [ ] **P0** Token budgeting before the LLM: fetched postings cap at `_MAX_WORDS = 1200`, but **pasted** `job_text` and the base resume bypass that. Apply a central budget in `tailoring.py`/`critic.py`.
- [ ] **P1** Cap model *output*: bound the tailored JSON (max experience entries, bullets/entry, chars/bullet) before rendering.
- [ ] **P1** Hard page cap: enforce a ceiling on `count_pdf_pages` instead of "best effort."
- [ ] **P0** Rate limiting (Flask-Limiter) per user/IP: generations 10/hr, chat 30/hr, retrieve 5/hr.
- [ ] **P1** Per-user budget/quota + job queue + per-user concurrency = 1 for LLM+PDF work.

## 2. Prompt injection defenses
Prompts are built by concatenating attacker-controllable content — pasted job text, **fetched web pages**, chat messages, company/role. The chat feature can **write to the base resume DOCX**, so injection isn't only cosmetic.

- [ ] **P0** Treat all external content as untrusted data everywhere. The *critic* prompt already says so — add the same guardrail to `tailoring.py` and `chat.py`.
- [ ] **P0** Strong delimiting + role separation. Fence untrusted blocks in tags; prefer an API call with real system/user separation over `claude -p` single-stdin concatenation.
- [ ] **P0** Lock down chat→resume write (`resume_editor.add_experience`): explicit user confirmation, length/charset validation, undo.
- [ ] **P1** Output sanitization / XSS: confirm Jinja autoescaping on `resume.html`; audit for `|safe`.
- [ ] **P1** Injection-resistant JSON parsing (`_extract_json` can't be tricked into a smuggled second object).
- [ ] **P2** Output moderation / content-policy checks.

## 3. SSRF — standout risk given the private network ⚠️
`job_url` is fetched **server-side** (`job_parser._fetch_url`), and headless Chromium renders HTML that can reference external/`file://` resources. An attacker can target `169.254.169.254` (cloud metadata), `localhost`, or the **internal Tailscale LLM (`100.108.61.107:11434`)**.

- [x] **P0** Allowlist schemes to `http`/`https` only. *(`job_parser._assert_public_url`)*
- [x] **P0** Resolve DNS and block private/loopback/link-local/CGNAT(Tailscale) ranges; re-check after redirects. *(`_is_blocked_ip`, `_safe_get`)*
- [x] **P0** Disable auto-following redirects to internal targets; strict timeouts; cap response size. *(`_safe_get`, `_MAX_FETCH_BYTES`)*
  - Note: `_fetch_url`'s main fetch now uses `_safe_get`; `_linkedin_description` still calls `requests.get` directly (fixed public host `linkedin.com`, low risk) — route it through `_safe_get` for consistency.
- [ ] **P1** Sandbox the PDF renderer: Chromium with no network / no `file://` access (or locked-down container).

## 4. AuthN / AuthZ / multi-tenancy (backlog #11)
Today: **no auth**, one **global** CSV store, one **shared** base-resume DOCX.

- [ ] **P0** Real authentication (hashed passwords via bcrypt/argon2, or OAuth); secure sessions.
- [ ] **P0** Per-user data isolation: scope every resume, store row, and file by `user_id` — including the base resume.
- [ ] **P0** Authorization on every route (`/files`, `/regenerate-resume`, `/generate-for-job`, `/chat`, `/update-email`).
- [ ] **P0** CSRF tokens on all POST forms (Flask-WTF).

## 5. File handling & path traversal
- [ ] **P0** Audit `/files/<path:filename>` for safe-join / traversal; ideally serve by opaque ID.
- [ ] **P0** Sanitize output filenames (`renderer._slug` strips `/` and spaces but not `..`/dots) to `[a-z0-9_-]` + length cap; write into per-user dirs.
- [ ] **P1** Rotate/limit DOCX backups (`*.backup-<ts>.docx`) and generated files (disk exhaustion).

## 6. Secrets & deployment config
- [ ] **P0** `app.secret_key` is hardcoded → sessions forgeable. Load random secret from env.
- [ ] **P0** `debug=True` → Werkzeug debugger is RCE. Off in prod; run under gunicorn behind nginx.
- [ ] **P0** Keep API keys out of repo/logs. (API keys live in `.env`, which is gitignored + `.dockerignore`d; the Claude API key `CLAUDE_CODE_API_KEY` is passed straight to the SDK, never logged.)

## 7. Web hardening
- [ ] **P1** Security headers: CSP, `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, HSTS.
- [ ] **P1** HTTPS/TLS; cookies `Secure` + `HttpOnly` + `SameSite=Lax`.
- [ ] **P1** Generic error pages (no stack traces) once `debug` is off.

## 8. Data store & concurrency (backlog #8)
- [ ] **P0 (multi-user)** The `threading.RLock` in `tracker.py` is per-process — multiple gunicorn workers can corrupt the CSV. Migrate CSV → Postgres (per-user rows, parameterized queries, transactions).
- [ ] **P1** PII: encryption at rest, retention/delete (GDPR/CCPA), backups, audit logging.

## 9. Operations / abuse monitoring
- [ ] **P1** Job queue + worker pool; per-user concurrency caps.
- [ ] **P2** Cost/abuse alerting; structured logging without PII/secrets; audit trail of generations and DOCX writes.

---

## Suggested order
1. **P0 gate before public exposure:** SSRF (#3), input/rate limits (#1), secrets+debug+WSGI (#6), auth + isolation + CSRF (#4), path traversal (#5), prompt-injection guardrails (#2).
2. **P1:** Postgres + concurrency (#8), output caps + XSS/CSP (#1/#2/#7), renderer sandbox (#3).
3. **P2:** moderation, monitoring, retention.

**Don't-ship-without:** SSRF (#3) and the chat→DOCX write path under prompt injection (#2).
