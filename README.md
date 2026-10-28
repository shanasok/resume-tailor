# resume-tailor

An AI job-search assistant that writes a tailored résumé for each job, and uses a
second AI to check that nothing in it was made up.

## What it does

- **Tailors your résumé to a job posting.** Give it your base résumé and a job
  posting (a link or pasted text). It first maps each requirement in the posting to
  your real experience, then writes a version that leads with your strongest true
  matches. You get an HTML preview and a PDF ready for applicant tracking systems.
- **Checks the result for honesty.** One AI writes the draft, another (ChatGPT)
  suggests ways to make it fit the job better, and a separate AI critic then checks
  every claim against your original résumé. Anything invented, inflated, copied
  from the posting, or written in an obvious "AI voice" is flagged and fixed
  before the résumé is accepted.
- **Finds jobs for you.** It scans thousands of company job boards (Greenhouse,
  Lever, Ashby, Workday, Rippling, SmartRecruiters) and keeps only the jobs that pass
  your rules: location, salary floor, remote or hybrid, and full-time.
- **Ranks what it finds.** Each kept job gets a 1–10 "AI Fit" score with a one-line
  reason, so the best matches are at the top of the list.
- **Tracks everything.** Every job, generated résumé (with full version history), and
  application is stored in a local database, with a web page to browse and manage it.

**Built with:** Python, Flask, PostgreSQL with pgvector (for comparing text by
meaning), Docker, Playwright (HTML to PDF), and the Claude, OpenAI, and local Ollama
AI models. You can choose which AI runs each step.

---

## About this README

The app takes your resume and a job posting, and generates a new resume tailored to
that job. It also keeps a history of every job you found, every resume it generated, and every
application you made. All of this is stored in a PostgreSQL database that runs in
Docker on your own computer. Nothing is sent anywhere except the text sent to the AI
models during generation.

The app can run in two ways:

- **Everything in Docker** — the app and the database both run in Docker containers.
  Best when you just want to use the app.
- **App on your machine, database in Docker** — best when you are changing the code,
  because your changes take effect the moment you save a file.

This README covers both, plus how to get code changes into the Docker version.

---

## 1. First-time setup (everything in Docker)

You only do this once.

**You will need:**

- Docker Desktop installed and running.
- A Claude Pro or Max subscription, and the `claude` command-line tool installed on
  your machine: `npm install -g @anthropic-ai/claude-code`, then run `claude` and use
  `/login` to sign in.
- Optional: an OpenAI API key. The app uses it for a second review of each resume.
  Without it, the app still works and simply skips that step.

**Steps:**

```bash
# 1. Get the code
git clone https://github.com/shanasok/resume-tailor.git
cd resume-tailor

# 2. Create your settings file from the template
cp .env.example .env

# 3. Create a Claude access token for the app to use.
#    (The app running inside Docker cannot use the login stored on your machine,
#    so it needs its own token.)
claude setup-token
```

Open `.env` in an editor and add the token you just created:

```
CLAUDE_CODE_OAUTH_TOKEN=<paste the token here>
CHATGPT_API_KEY=<your OpenAI key, if you have one>
```

Then build and start the app:

```bash
# 4. Build and start. The first build downloads a browser and other tools,
#    so it takes a few minutes. Later starts are instant.
docker compose up -d --build

# 5. Open the app in your browser (use 127.0.0.1, not localhost):
#    http://127.0.0.1:5001
```

The first time it starts, the database sets itself up automatically. The home page
will ask you to upload your base resume (a `.docx` file) — this is the resume all
tailored versions are built from.

**If port 5432 is already in use on your machine** (you already run PostgreSQL):
in `docker-compose.yml`, under the `db` service, change `"5432:5432"` to
`"5433:5432"`, and add this line to `.env`:
`DATABASE_URL=postgresql://resume:resume@localhost:5433/resume_handler`

---

## 2. Running the app day-to-day

```bash
docker compose up -d      # start the app and the database
docker compose down       # stop them
```

The `-d` means "run in the background". Without it, the command works but holds your
terminal open, and pressing Ctrl-C stops the app.

**You often don't need any command at all.** The containers are set to start again
automatically whenever Docker Desktop starts. So the usual routine is: open Docker
Desktop, then open http://127.0.0.1:5001 in your browser. Only if the page doesn't
load, run `docker compose up -d`.

Starting and stopping never affects your data — it is stored in ordinary files on
your computer (see section 6).

Other useful commands:

```bash
docker compose ps           # is it running? is it healthy?
docker compose logs -f app  # watch what the app is doing
```

---

## 2a. Résumé history (getting an older version back)

Every résumé has two "regenerate" buttons — **↻ From JSON** (re-makes the PDF and
re-scores it) and **✎ With LLM** (a full AI rewrite). Sometimes a rewrite comes out
*worse* than what you had. You no longer lose the good one.

**Before either button changes anything, the app saves a copy of the current résumé.**
Nothing is ever thrown away.

To get an older version back:

1. On the job's row, click **🕑 History**.
2. A window opens listing every earlier version, newest first — with the date it was
   saved, its match score, and which AI made it.
3. For any version you can:
   - **Preview** — open it in a new tab to read it.
   - **PDF ↓** — download that version's PDF.
   - **Restore** — make that version the current résumé again.

**Restore is safe.** When you restore an old version, the app saves your *current*
résumé to history first, then swaps the old one in. So you can always change your mind
and restore the other one back. Nothing is ever permanently overwritten.

The saved versions are ordinary files on your computer, in a `versions` folder next to
each résumé (see section 6).

---

## 3. Developing: app on your machine, database in Docker

This is the best way to work on the code. The app runs as a normal Python program on
your machine, so every time you save a file, it restarts itself with your change.
Only the database runs in Docker.

**One-time preparation** (skip if you have already done this):

```bash
python3 -m venv .venv                 # create a Python environment for this project
source .venv/bin/activate
pip install -r requirements.txt       # install the app's Python packages
playwright install chromium           # install the browser used to create PDFs
```

**Every time you want to develop:**

```bash
docker compose stop app     # stop the Docker copy of the app so the port is free
docker compose up -d db     # make sure the database is running
source .venv/bin/activate
python app.py
```

Then open http://127.0.0.1:5001 as usual. When you're done, press Ctrl-C to stop it.

Two notes:

- In this mode, Claude uses the login already stored on your machine. No token needed.
- `docker compose up -d db` — the word `db` at the end means "only start the service
  named db" (the database). Without it, Docker would also start its own copy of the
  app, which would conflict with the one you are running.

---

## 4. Getting your code changes into the Docker version of the app

This part confuses everyone at first, so here is what is actually going on:

When Docker builds the app, it **copies** your code into the image (an image is a
frozen snapshot that containers run from). The running container uses that copy, not
your files. So when you edit code on your machine, **the running Docker app does not
see the change**. It keeps running the old copy until you rebuild.

You do NOT need to stop anything, delete anything, or rebuild from scratch. One
command does everything — rebuilds the image with your current code, then replaces
the running container with a new one:

```bash
docker compose up -d --build app
```

A rebuild after a code change takes only a few seconds, because Docker reuses the
slow parts (installing Python packages, the browser, and so on) from the previous
build. Your data is not affected.

**When do you need this?** Only when you want the Docker version to run your latest
code — for example, after finishing a round of changes you made in dev mode
(section 3). Also run it after changing `requirements.lock` or the `Dockerfile`.

**Housekeeping:** each rebuild leaves the old image behind as unused junk. Every so
often, run `docker image prune -f` and `docker builder prune` to clear it (this
freed 55 GB on the machine this was developed on).

**Alternative — see changes in Docker without rebuilding:** there is a second compose
file that connects the running container directly to the code on your machine and
turns on auto-restart, so saves take effect immediately:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up app
```

This is only worth it when you need to test behavior that is specific to the
container itself. For normal work, section 3 is simpler.

---

## 5. Which copy of the app am I talking to?

The Docker app and the local dev app both use port 5001, so it's possible to be
unsure which one answered your browser. Ask the app itself:

```bash
curl http://127.0.0.1:5001/health
# {"ok":true,"runtime":"docker"}   <- the Docker container answered
# {"ok":true,"runtime":"local"}    <- your local `python app.py` answered
```

To avoid confusion, run only one at a time: `docker compose stop app` before
starting the local one, and `docker compose start app` when you're done.

If you want to see every program listening on the port:
`lsof -nP -iTCP:5001 -sTCP:LISTEN` — `Python` is the local app,
`com.docker.backend` is Docker.

---

## 6. Where your data is stored

Everything personal is stored as ordinary files **inside this project folder** on
your computer. Deleting Docker containers, images, or even uninstalling Docker
Desktop cannot delete this data. None of it is ever committed to git.

| Folder / file | What's in it |
| --- | --- |
| `postgres_data/` | The database's files — all job, resume, and application history |
| `resumes/baseResume.docx` | Your base resume (uploaded through the app) |
| `resumes/backups/` | Automatic copies of previous base resumes |
| `resumes/generated_resumes/` | The generated resumes — one `.html`, `.pdf`, and `.json` per job (scheduled runs go into dated subfolders like `20260705/`) |
| `data/filter_config.json` | Your job-filter settings |
| `data/scheduled_runs.json` | The state of scheduled batch runs |
| `data/Connections.csv` | Optional: your LinkedIn connections export |
| `backups/` | Database backup files you create (see below) |
| `.env` | Your keys and tokens |

Because this is all ordinary files, your normal computer backup (such as Time
Machine) protects it. The one thing Docker can't protect you from is deleting the
files yourself — `rm -rf` on this folder, or `git clean -fdx`, would remove them. So
it's worth occasionally saving a database backup:

```bash
docker compose exec db pg_dump -U resume resume_handler > backups/backup-$(date +%Y%m%d).sql
```

To look inside the database directly:

```bash
docker compose exec db psql -U resume resume_handler
# then, for example:  SELECT company, role, match_score FROM job_overview;
```

---

## 7. Using the command line instead of the web page

```bash
source .venv/bin/activate
python main.py --company "Johnson & Johnson" \
               --role "Sr Program Manager - AI Literacy & Adoption" \
               --job-url "https://www.linkedin.com/jobs/view/..."
```

Other ways to provide the job posting:
- `--job-file path/to/posting.txt` — read it from a text file
- `--job-text "paste the description here"` — pass the text directly
- `--render-only` — recreate the HTML and PDF from saved data without calling the AI
  again (useful after editing the resume template)

---

## 8. Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

The database tests need the Docker database to be running. They create their own
separate throwaway database (`resume_handler_test`) and never touch your real data.

---

## 9. How the app finds and ranks jobs (AI Fit)

When you click **Retrieve Jobs**, the app walks through the company job boards it
knows about, several boards at a time. Each board's jobs go through every step
below as soon as that board is read, so good matches start appearing on the home
page within the first few minutes. You don't have to wait for the whole scan to
finish. The scan card shows a live feed of the latest jobs it kept, and the page
reloads by itself when the scan is done. If a scan is interrupted, everything it
already processed is kept.

For each job, the app decides what to keep in four steps:

1. **A wide keyword net.** A job's title only needs ONE relevant word or phrase
   from your keyword lists (for example "engineering manager" or "senior engineer")
   to be looked at further. Generic words alone ("manager", "director",
   "software") are not enough, so "Account Manager" stays out.
2. **A meaning check for unusual titles.** Titles with none of your keywords get one
   more chance: the AI compares the title's *meaning* against the "Target roles"
   paragraph on the Settings page, so a well-matched role can still get in even
   though you never listed its exact words.
3. **Hard rules that always apply.** A job is dropped (never stored) when it is
   not in the United States, states no salary, states a salary under your floor
   (set on the Settings page), is on-site, or is hybrid outside the commute zone
   you configured in Settings. Fully remote US jobs pass from anywhere. Contract,
   part-time, and old postings are dropped too. The app remembers every job it
   dropped here, so later scans skip it straight away.
4. **Store and score.** A job that passes is saved together with its full
   description, and an AI immediately gives it an **AI Fit** score out of 10 with
   a one-line reason. Hover over the score on the home page to read the reason.
   The job list is sorted by AI Fit, best first.

Things you may want to do:

- **Tune what gets through.** Open **Settings** and edit the keyword lists, the
  "Target roles" paragraph, the meaning-match threshold, the salary floor, the
  commutable locations, or how old a posting can be.
- **Choose which AI scores the jobs.** In the "LLM selection" section of the home
  page, the "Job scan - AI Fit verdict" picker chooses ChatGPT (the default),
  Claude, or a local model.
- **Score jobs that have no AI Fit yet.** Click **Score fits (AI)** next to
  Retrieve Jobs. It scores every stored job that doesn't have a score yet, using
  the description saved with it.
- **Remove a job you don't want.** Click 🗑 on its row. A deleted job never comes
  back in a later scan.
- **After replacing your base resume.** Old AI Fit scores are cleared
  automatically, because they were judged against the old resume. Click
  **Score fits (AI)** to score the jobs again.

Old postings are hidden from the job list as well as from scans, based on the
posting-age setting. Jobs you applied to or generated a resume for always stay
visible. Nothing is deleted, so widening the age setting brings hidden jobs back.

**What it costs, and running without an OpenAI key.** By default the meaning check
uses OpenAI to compare text (a fraction of a cent per thousand jobs) and ChatGPT
for AI Fit (a few cents per new job; jobs that already have a score aren't scored
again). Without an OpenAI key the scan still works on keywords alone. To keep every
feature without paying for any AI, set `EMBEDDING_BACKEND=local` in `.env` (this
uses a free local model through Ollama) and pick Claude or a local model in the
AI Fit picker.

---

## Notes

- When you pick **Claude** in the app, it uses one of two things: if you have put a
  `CLAUDE_CODE_API_KEY` in your `.env` file, it calls the paid Claude API with that key
  (this is the cheaper, faster option and never hits a daily usage limit). If you have
  not set that key, it falls back to the `claude` command-line tool, which uses your
  Claude.ai subscription instead. You can change which Claude model the API uses by
  setting `CLAUDE_API_MODEL` in `.env`.
- If you have an old `resumes/applications.csv` or `logs/generation_log.csv` from a
  version of this app that used CSV files, they are imported into the database
  automatically the first time the app runs against an empty database. The CSV files
  themselves are left untouched, as a backup.
