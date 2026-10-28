# Python Primer (for this codebase)

You don't need to know Python to follow the architecture, but when you read the actual
code you'll hit syntax that may be unfamiliar. This explains the Python concepts this
project uses, **with real examples taken from these files**. It's not a full Python course —
just the things you'll actually see here.

> Tip: read this alongside the code. Each section names the concept, shows a snippet from
> the project, and explains it in plain English.

---

## 1. Modules and `import`

A **module** is one `.py` file. To use code from another file, you `import` it.

```python
# from src/pipeline.py
from src.resume_parser import parse_resume
from src.tailoring import tailor_resume
```

- `from src.resume_parser import parse_resume` means "from the file `src/resume_parser.py`,
  grab the `parse_resume` function so I can call it here."
- `src` is a folder (a **package**); `src.ats.greenhouse` means the file
  `src/ats/greenhouse.py`.

The `__init__.py` files you see (e.g. `src/__init__.py`) are usually empty — their presence
just tells Python "this folder is a package you can import from."

---

## 2. Functions: `def`

A **function** is a named, reusable block of code. You "call" it to run it.

```python
# from src/renderer.py
def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in the generated PDF."""
    from pypdf import PdfReader
    return len(PdfReader(str(pdf_path)).pages)
```

- `def count_pdf_pages(...)` — defines a function named `count_pdf_pages`.
- `pdf_path: Path` — the input ("parameter") named `pdf_path`. The `: Path` is a **type
  hint** (section 7) — a note that it should be a file path. Hints are documentation; Python
  doesn't enforce them.
- `-> int` — another hint: this function gives back ("returns") an integer.
- The triple-quoted string right under `def` is a **docstring** — built-in documentation for
  the function. Most functions here have one.
- `return X` — hands `X` back to whoever called the function.

You call it like: `pages = count_pdf_pages(my_pdf)`.

### Default values and keyword arguments

```python
# from src/pipeline.py
def generate(company, role, source, max_pages=2, max_attempts=3, ...):
```

`max_pages=2` means "if the caller doesn't supply `max_pages`, use 2." Callers can also
pass arguments **by name** for clarity:

```python
generate("Acme", "Director", source, max_attempts=5)
```

---

## 3. Variables and basic types

```python
company = "Acme Corp"     # a string (text), in quotes
max_pages = 2             # an integer (whole number)
polite_delay = 0.2        # a float (decimal number)
use_external_critic = True # a boolean (True / False)
suggestions = None        # "None" = "no value / nothing here"
```

`None` is important and everywhere in this code: it's Python's way of saying "empty / not
set." You'll often see checks like `if suggestions is not None:` ("if we actually have
suggestions").

---

## 4. Lists, dictionaries, and tuples

These three "container" types hold multiple values.

### List — an ordered sequence (square brackets)

```python
# from src/matching.py
_REMOTE_MARKERS = ["remote", "distributed", "work from home", "wfh", "anywhere"]
```

A list of strings. You can loop over it, check membership (`"remote" in _REMOTE_MARKERS`),
count it (`len(...)`), etc.

### Dictionary ("dict") — labeled values (curly braces, `key: value`)

```python
# from app.py
summary = {"total": total, "applied": applied, "generated": generated,
           "no_resume_yet": no_resume_yet, "not_applied": not_applied}
```

A dict maps **keys** to **values**. You read a value by its key: `summary["total"]`. Much of
this app passes data around as dicts — e.g. a "job" is a dict with keys like `"title"`,
`"location"`, `"url"`. A safe way to read a key that might be missing is `.get()`:

```python
job.get("title", "")   # the title, or "" (empty string) if there's no "title" key
```

### Tuple — a fixed, ordered group (parentheses)

```python
# from src/job_parser.py
return parse_job_posting(job_url) or "", job_url   # returns a 2-tuple: (text, url)
```

A tuple is like a list but typically a fixed-size bundle. Functions often return a tuple to
hand back several things at once. The caller "unpacks" it:

```python
description, source = fetch_job_description(job_id, job_url)
```

This puts the first returned value into `description` and the second into `source`.

---

## 5. `if` / `else`, and truthiness

```python
# from src/pipeline.py
if accepted:
    break
elif attempt < max_attempts:
    feedback = _build_feedback(...)
```

- `if CONDITION:` runs the indented block only when the condition is true.
- `elif` = "else if" (check another condition). `else:` = "otherwise."
- **Indentation matters in Python** — the indented lines under an `if` are "inside" it.
  There are no `{ }` braces like some languages; the spacing *is* the structure.

**Truthiness:** Python treats some values as "falsy" — `None`, `0`, `""` (empty string), and
`[]` (empty list) all count as false in an `if`. So `if suggestions:` means "if suggestions
is not empty/None." That's why you see `if not company or not role:` ("if either is empty").

---

## 6. Loops

```python
# from src/job_search.py
for board in boards:
    jobs = greenhouse.list_jobs(board["token"])
    ...
```

`for X in COLLECTION:` runs the indented block once per item, with `X` set to each item in
turn. `while CONDITION:` loops as long as the condition stays true.

- `break` — exit the loop immediately.
- `continue` — skip to the next item.
- `range(1, max_attempts + 1)` — produces the numbers 1, 2, …, max_attempts (used in
  `pipeline.py` to count attempts).

### List comprehensions (a compact loop that builds a list)

This is very common here and looks odd at first:

```python
# from src/critic.py
[str(x).strip() for x in (review.get("revisions") or []) if str(x).strip()]
```

Read it right-to-left-ish: "for each `x` in the revisions list, IF it's non-empty after
trimming, include `str(x).strip()` in the new list." It's shorthand for a `for` loop that
builds a list. The equivalent long form:

```python
result = []
for x in (review.get("revisions") or []):
    if str(x).strip():
        result.append(str(x).strip())
```

---

## 7. Type hints

```python
def list_jobs(token: str) -> list[dict]:
```

- `token: str` — the parameter `token` is expected to be a string.
- `-> list[dict]` — the function returns a list of dicts.
- `tuple[str, str] | None` (seen in `tracker.py`) — returns either a 2-tuple of strings
  **or** `None`. The `|` means "or."

Again: hints are **documentation and editor help**, not enforced rules. They make the code
self-describing.

---

## 8. f-strings (formatted text)

```python
# from src/tracker.py
return f"{job.get('source', 'greenhouse')}:{job['board_token']}:{job['id']}"
```

A string prefixed with `f` lets you drop values inside `{ }`. So if source is "greenhouse",
board_token is "stripe", id is "42", this produces `"greenhouse:stripe:42"`. f-strings are
used everywhere for building messages, filenames, and prompts.

---

## 9. Exceptions: `try` / `except` / `raise`

When something can fail (a network call, bad data), Python "raises an exception." You handle
it with `try`/`except`:

```python
# from src/ats/greenhouse.py
try:
    resp = requests.get(url, ...)
    resp.raise_for_status()       # raise an error if the server said "not OK"
    jobs = resp.json().get("jobs", [])
except (requests.RequestException, ValueError) as e:
    raise GreenhouseError(f"Could not list jobs for board '{token}': {e}") from e
```

- Code in the `try:` block is "attempted."
- If it fails with one of the listed error types, the `except` block runs instead of the
  program crashing. `as e` captures the error so you can include it in a message.
- `raise SomeError(...)` deliberately signals a failure up to the caller.

You'll also see `except Exception: pass` — "if anything goes wrong here, ignore it and move
on" — used for best-effort steps that shouldn't crash the whole run.

---

## 10. Classes (used sparingly here)

A **class** is a blueprint for objects that bundle data + behavior. This codebase is mostly
plain functions, but you'll see a few classes:

```python
# from src/ats/greenhouse.py
class GreenhouseError(RuntimeError):
    """Raised when a board can't be read (bad token, network error, non-200)."""
```

This defines a custom **error type** so failures from the Greenhouse connector can be caught
specifically (`except greenhouse.GreenhouseError`). `(RuntimeError)` means it's a kind of
the built-in `RuntimeError`.

---

## 11. Decorators (`@something`)

A line starting with `@` above a function "wraps" it with extra behavior.

```python
# from app.py (Flask web routes)
@app.route("/generate", methods=["POST"])
def generate_route():
    ...
```

`@app.route("/generate", ...)` tells Flask "when a browser sends data to the `/generate`
web address, run this function." You don't call `generate_route` yourself — Flask does, when
a request comes in.

```python
# from src/tracker.py
@_synchronized
def log_application(...):
```

`@_synchronized` wraps the function so only one of these can run at a time — preventing two
simultaneous web requests from corrupting the CSV file.

---

## 12. `with` blocks (automatic cleanup)

```python
# from src/tracker.py
with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
```

`with ... as f:` opens a resource (here a file) and **guarantees it's closed** when the block
ends, even if an error happens. `f` is the open file, used inside the block.

---

## 13. `async` / `await` (only in the PDF renderer)

```python
# from src/renderer.py
async def _html_to_pdf(html_path, pdf_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ...
```

`async`/`await` is Python's way of doing things that involve waiting (here, driving a
headless browser to print a PDF) without blocking. You can mostly read it as normal code;
`await` means "start this and wait for it to finish." `asyncio.run(...)` kicks the whole
async function off from regular code. This is the only place the project uses it.

---

## 14. Paths with `pathlib`

```python
# from src/renderer.py
OUTPUT_DIR = Path(__file__).parent.parent / "resumes" / "generated_resumes"
```

`Path` is a clean way to work with file locations:

- `__file__` — the path of the current `.py` file.
- `.parent` — the folder containing it (`.parent.parent` = two folders up).
- The `/` operator joins path pieces (it's not division here!). So this builds the path to
  `resumes/generated_resumes/` relative to the project, on any operating system.

Common helpers you'll see: `path.exists()`, `path.read_text(...)`, `path.write_text(...)`,
`path.mkdir(parents=True, exist_ok=True)` (make a folder, OK if it already exists).

---

## 15. The `if __name__ == "__main__":` line

```python
# at the bottom of main.py
if __name__ == "__main__":
    main()
```

This means "only run `main()` if this file was executed directly (e.g.
`python main.py`), not if it was imported by another file." It's the standard Python way to
mark "the code that runs when you launch this file."

---

## Where to go next

- Open `main.py` — it's small and now heavily commented; reading it top to bottom is a good
  first exercise.
- Then `src/pipeline.py` — the commented walkthrough there matches Workflow 1 in
  `WORKFLOW.md`.
- Keep this primer open in a side tab and look up any syntax that trips you up.
```