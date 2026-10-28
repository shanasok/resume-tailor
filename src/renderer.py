# src/renderer.py — turns the tailored résumé DATA (a Python dict / JSON) into real files:
# first an HTML file (using the visual template templates/resume.html), then a PDF printed
# from that HTML by a headless browser. Also counts the PDF's pages for the length check.

import asyncio                                       # runs the async PDF code (see PRIMER §13)
import json                                          # read/write JSON (the cached résumé data)
from pathlib import Path                             # OS-independent file paths
from jinja2 import Environment, FileSystemLoader     # Jinja: fills {{ placeholders }} in templates

# Where the HTML templates live, and where generated résumés are written. Both are built
# relative to this file so they work no matter where you run the app from.
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
OUTPUT_DIR = Path(__file__).parent.parent / "resumes" / "generated_resumes"


def _slug(company: str, role: str) -> str:
    """Build a safe filename piece from the company + role, e.g. "Acme Corp" + "Sr. Eng" →
    "acme_corp_sr._eng". (A "slug" is a simplified, filename-safe version of text.)"""
    # A small nested helper that lowercases and replaces characters illegal/awkward in filenames.
    def clean(s):
        return s.lower().replace(" ", "_").replace("/", "_").replace(",", "")
    return f"{clean(company)}_{clean(role)}"


def json_path(company: str, role: str, out_dir: Path = None) -> Path:
    """The path to the cached JSON data file for this company+role résumé. `out_dir`
    overrides the default output folder (e.g. a dated sub-folder for scheduled runs)."""
    return (out_dir or OUTPUT_DIR) / f"{_slug(company, role)}_resume.json"


def load_cached_data(company: str, role: str, out_dir: Path = None) -> dict:
    """Load previously-tailored resume data so the HTML/PDF can be re-rendered
    after a template tweak without another (quota-consuming) model call. `out_dir`
    is the folder the résumé lives in (default OUTPUT_DIR)."""
    path = json_path(company, role, out_dir)
    if not path.exists():
        # raise = signal an error to the caller. FileNotFoundError is a built-in error type;
        # the message tells you how to fix it.
        raise FileNotFoundError(
            f"No cached tailored data at {path}. Run without --render-only first."
        )
    # Read the file's text and parse it from JSON into a Python dict.
    return json.loads(path.read_text(encoding="utf-8"))


def render_html(resume_data: dict, company: str, role: str, out_dir: Path = None) -> Path:
    """Fill the résumé template with `resume_data` and write the HTML file. Also caches the
    data as JSON next to it (so it can be re-rendered later for free). Returns the HTML path.

    `out_dir` chooses where the HTML + JSON are written (default OUTPUT_DIR). Scheduled
    (cron) runs pass a dated sub-folder so each day's batch is grouped together; the folder
    is created if it doesn't exist."""
    out_dir = out_dir or OUTPUT_DIR
    # Set up Jinja to load templates from our templates/ folder, then grab resume.html.
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("resume.html")
    html = template.render(resume=resume_data)       # produce the final HTML string

    out_dir.mkdir(parents=True, exist_ok=True)       # create the (dated) folder if needed
    # Cache the tailored data alongside the output for fast re-rendering (the --render-only path).
    # json.dumps(..., indent=2) turns the dict into nicely-indented JSON text.
    json_path(company, role, out_dir).write_text(
        json.dumps(resume_data, indent=2), encoding="utf-8"
    )

    # Write the HTML file and return its path.
    output_path = out_dir / f"{_slug(company, role)}_resume.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


async def _html_to_pdf(html_path: Path, pdf_path: Path):
    """Open the HTML file in a headless (invisible) Chromium browser and print it to PDF.
    This is `async` because driving a browser involves a lot of waiting (see PRIMER §13)."""
    # Imported here (inside the function) rather than at the top so the app still starts even
    # if Playwright isn't installed until it's actually needed.
    from playwright.async_api import async_playwright
    async with async_playwright() as p:              # start Playwright (auto-cleans up after)
        # --no-sandbox: Chromium's user-namespace sandbox is blocked inside the Docker
        # container (default seccomp, non-root user). Safe here — the only page ever
        # rendered is our own template's output, never remote content.
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page()              # open a blank tab
        # Load our local HTML file; wait until the page has finished loading ("networkidle").
        await page.goto(f"file://{html_path.absolute()}", wait_until="networkidle")
        # Print the page to PDF. NOTE: this `margin` controls the PDF's page margins — see the
        # "PDF Layout Gotcha" in CLAUDE.md before changing it (editing the HTML padding won't work).
        await page.pdf(
            path=str(pdf_path),
            format="Letter",
            margin={"top": "0.35in", "bottom": "0.35in", "left": "0.4in", "right": "0.4in"},
            print_background=True                    # include background colors/shading
        )
        await browser.close()                        # shut the browser down


def render_pdf(html_path: Path) -> Path:
    """Make a PDF next to the given HTML file and return the PDF's path. This is the normal
    (non-async) function the rest of the app calls; it runs the async helper above."""
    pdf_path = html_path.with_suffix(".pdf")         # same name, ".pdf" instead of ".html"
    asyncio.run(_html_to_pdf(html_path, pdf_path))   # kick off the async work and wait for it
    return pdf_path


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in the generated PDF (used by the length feedback loop)."""
    from pypdf import PdfReader                       # a library that reads PDF files
    return len(PdfReader(str(pdf_path)).pages)        # open the PDF and count its pages
