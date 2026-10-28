#!/usr/bin/env python3
# ^ This first line (a "shebang") lets you run the file directly as ./main.py on
#   Mac/Linux. It tells the OS to execute it with python3.
"""
Resume Tailor — Phase 1 (command-line version)

Tailors the base résumé to a specific job posting and outputs HTML + PDF.

Usage:
  python main.py --company "Acme Corp" --role "Director of Engineering" --job-file posting.txt
  python main.py --company "Acme Corp" --role "Director of Engineering" --job-url https://...
  python main.py --company "Acme Corp" --role "Director of Engineering" --job-text "We are looking for..."

This file is just a thin "wrapper": it reads the options you typed on the command line,
then hands everything to generate() in src/pipeline.py, which does the real work. (The web
app, app.py, calls the same generate() — so the two share one brain.)
"""
# ── Imports: pull in code we need from Python's standard library and this project ──
import argparse                      # Python's built-in tool for parsing command-line options
from dotenv import load_dotenv       # reads a .env file into environment variables (API keys, etc.)

# Load the .env file NOW, before importing project modules — some of them read API keys /
# settings from the environment at import time, so the values must already be in place.
load_dotenv()

# These imports come AFTER load_dotenv() on purpose (see the comment above).
from src.renderer import render_html, render_pdf, load_cached_data
from src.pipeline import (generate, DEFAULT_MAX_PAGES, DEFAULT_MIN_PAGES,
                          DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_POLISH, DEFAULT_MIN_SCORE)


def main():
    """The entry point: runs when you launch this file. Reads the command-line options,
    decides what to do, and prints where the output landed."""

    # ── 1. Describe the command-line options this script accepts ──────────────────
    # argparse builds a parser that knows about each "--option", checks them, and even
    # generates a --help message automatically.
    parser = argparse.ArgumentParser(
        description="Tailor your resume for a specific job posting (ATS-optimized)."
    )
    # required=True means the script refuses to run without these two.
    parser.add_argument("--company", required=True, help="Company name (used for output filename)")
    parser.add_argument("--role", required=True, help="Job title / role")
    # action="store_true" makes this a yes/no flag: present = True, absent = False.
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip tailoring; re-render HTML+PDF from the cached JSON (use after template tweaks).",
    )
    # type=int converts the typed text into a number. default=… is used if you omit the option.
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                        help=f"Target maximum pages (default {DEFAULT_MAX_PAGES}).")
    parser.add_argument("--min-pages", type=int, default=DEFAULT_MIN_PAGES,
                        help=f"Minimum pages — a shorter result means real content was "
                             f"dropped and triggers a retry (default {DEFAULT_MIN_PAGES}).")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help=f"Max tailoring retries to hit the targets (default {DEFAULT_MAX_ATTEMPTS}).")
    parser.add_argument("--max-polish", type=int, default=DEFAULT_MAX_POLISH,
                        help=f"After the retry loop, surgically fix any residual truthfulness / "
                             f"AI-style flags up to this many times (default {DEFAULT_MAX_POLISH}; "
                             f"0 disables).")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                        help=f"Minimum critic quality score 0-100 to accept (default {DEFAULT_MIN_SCORE}).")
    parser.add_argument("--no-external-critic", action="store_true",
                        help="Skip the external review pass entirely; use internal checks only.")
    parser.add_argument("--no-evidence", action="store_true",
                        help="Skip the evidence pre-pass (map posting needs → real experience) before tailoring.")
    # Per-stage LLM backend selection (mirrors the web UI's four pickers). Each of the four
    # AI steps can run on a different backend; all default to Claude except the external
    # review, which defaults to ChatGPT to preserve the historical cross-check.
    # Backends are validated at dispatch (src.tailoring.local_tag), not by a fixed list —
    # local models are discovered at runtime, so any "local:<tag>" is accepted.
    _BACKENDS_HELP = "claude, chatgpt, or local:<tag> (e.g. local:qwen3.6:27b)"
    parser.add_argument("--gen-llm", default="claude",
                        help="LLM for stage 1 (generate draft): " + _BACKENDS_HELP)
    parser.add_argument("--external-llm", default="chatgpt",
                        help="LLM for stage 2 (external review): " + _BACKENDS_HELP + ", or off")
    parser.add_argument("--check-llm", default="claude",
                        help="LLM for stage 3 (truthfulness check): " + _BACKENDS_HELP)
    parser.add_argument("--regen-llm", default="claude",
                        help="LLM for stage 4 (regenerate w/ feedback): " + _BACKENDS_HELP)

    # The job posting can come from exactly ONE of three places. A "mutually exclusive group"
    # tells argparse: allow at most one of these at a time.
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--job-file", metavar="PATH", help="Path to a .txt file containing the job posting")
    source.add_argument("--job-url", metavar="URL", help="URL of the job posting page")
    source.add_argument("--job-text", metavar="TEXT", help="Job posting text passed directly as a string")

    # Actually read what the user typed. args.company, args.max_pages, etc. now hold the values.
    args = parser.parse_args()

    # ── 2. Do the work ────────────────────────────────────────────────────────────
    if args.render_only:
        # FAST PATH: don't call the AI at all. Re-build the HTML + PDF from the previously
        # saved JSON data. Handy after you tweak the visual template (templates/resume.html).
        print("  Loading cached tailored data, rendering...")
        tailored = load_cached_data(args.company, args.role)        # load the saved résumé data
        html_path = render_html(tailored, args.company, args.role)  # data → HTML file
        pdf_path = render_pdf(html_path)                            # HTML file → PDF file
    else:
        # NORMAL PATH: figure out where the job posting is. Exactly one of these is set;
        # the "or" chain picks the first one that isn't empty/None.
        job_source = args.job_file or args.job_url or args.job_text
        if not job_source:
            # parser.error() prints the message + usage and exits — none of the three was given.
            parser.error("one of --job-file, --job-url, or --job-text is required (unless --render-only)")

        # Hand everything to the shared pipeline. It returns three things:
        #   html_path, pdf_path — where the files were written
        #   meta                — a dict of stats (pages, score, attempts, issues found…)
        # on_progress=… is a small function the pipeline calls to report progress; here we
        # just print each message (the "lambda" is a one-line throwaway function).
        html_path, pdf_path, meta = generate(
            args.company, args.role, job_source,
            max_pages=args.max_pages, min_pages=args.min_pages,
            max_attempts=args.max_attempts,
            max_polish=args.max_polish,
            min_score=args.min_score,
            use_external_critic=not args.no_external_critic,
            use_evidence=not args.no_evidence,
            stage_llms={"generate": args.gen_llm, "external": args.external_llm,
                        "check": args.check_llm, "regenerate": args.regen_llm},
            on_progress=lambda m: print(f"  {m}"),
        )

    # ── 3. Report where the output went ───────────────────────────────────────────
    print("\nDone!")
    print(f"  HTML → {html_path}")
    print(f"  PDF  → {pdf_path}")
    # In render-only mode there's no "meta" (we never ran the AI), so only show stats otherwise.
    if not args.render_only:
        status = "✓ accepted" if meta["accepted"] else "⚠ best effort"
        chatgpt = "ChatGPT review applied" if meta.get("external_review_used") else "internal checks only"
        print(f"  {meta['pages']} page(s), score {meta['score']}/100 "
              f"({status}, pass-2 ×{meta['attempts']}, {chatgpt})")
        # The critic returns a list of any truthfulness problems it couldn't fully resolve;
        # print each so you can fix them by hand if needed.
        for issue in meta["fidelity_issues"]:
            print(f"    ⚠ fidelity: {issue}")


# Only run main() when this file is launched directly (python main.py), NOT when it's
# imported by another file. See PYTHON_PRIMER.md section 15 for what this line means.
if __name__ == "__main__":
    main()
