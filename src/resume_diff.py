# src/resume_diff.py — structured "what changed" diff between a user's BASE résumé profile and
# a GENERATED (tailored) résumé. Both sides are the SAME tailoring._SCHEMA dict shape
# (name/contact/summary/competencies/experience[company,title,dates,bullets]/education/
# certifications/skills), so we diff field-by-field and section-by-section rather than diffing
# raw text — reordering (a big part of what tailoring does) reads as "moved", not as noise.
#
# WORD-LEVEL diff: within a matched pair (the summary, a matched-and-reworded bullet, a changed
# skill/education line) we highlight the exact words that were removed (left) and inserted
# (right), GitHub-style. Bullets are FUZZY-PAIRED first: each generated bullet is matched to its
# most-similar base bullet, so a reworded bullet reads as one CHANGED row with inline word
# highlights instead of a removed+added pair. Competencies and certifications stay a plain
# add/remove set diff — they're discrete labels, and fuzzy-pairing them would invent bogus
# "renames" out of genuinely different items.
#
# Each row carries render-ready, already-escaped HTML in left_html/right_html (the template marks
# them |safe). The module stays pure (stdlib difflib + html only, no Flask/DB) so it's easy to test.

import re
import html
from difflib import SequenceMatcher

# Fuzzy-pair a base and generated bullet only when they're at least this char-similar; below it
# they're treated as an independent remove + add. Bullets are whole sentences, so 0.5 reliably
# links a reworded bullet to its origin without linking two unrelated ones.
_BULLET_MATCH_THRESHOLD = 0.5


# ── text helpers ──────────────────────────────────────────────────────────────
def _norm(text) -> str:
    """Whitespace-normalised, case-folded form used ONLY for equality/similarity, never display."""
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _similarity(a, b) -> float:
    """0..1 character-level similarity of two strings (their normalised forms)."""
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _tokens(text: str):
    """Split into words / whitespace runs / punctuation so a word diff aligns on words but can
    still reconstruct the original text verbatim by concatenation."""
    return re.findall(r"\s+|\w+|[^\w\s]", text or "")


def inline_diff(base: str, gen: str):
    """Word-level diff of two strings → (left_html, right_html), each already HTML-escaped and
    render-ready. The left keeps the base text with removed words wrapped in .wd-del; the right
    keeps the generated text with inserted words wrapped in .wd-ins. Shared words are plain."""
    a, b = _tokens(base or ""), _tokens(gen or "")
    sm = SequenceMatcher(None, a, b, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ltext, rtext = html.escape("".join(a[i1:i2])), html.escape("".join(b[j1:j2]))
        if tag == "equal":
            left.append(ltext)
            right.append(rtext)
        elif tag == "delete":
            left.append(f'<span class="wd-del">{ltext}</span>')
        elif tag == "insert":
            right.append(f'<span class="wd-ins">{rtext}</span>')
        else:  # replace
            left.append(f'<span class="wd-del">{ltext}</span>')
            right.append(f'<span class="wd-ins">{rtext}</span>')
    return "".join(left), "".join(right)


def _row(status: str, left="", right="") -> dict:
    """One diff row. left_html/right_html are the render-ready columns: for same/added/removed
    they're the plain (escaped) text; for changed they carry inline word-level highlights."""
    left, right = left or "", right or ""
    if status == "changed":
        left_html, right_html = inline_diff(left, right)
    else:
        left_html, right_html = html.escape(left), html.escape(right)
    return {"status": status, "left": left, "right": right,
            "left_html": left_html, "right_html": right_html}


# ── section differs ───────────────────────────────────────────────────────────
def _diff_text(base, gen) -> list:
    """A single prose block (the summary). One row: same, changed (word-level), added, removed."""
    b, g = (base or "").strip(), (gen or "").strip()
    if not b and not g:
        return []
    if not b:
        return [_row("added", right=g)]
    if not g:
        return [_row("removed", left=b)]
    if _norm(b) == _norm(g):
        return [_row("same", left=b, right=g)]
    return [_row("changed", left=b, right=g)]


def _diff_set(base, gen) -> list:
    """Order-insensitive add/remove diff of plain-string labels (competencies, certifications).
    No fuzzy pairing: discrete labels are either present in both, added, or removed."""
    base = [x for x in (base or []) if (x or "").strip()]
    gen = [x for x in (gen or []) if (x or "").strip()]
    gen_norms = {_norm(x) for x in gen}
    rows, seen = [], set()
    for item in base:
        n = _norm(item)
        seen.add(n)
        rows.append(_row("same", left=item, right=item) if n in gen_norms
                    else _row("removed", left=item))
    for item in gen:
        if _norm(item) not in seen:               # a label the tailored résumé introduced
            rows.append(_row("added", right=item))
    return rows


def _pair_items(base, gen, threshold):
    """Greedily pair base↔generated items by descending similarity (highest first, each item used
    once) above `threshold`. Returns ({base_idx: gen_idx}, base_only_idxs, gen_only_idxs). Exact
    matches (ratio 1.0) win first, so identical items always pair before fuzzy ones."""
    scored = []
    for i, a in enumerate(base):
        for j, b in enumerate(gen):
            r = _similarity(a, b)
            if r >= threshold:
                scored.append((r, i, j))
    scored.sort(key=lambda t: t[0], reverse=True)
    used_b, used_g, match = set(), set(), {}
    for _, i, j in scored:
        if i in used_b or j in used_g:
            continue
        used_b.add(i)
        used_g.add(j)
        match[i] = j
    base_only = [i for i in range(len(base)) if i not in used_b]
    gen_only = [j for j in range(len(gen)) if j not in used_g]
    return match, base_only, gen_only


def _diff_fuzzy(base, gen, threshold=_BULLET_MATCH_THRESHOLD) -> list:
    """Diff two lists of prose strings (résumé bullets) with fuzzy pairing + word-level changes.
    Each base bullet shows in order as same / changed (word diff) / removed; generated bullets
    that matched nothing follow as added."""
    base = [x for x in (base or []) if (x or "").strip()]
    gen = [x for x in (gen or []) if (x or "").strip()]
    match, _base_only, gen_only = _pair_items(base, gen, threshold)
    rows = []
    for i, item in enumerate(base):
        if i in match:
            g = gen[match[i]]
            rows.append(_row("same", left=item, right=g) if _norm(item) == _norm(g)
                        else _row("changed", left=item, right=g))
        else:
            rows.append(_row("removed", left=item))
    for j in gen_only:
        rows.append(_row("added", right=gen[j]))
    return rows


def _diff_keyed(base, gen, key, render) -> list:
    """Diff two lists of dicts matched by a KEY field (skills by category, education by
    institution). Matched pair → same or changed (word-level) on the RENDERED value; unmatched →
    removed/added. Preserves base order, then appends generated-only entries."""
    def index(items):
        out = {}
        for it in items or []:
            k = _norm(it.get(key))
            if k and k not in out:                # first-seen wins on duplicate keys
                out[k] = it
        return out

    b_idx, g_idx = index(base), index(gen)
    rows = []
    for it in base or []:
        k = _norm(it.get(key))
        if not k:
            continue
        left = render(it)
        if k in g_idx:
            right = render(g_idx[k])
            rows.append(_row("same" if _norm(left) == _norm(right) else "changed",
                             left=left, right=right))
        else:
            rows.append(_row("removed", left=left))
    for it in gen or []:
        k = _norm(it.get(key))
        if k and k not in b_idx:
            rows.append(_row("added", right=render(it)))
    return rows


def _render_skill(s) -> str:
    cat, items = (s.get("category") or "").strip(), (s.get("items") or "").strip()
    return f"{cat}: {items}" if cat else items


def _render_education(e) -> str:
    parts = [e.get("degree"), e.get("institution"), e.get("date"), e.get("detail")]
    return " — ".join(p.strip() for p in parts if (p or "").strip())


def _role_header(role) -> str:
    title, company = (role.get("title") or "").strip(), (role.get("company") or "").strip()
    dates = (role.get("dates") or "").strip()
    head = " · ".join(p for p in (title, company) if p)
    return f"{head}  ({dates})" if dates else head


def _diff_experience(base, gen) -> list:
    """Diff experience role-by-role. Roles are matched by (company, title); within a matched
    role, bullets are fuzzy-paired and word-diffed. Returns a list of role blocks, each with a
    status, a left/right header, and its bullet rows. Unmatched roles are whole removed/added
    blocks."""
    def key(role):
        return (_norm(role.get("company")), _norm(role.get("title")))

    g_by_key, g_used = {}, set()
    for i, role in enumerate(gen or []):
        g_by_key.setdefault(key(role), []).append(i)

    blocks = []
    for role in base or []:
        k = key(role)
        matches = [i for i in g_by_key.get(k, []) if i not in g_used]
        if matches:
            gi = matches[0]
            g_used.add(gi)
            g_role = gen[gi]
            bullet_rows = _diff_fuzzy(role.get("bullets"), g_role.get("bullets"))
            status = "same" if all(r["status"] == "same" for r in bullet_rows) else "changed"
            blocks.append({
                "status": status,
                "left_header": _role_header(role),
                "right_header": _role_header(g_role),
                "rows": bullet_rows,
            })
        else:
            blocks.append({
                "status": "removed",
                "left_header": _role_header(role),
                "right_header": "",
                "rows": _diff_fuzzy(role.get("bullets"), []),
            })
    for i, role in enumerate(gen or []):
        if i not in g_used:                       # a role the tailored résumé surfaced that base didn't match
            blocks.append({
                "status": "added",
                "left_header": "",
                "right_header": _role_header(role),
                "rows": _diff_fuzzy([], role.get("bullets")),
            })
    return blocks


# ── top-level ─────────────────────────────────────────────────────────────────
def diff_resumes(base: dict, gen: dict) -> dict:
    """Compare a base résumé profile against a generated résumé (both tailoring._SCHEMA dicts).
    Returns a render-ready structure: an ordered list of sections plus change counts. `experience`
    is a special section kind (role blocks); every other section is a flat list of diff rows."""
    base, gen = base or {}, gen or {}
    sections = [
        {"kind": "block", "title": "Professional Summary",
         "rows": _diff_text(base.get("summary"), gen.get("summary"))},
        {"kind": "block", "title": "Core Competencies",
         "rows": _diff_set(base.get("competencies"), gen.get("competencies"))},
        {"kind": "experience", "title": "Experience",
         "roles": _diff_experience(base.get("experience"), gen.get("experience"))},
        {"kind": "block", "title": "Skills",
         "rows": _diff_keyed(base.get("skills"), gen.get("skills"), "category", _render_skill)},
        {"kind": "block", "title": "Education",
         "rows": _diff_keyed(base.get("education"), gen.get("education"), "institution", _render_education)},
        {"kind": "block", "title": "Certifications",
         "rows": _diff_set(base.get("certifications"), gen.get("certifications"))},
    ]

    counts = {"same": 0, "changed": 0, "added": 0, "removed": 0}
    def tally(rows):
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
    for sec in sections:
        if sec["kind"] == "experience":
            for block in sec["roles"]:
                tally(block["rows"])
        else:
            tally(sec["rows"])

    return {"sections": sections, "counts": counts}
