# src/base_resume.py — the base résumé the tailoring pipeline draws from, PER USER.
#
# Historically the base résumé was a single global .docx flattened to text. Now each user
# has a structured profile in the DB (the resume_profiles + section tables, entered/edited
# on /base-resume or imported from a .docx). This module is the sole source of truth for
# generation, rendering the DB profile to the same plain-text form the pipeline has always
# tailored from.

from src import resume_parser


def profile_to_text(profile: dict) -> str:
    """Render a structured base-résumé profile (tailoring._SCHEMA shape) into clean plain
    text — the form the tailoring prompt and the critic expect. Empty sections are omitted."""
    p = profile or {}
    out = []

    name = (p.get("name") or "").strip()
    if name:
        out.append(name)
    contact = p.get("contact") or {}
    contact_line = " | ".join(
        v for v in ((contact.get(k) or "").strip()
                    for k in ("location", "phone", "email", "website")) if v)
    if contact_line:
        out.append(contact_line)

    summary = (p.get("summary") or "").strip()
    if summary:
        out += ["", "SUMMARY", summary]

    comps = [c.strip() for c in (p.get("competencies") or []) if c and c.strip()]
    if comps:
        out += ["", "CORE COMPETENCIES", ", ".join(comps)]

    experience = p.get("experience") or []
    if experience:
        out.append("")
        out.append("EXPERIENCE")
        for e in experience:
            title, company = (e.get("title") or "").strip(), (e.get("company") or "").strip()
            head = f"{title} — {company}" if title and company else (title or company)
            loc = (e.get("location") or "").strip()
            if loc:
                head = f"{head} ({loc})" if head else f"({loc})"
            dates = (e.get("dates") or "").strip()
            if dates:
                head = f"{head} | {dates}" if head else dates
            if head:
                out += ["", head]
            for b in (e.get("bullets") or []):
                if b and b.strip():
                    out.append(f"- {b.strip()}")

    education = p.get("education") or []
    if education:
        out.append("")
        out.append("EDUCATION")
        for ed in education:
            degree, inst = (ed.get("degree") or "").strip(), (ed.get("institution") or "").strip()
            line = f"{degree}, {inst}" if degree and inst else (degree or inst)
            date = (ed.get("date") or "").strip()
            if date:
                line = f"{line} ({date})" if line else f"({date})"
            detail = (ed.get("detail") or "").strip()
            if detail:
                line = f"{line} — {detail}" if line else detail
            if line:
                out.append(line)

    skills = [s for s in (p.get("skills") or [])
              if (s.get("category") or "").strip() or (s.get("items") or "").strip()]
    if skills:
        out.append("")
        out.append("SKILLS")
        for s in skills:
            cat, items = (s.get("category") or "").strip(), (s.get("items") or "").strip()
            out.append(f"{cat}: {items}" if cat and items else (cat or items))

    certs = [c.strip() for c in (p.get("certifications") or []) if c and c.strip()]
    if certs:
        out.append("")
        out.append("CERTIFICATIONS")
        out += [f"- {c}" for c in certs]

    return "\n".join(out).strip()


def profile_has_content(profile) -> bool:
    """True when a profile has any real résumé content (so it's worth tailoring from). A bare
    profile row with everything empty is treated as "no résumé" → the UI shows an entry banner."""
    if not isinstance(profile, dict):
        return False
    if (profile.get("name") or "").strip() or (profile.get("summary") or "").strip():
        return True
    if any(profile.get(k) for k in ("experience", "education", "skills")):
        return True
    return any((x or "").strip()
               for key in ("competencies", "certifications")
               for x in (profile.get(key) or []))


def has_base_resume(user_id: int) -> bool:
    """True when this user has a usable structured base-résumé profile in the DB. Used by
    the generation guards in app.py and the scan warm-up."""
    from src import db                                    # local import avoids an import cycle
    with db.connect() as conn:
        profile = db.fetch_resume_profile(conn, user_id)
    return profile_has_content(profile)


def structured_profile(user_id: int):
    """The user's structured base-résumé profile (tailoring._SCHEMA shape) when it has real
    content — the source of truth the tailoring step's fact-locking overwrites the draft from
    (skills, education, certifications, contact, each role's company/title/dates). Returns None
    for a user with no profile content entered yet: there's nothing structured to lock against,
    so tailoring leaves the draft as the model produced it."""
    from src import db                                    # local import avoids an import cycle
    with db.connect() as conn:
        profile = db.fetch_resume_profile(conn, user_id)
    return profile if profile_has_content(profile) else None


def resume_source_text(user_id: int) -> str:
    """The base-résumé TEXT the pipeline tailors from, for this user, rendered from their
    structured DB profile (the single source of truth). Chat-added experience is appended.
    Raises FileNotFoundError (pointing to the Base Résumé screen) when the user has no
    profile yet."""
    from src import db                                    # local import avoids an import cycle
    with db.connect() as conn:
        profile = db.fetch_resume_profile(conn, user_id)
    if profile_has_content(profile):
        return resume_parser.append_additions(profile_to_text(profile))
    raise FileNotFoundError(
        "No base résumé for this user — enter or import one on the Base Résumé screen "
        "(/base-resume).")
