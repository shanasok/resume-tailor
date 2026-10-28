"""Save candidate-provided experience captured via the Chat feature.

When Chat captures a real accomplishment, we append it to the additions file
(additional_experience.txt). base_resume.resume_source_text() merges that file into the
base résumé for every future generation, so nothing is ever lost.
"""
from src.resume_parser import append_addition        # the additions-file writer


def add_experience(bullet: str, company: str = "") -> str:
    """Add a real experience bullet (captured via chat) to the candidate's additions list,
    which is merged into the base résumé at generation time. Returns a short human
    description of where it landed. `company`, if given, only refines the note."""
    bullet = bullet.strip().lstrip("-• ").strip()     # clean the text
    if not bullet:                                    # nothing to add
        return ""
    append_addition(bullet)
    return ("your additional-experience list" if not company else
            f"your additional-experience list (added under '{company}')")
