"""The one licence predicate, shared by the corpus gate and the index.

WHY DOCUMENTS THAT FAILED THE GATE WERE STILL BEING RETRIEVED
-------------------------------------------------------------
The fetch gate in scripts/fetch_corpus.py learned to reject NoDerivatives
licences partway through the project - "going forward". Five documents
harvested before that (three CC-BY-ND-4.0, two CC-BY-NC-ND-4.0) were already
chunked, embedded and indexed, and a full re-index costs hours, so they stayed.

That was the wrong resting place, for two reasons that turned out to be the
same reason:

  * Whether chunking and embedding creates a derivative work is arguable, and
    "arguable" is a bad posture for a competition submission when the cost of
    certainty is five documents out of a thousand.

  * Two of the five are the "Adapting green innovation centers to climate
    change" reports - the exact documents whose chunks polluted retrieval for
    the submitted cassava test prompt (REPORT §6.3) and supplied the "delayed
    harvesting" misdiagnosis. The licence gate and retrieval quality were
    pointing at the same documents.

Every chunk carries its licence (see chunker.py: provenance travels with the
text), so the index can enforce the rule at load time on whatever it is given:
excluded chunks are dropped, with their vector rows, before any query runs.
The next full re-index removes them permanently; until then the shipped system
cannot retrieve from them.

WHY ONE MODULE
--------------
The gate and the index previously could not share the rule because it lived as
an inline regex in a script. Two copies of a policy drift - this project has
measured that four times now with code paths - so the predicate lives here and
both callers import it.
"""

from __future__ import annotations

import re

#: NoDerivatives in its observed spellings: "CC-BY-ND-4.0", "CC BY-NC-ND",
#: "noderivs", "No Derivatives". The bare \bnd\b arm catches licence strings
#: that abbreviate without a version suffix.
_ND = re.compile(r"\bnd\b|\bnd-\d|noderiv|no.?deriv", re.IGNORECASE)

#: All-rights-reserved material must never be indexed at all.
_PROPRIETARY = re.compile(r"copyrighted|all rights reserved", re.IGNORECASE)


def excluded(licence: str) -> tuple[bool, str]:
    """Whether material under `licence` may be chunked, embedded and retrieved.

    Returns (excluded, reason). Unknown or empty licences are NOT excluded
    here: the fetch gate rejects them earlier with better context, and the
    index's job is only to enforce the specific exclusions above on material
    that was already admitted.
    """
    if _ND.search(licence):
        return True, f"NoDerivatives ({licence})"
    if _PROPRIETARY.search(licence):
        return True, f"proprietary ({licence})"
    return False, ""
