# Corpus licences

Compiled by hand from corpus/manifest.json during the licence audit (an earlier
version of this header claimed a generator that does not exist). Documents are
NOT redistributed in this repository - only their provenance is. Run
scripts/fetch_corpus.py to obtain them; each is verified against the SHA-256
recorded in the manifest.

## Excluded at runtime: NoDerivatives and proprietary (7 documents)

The seven documents below are in the manifest and were indexed before the
fetch gate learned to reject their licences. They are now **excluded at index
load** (`agbe/rag/licences.py`, enforced in `VectorIndex.load`): their 1,723
chunks and vector rows are dropped before any query runs, so the shipped
system cannot retrieve from them. The next full re-index removes them from the
artifact permanently.

| Licence | Documents | Why excluded |
|---|---|---|
| CC-BY-ND-4.0 | 3 | chunking/embedding is arguably a derivative work; "arguably" is not a position this submission takes |
| CC-BY-NC-ND-4.0 | 2 | as above |
| Copyrighted; Non-commercial use only | 2 | proprietary terms, no derivative or redistribution rights to rely on |

Measured effect of exclusion: held-out coverage and refusal both unchanged at
100% (31/31, 6/6). Two of the ND documents were also the top noise sources in
the REPORT §6.3 retrieval failure, so the exclusion improved retrieval and
settled the licence question in the same move.

**214 documents** from 1 institutional sources.

## Licence declared in repository metadata

| Licence | Documents |
|---|---|
| unspecified | 81 |
| CC-BY-4.0 | 65 |
| Other | 32 |
| CC-BY-NC-4.0 | 15 |
| CC-BY-NC-SA-3.0 | 5 |
| CC-BY-SA-4.0 | 4 |
| CC-BY-ND-4.0 | 3 |
| CC-BY-NC-ND-4.0 | 2 |
| CC-BY-NC-SA-4.0 | 2 |
| Copyrighted; Non-commercial use only | 2 |
| CC-BY-3.0 | 1 |
| CC-BY | 1 |
| CC-BY-NC-SA 4.0 | 1 |

## Source hosts

| Host | Documents |
|---|---|
| cgspace.cgiar.org | 214 |

## Publishers

| Publisher | Documents |
|---|---|
| CGIAR | 52 |
| International Institute of Tropical Agricult | 36 |
| International Livestock Research Institute | 32 |
| Accelerating Impacts of CGIAR Climate Resear | 15 |
| International Food Policy Research Institute | 14 |
| International Maize and Wheat Improvement Ce | 7 |
| International Water Management Institute | 6 |
| Springer | 5 |
| CGIAR Research Program on Climate Change, Ag | 5 |
| International Potato Center | 4 |
| International Center for Tropical Agricultur | 2 |
| Africa Soil Health Consortium | 2 |

## Bulk sources

## Access-rights gate

Every document above passed a gate requiring repository metadata to state
Open Access, and rejecting any record declaring "all rights reserved". Where
a licence reads "unspecified", the repository recorded Open Access without a
specific licence string; those documents are used under the fetch-do-not-
redistribute model and are not republished here.
