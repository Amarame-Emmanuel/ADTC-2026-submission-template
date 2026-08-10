"""Yoruba <-> English translation bridge (NLLB-200 distilled, int8, CTranslate2).

WHY A BRIDGE INSTEAD OF A MULTILINGUAL MODEL
--------------------------------------------
The obvious alternative is to ask the language model to answer in Yoruba
directly. It does not work well. 3B-class instruction models have seen very
little Yoruba, and no amount of prompting fixes a training-data gap - the
model produces confident, fluent, wrong Yoruba, which is worse than none.

Moving up to a 7-8B multilingual model would cost roughly 4.7 GB of weights
and halve generation speed on a four-core CPU, and would still be beaten at
Yoruba by a specialist that actually trained on `yor_Latn`.

So: translate in, reason in English, translate out. ~600 MB at int8, loaded
only when a non-English request arrives.

WHAT THIS COSTS, STATED PLAINLY
-------------------------------
Round-tripping loses nuance, and agricultural terminology is exactly where a
general-domain translation model is weakest - "cassava mosaic disease" and
"stem cuttings" are not common phrases in NLLB's training data. Some of that
is mitigated by a glossary applied after translation, but not all of it.

This is why the native-speaker evaluation set exists, and why the Yoruba claim
in REPORT.md is scoped to what has actually been checked by a speaker rather
than to everything NLLB will happily emit.

RETRIEVAL HAPPENS IN ENGLISH
----------------------------
A Yoruba question is translated before it reaches the embedder. That is what
lets the system use a small English-only embedding model instead of a larger
multilingual one - a deliberate trade of translation latency for memory.
"""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path

from agbe import config

YORUBA = "yor_Latn"
ENGLISH = "eng_Latn"

#: Terms NLLB reliably mangles in the agricultural domain, mapped to the
#: wording Yoruba-speaking extension materials actually use.
#:
#: DELIBERATELY EMPTY UNTIL VALIDATED. Populating this from a dictionary
#: without a Yoruba speaker checking each entry would encode confident errors
#: into every answer and be harder to detect than raw translation noise,
#: because it would look deliberate. Entries are added only once the native
#: speaker has confirmed them against the evaluation set.
GLOSSARY: dict[str, str] = {}


@dataclass
class Translation:
    text: str
    source_lang: str
    target_lang: str
    n_segments: int = 1


def looks_yoruba(text: str) -> bool:
    """Cheap heuristic for routing input to the translator.

    Yoruba orthography uses under-dot vowels and tone marks that do not occur
    in English. Combined with common function words, that is enough to route
    a query without loading a language-ID model.

    Deliberately biased towards answering "no": an English question sent
    through the translator wastes 600 MB and a few seconds, whereas a Yoruba
    question treated as English retrieves nothing and produces a refusal for
    a question the corpus could have answered.
    """
    if re.search(r"[ẹọṣẸỌṢ]", text):
        return True
    markers = {
        "ni", "mi", "mo", "ti", "won", "wọn", "ba", "bá", "ki", "kí",
        "se", "ṣe", "ohun", "kini", "kíni", "bawo", "báwo", "oko", "oko",
        "ewe", "ewé", "gbin", "ilẹ", "ile", "ọgbin", "eweko",
    }
    words = set(re.findall(r"\w+", text.lower()))
    return len(words & markers) >= 2


class Translator:
    """NLLB-200 distilled 600M via CTranslate2, int8."""

    def __init__(self, model_dir: Path | None = None, n_threads: int | None = None):
        self.model_dir = Path(model_dir or config.TRANSLATION.path)
        self.n_threads = n_threads or config.TRANSLATION.n_threads
        self._translator = None
        self._tokenizer = None

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if self._translator is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"translation model missing: {self.model_dir}\n"
                "run `make convert-models` (needs the converter image; torch is "
                "not in the runtime image)"
            )

        import ctranslate2
        from transformers import AutoTokenizer

        self._translator = ctranslate2.Translator(
            str(self.model_dir),
            device="cpu",
            compute_type=config.TRANSLATION.quantization,
            inter_threads=1,
            intra_threads=self.n_threads,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

    def close(self) -> None:
        """Release ~600 MB.

        Called when a session goes idle. An English-only conversation should
        not hold the translator resident for its whole life.
        """
        self._translator = None
        self._tokenizer = None
        gc.collect()

    def __enter__(self) -> "Translator":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- translation -------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split into sentences before translating.

        NLLB was trained on sentence pairs and degrades on long multi-sentence
        inputs, truncating or dropping clauses. Answers here are several
        sentences long, so they are translated one sentence at a time.
        """
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p.strip()]

    def _translate_segments(
        self, segments: list[str], src: str, tgt: str
    ) -> list[str]:
        self.load()
        assert self._translator is not None and self._tokenizer is not None

        self._tokenizer.src_lang = src
        encoded = [
            self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(seg))
            for seg in segments
        ]

        results = self._translator.translate_batch(
            encoded,
            target_prefix=[[tgt]] * len(encoded),
            beam_size=2,          # 4 is standard; 2 halves latency for a
                                  # quality difference that does not survive
                                  # the round trip on a CPU-bound device
            max_batch_size=8,
        )

        out: list[str] = []
        for result in results:
            # hypotheses[0][0] is the target language token we supplied as the
            # prefix; dropping it prevents "eng_Latn" appearing in the output.
            tokens = result.hypotheses[0][1:]
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            out.append(self._tokenizer.decode(ids, skip_special_tokens=True))
        return out

    def to_english(self, text: str) -> Translation:
        segments = self._split_sentences(text)
        translated = self._translate_segments(segments, YORUBA, ENGLISH)
        return Translation(" ".join(translated), YORUBA, ENGLISH, len(segments))

    def to_yoruba(self, text: str) -> Translation:
        segments = self._split_sentences(text)
        translated = self._translate_segments(segments, ENGLISH, YORUBA)
        out = " ".join(translated)
        for english_term, yoruba_term in GLOSSARY.items():
            out = re.sub(
                r"\b" + re.escape(english_term) + r"\b", yoruba_term, out,
                flags=re.IGNORECASE,
            )
        return Translation(out, ENGLISH, YORUBA, len(segments))
