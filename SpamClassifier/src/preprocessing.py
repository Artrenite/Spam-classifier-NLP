from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.corpus import stopwords
except Exception:
    stopwords = None

try:
    from nltk.stem import WordNetLemmatizer
except Exception:
    WordNetLemmatizer = None

from nltk.stem import PorterStemmer


_clean_re = re.compile(r"[^a-zA-Z\s]")
_space_re = re.compile(r"\s+")


def _get_stopwords(extra_stopwords: Iterable[str] | None = None) -> set[str]:
    sw = set(ENGLISH_STOP_WORDS)
    if stopwords is not None:
        try:
            sw |= set(stopwords.words("english"))
        except Exception:
            pass
    if extra_stopwords:
        sw |= set(extra_stopwords)
    return sw


def _tokenize(text: str) -> List[str]:
    return nltk.tokenize.wordpunct_tokenize(text)


def _safe_lemmatizer():
    """
    Return WordNetLemmatizer instance only if WordNet resource is available.
    If not available -> return None (we will fallback to stemming).
    """
    if WordNetLemmatizer is None:
        return None
    try:
        # Will raise LookupError if wordnet is not downloaded
        nltk.data.find("corpora/wordnet")
    except Exception:
        return None
    try:
        return WordNetLemmatizer()
    except Exception:
        return None


def preprocess_text(text: str, extra_stopwords: Iterable[str] | None = None) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = _clean_re.sub(" ", text)
    text = _space_re.sub(" ", text).strip()
    if not text:
        return ""

    tokens = _tokenize(text)
    sw = _get_stopwords(extra_stopwords)

    lemmatizer = _safe_lemmatizer()
    stemmer = PorterStemmer()

    out: List[str] = []
    for t in tokens:
        if len(t) <= 1:
            continue
        if t in sw:
            continue

        if lemmatizer is not None:
            # lemmatization (only when wordnet exists)
            out.append(lemmatizer.lemmatize(t))
        else:
            # fallback: stemming (always works)
            out.append(stemmer.stem(t))

    return " ".join(out)


def preprocess_corpus(texts: Iterable[str]) -> np.ndarray:
    return np.array([preprocess_text(t) for t in texts], dtype=object)
