import re
import xml.etree.ElementTree as ET

import httpx

ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}
FETCH_TIMEOUT = 20
MIN_QUOTE_WORDS = 4

# [arXiv:2312.00752] next to "an exact quote from the abstract"
_CITATION = re.compile(r'\[arXiv:(?P<id>[^\s\]"]+)[^\]]*\]')
_QUOTE = re.compile(r'"([^"]+)"')
_VERSION = re.compile(r"v\d+$")


def _norm_id(raw: str) -> str:
    return _VERSION.sub("", raw.strip().rsplit("/abs/", 1)[-1])


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def _unslash(text: str) -> str:
    """Curly quotes to straight, and drop the backslash escapes models add before percent signs."""
    return text.translate({0x201C: 34, 0x201D: 34}).replace("\\", "")


def _fragments(quote: str) -> list[str]:
    """Pieces of a quote split at ellipses, edge punctuation trimmed. Pieces under
    MIN_QUOTE_WORDS are dropped: scare quotes and short terms are not evidence."""
    pieces = (f.strip(" .,;:!?") for f in re.split(r"\.\.\.|…", quote))
    return [f for f in pieces if len(f.split()) >= MIN_QUOTE_WORDS]


def parse_feed(xml_text: str) -> list[dict]:
    """arXiv Atom feed -> [{id, title, authors, published, abstract}]."""
    papers = []
    for entry in ET.fromstring(xml_text).findall("a:entry", _NS):
        def text(tag: str) -> str:
            node = entry.find(f"a:{tag}", _NS)
            return " ".join((node.text or "").split()) if node is not None else ""

        papers.append({
            "id": _norm_id(text("id")),
            "title": text("title"),
            "authors": [" ".join((n.text or "").split())
                        for n in entry.findall("a:author/a:name", _NS)],
            "published": text("published")[:10],
            "abstract": text("summary"),
        })
    return papers


def fetch_papers(query: str, max_papers: int) -> list[dict]:
    """Newest arXiv papers matching `query`. Raises if the fetch fails or finds nothing."""
    params = {"search_query": query, "max_results": max_papers,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    resp = httpx.get(ARXIV_API, params=params, timeout=FETCH_TIMEOUT, follow_redirects=True)
    resp.raise_for_status()
    papers = parse_feed(resp.text)
    if not papers:
        raise RuntimeError(f"arXiv returned no papers for query {query!r}")
    return papers


def format_papers(papers: list[dict]) -> str:
    return "\n\n".join(
        f"[arXiv:{p['id']}] {p['title']} ({p['published']}; {', '.join(p['authors'][:3])})\n"
        f"Abstract: {p['abstract']}"
        for p in papers
    )


def check_citations(answer: str, papers: list[dict]) -> tuple[bool, str]:
    """Deterministic gate. Per paragraph: cited IDs must be fetched papers, and every
    quoted span of 4+ words must appear in the abstract of a paper cited there. Every
    cited paper needs at least one such verified quote somewhere in the answer."""
    by_id = {p["id"]: p for p in papers}
    problems, backed, cited = [], set(), set()

    for para in re.split(r"\n\s*\n", _unslash(answer or "")):
        ids = {_norm_id(m["id"]) for m in _CITATION.finditer(para)}
        cited |= ids
        unknown = sorted(ids - by_id.keys())
        problems += [f"{i}: not one of the fetched papers ({', '.join(by_id)})" for i in unknown]
        evidence = {i: _squash(by_id[i]["abstract"] + " " + by_id[i]["title"])
                    for i in ids if i in by_id}
        for q in _QUOTE.findall(para):
            for frag in _fragments(q):
                hit = [i for i, text in evidence.items() if _squash(frag) in text]
                if hit:
                    backed.update(hit)
                else:
                    problems.append(f'quote not found in any paper cited next to it: "{frag}"')

    if not cited:
        return False, 'no citations found; cite claims as [arXiv:ID] next to a quoted "exact phrase" from its abstract'
    problems += [f"{i}: cited but no verified quote from it" for i in sorted(cited & by_id.keys() - backed)]
    if problems:
        return False, "citation check failed:\n- " + "\n- ".join(dict.fromkeys(problems))
    return True, f"ALL {len(cited)} CITED PAPERS BACKED BY VERIFIED QUOTES"
