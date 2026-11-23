"""Fetch real arXiv metadata (authors, title, year, abstract) for the curated
citation sets and emit refs.json + refs.bib. Keyless arXiv API."""
import json, re, time
from pathlib import Path
import httpx
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"

T2_IDS = ["2608.11017","2608.13438","2608.09857","2608.09516","2608.12990","2607.10350",
          "2608.09410","2608.12626","2608.13474","2608.13103","2608.11739","2607.06990",
          "2607.24744","2608.05738","2608.11350","2608.12707"]
T5_IDS = ["2608.11888","2608.12273","2607.18548","2608.09303","2608.09857","2608.13438",
          "2608.06830","2608.11350","2607.06990","2608.09516","2607.22651","2608.10915",
          "2608.07734","2608.05588"]

ALL = list(dict.fromkeys(T2_IDS + T5_IDS))
OUT = Path(__file__).parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def bibkey(authors, year, aid):
    if authors:
        last = re.sub(r"[^A-Za-z]", "", authors[0].split()[-1])
    else:
        last = "arxiv"
    return f"{last}{year}_{aid.replace('.','')}"


def fetch(ids):
    recs = {}
    with httpx.Client(timeout=60) as c:
        for i in range(0, len(ids), 8):
            batch = ids[i:i+8]
            r = c.get("https://export.arxiv.org/api/query",
                      params={"id_list": ",".join(batch), "max_results": len(batch)})
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for e in root.findall(f"{ATOM}entry"):
                idurl = e.findtext(f"{ATOM}id") or ""
                m = re.search(r"abs/([0-9]+\.[0-9]+)", idurl)
                aid = m.group(1) if m else idurl
                title = " ".join((e.findtext(f"{ATOM}title") or "").split())
                summary = " ".join((e.findtext(f"{ATOM}summary") or "").split())
                pub = e.findtext(f"{ATOM}published") or ""
                year = pub[:4] if pub else "2026"
                authors = [a.findtext(f"{ATOM}name") for a in e.findall(f"{ATOM}author")]
                recs[aid] = {"arxiv_id": aid, "title": title, "authors": authors,
                             "year": year, "abstract": summary,
                             "key": bibkey(authors, year, aid)}
            time.sleep(1)
    return recs


def to_bib(rec):
    auth = " and ".join(rec["authors"]) if rec["authors"] else "arXiv"
    return (f"@article{{{rec['key']},\n"
            f"  title={{{rec['title']}}},\n"
            f"  author={{{auth}}},\n"
            f"  journal={{arXiv preprint arXiv:{rec['arxiv_id']}}},\n"
            f"  year={{{rec['year']}}}\n}}\n")


if __name__ == "__main__":
    recs = fetch(ALL)
    print("fetched", len(recs), "of", len(ALL))
    (OUT / "refs.json").write_text(json.dumps(
        {"t2_ids": T2_IDS, "t5_ids": T5_IDS, "records": recs}, indent=2))
    with open(OUT / "refs.bib", "w") as f:
        for aid, rec in recs.items():
            f.write(to_bib(rec) + "\n")
    for aid in ALL:
        if aid not in recs:
            print("MISSING:", aid)
    print("wrote", OUT / "refs.json", "and refs.bib")
