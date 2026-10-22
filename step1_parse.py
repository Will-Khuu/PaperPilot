"""Step 1: parse one paper and look hard at what comes out.

Setup:
    pip install pymupdf

Usage:
    python step1_parse.py papers/attention.pdf              # stats + writes out/attention.txt and out/attention.clean.txt
    python step1_parse.py papers/attention.pdf --blocks 3   # layout blocks on page 3

Nothing here is clever on purpose. The goal is to see what a PDF really
gives you before any later step depends on it.

The raw and cleaned text are both written so you can diff them and see
exactly what each cleaning rule removed or joined.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf  # PyMuPDF


def parse(pdf_path):
    """Return page numbers, body text and layout-separated footnotes."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
            sizes = Counter()
            for block in blocks:
                for line in block["lines"]:
                    for span in line["spans"]:
                        sizes[round(span["size"])] += len(span["text"])
            body_size = sizes.most_common(1)[0][0] if sizes else 0
            body, footnotes = [], []
            for block in blocks:
                text = "".join("".join(s["text"] for s in line["spans"]) + "\n"
                               for line in block["lines"])
                target = footnotes if is_footnote(block, body_size, page.rect.height) else body
                target.append(text)
            pages.append({"page": i, "text": "".join(body), "footnotes": "".join(footnotes)})
    return pages


def is_footnote(block, body_size, page_height):
    spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
    if not spans or block["bbox"][1] < page_height * 0.7:
        return False
    text = "".join(s["text"] for s in block["lines"][0]["spans"]).lstrip()
    marked = spans[0]["flags"] & 1 or re.match(r"^[∗†‡§¶]|^\d+[^\W\d_]", text)
    if re.match(r"^\d+(?:st|nd|rd|th)\b", text):
        marked = False
    sizes = Counter()
    for span in spans:
        sizes[round(span["size"])] += len(span["text"])
    return bool(marked and sizes.most_common(1)[0][0] < body_size)


def paper_title(pdf_path):
    """Join the largest horizontal text, excluding rotated margin stamps."""
    with pymupdf.open(pdf_path) as doc:
        spans = [s for b in doc[0].get_text("dict")["blocks"] for line in b.get("lines", [])
                 if line["dir"] == (1.0, 0.0) for s in line["spans"] if s["text"].strip()]
    largest = max(s["size"] for s in spans)
    return " ".join(s["text"].strip() for s in spans if abs(s["size"] - largest) < 0.1)


# Known boilerplate, matched as whole passages. Each entry was seen in a real
# paper's output; add one only after you see it, so the list stays honest.
BOILERPLATE = [
    re.compile(r"Provided proper attribution is provided.*?scholarly works\.\n", re.DOTALL),
]

HEADING_NUMBER = re.compile(r"^\d{1,2}(\.\d{1,2})*$")  # "3", "3.2", "3.2.1"
HEADING_TITLE = re.compile(r"^[A-Z][A-Za-z&\- ]*$")  # plain words: no digits, brackets or "+"


WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def clean(pages):
    """Same shape in and out, so Step 2 does not care whether this ran."""
    # Every word the paper uses, so hyphen fixes can check how the paper
    # itself spells a word elsewhere.
    vocab = {w.lower() for p in pages for w in WORD.findall(p["text"] + "\n" + p.get("footnotes", ""))}
    return [{"page": p["page"], "text": clean_page(p["page"], p["text"], vocab),
             "footnotes": clean_page(p["page"], p["footnotes"], vocab) if p.get("footnotes") else ""}
            for p in pages]


def clean_page(page_number, text, vocab):
    for pattern in BOILERPLATE:
        text = pattern.sub("", text)
    lines = text.splitlines()
    lines = drop_page_number(page_number, lines)
    lines = drop_word_runs(lines)
    lines = join_heading_numbers(lines)
    lines = join_hyphen_breaks(lines, vocab)
    return "\n".join(lines) + "\n"


def drop_page_number(page_number, lines):
    """Drop the last line only if it is this page's own number.

    A looser rule ("drop any line that is just a number") would also delete
    section numbers like "3" and table cells like "512".
    """
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if lines and lines[-1].strip() == str(page_number):
        lines = lines[:-1]
    return lines


def drop_word_runs(lines, min_run=10):
    """Drop 10+ one-token lines in a row when most of them are words.

    This is what attention heatmaps turn into (one word per line). Tables
    also give one token per line, but those tokens are mostly numbers, so
    requiring words leaves tables alone.
    """
    kept, run = [], []

    def flush():
        wordy = sum(1 for line in run if re.search(r"[A-Za-z]", line))
        if len(run) < min_run or wordy <= len(run) / 2:
            kept.extend(run)
        run.clear()

    for line in lines:
        if line.strip() and len(line.split()) == 1:
            run.append(line)
        else:
            flush()
            kept.append(line)
    flush()
    return kept


def join_heading_numbers(lines):
    """Put "3.2.1" and "Scaled Dot-Product Attention" back on one line.

    Step 2 splits at headings, so a heading must be one recognizable line.
    Tables also put a number on one line and a name on the next ("23.75",
    then "Deep-Att + PosUnk [39]"); gluing those would pin a score on the
    wrong model. So the title must be short plain words, which table rows
    with brackets, "+" or "et al. (2006)" are not.
    """
    out, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if HEADING_NUMBER.match(line) and HEADING_TITLE.match(nxt) and len(nxt.split()) <= 6:
            out.append(f"{line} {nxt}")
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return out


HYPHEN_LOG = []  # (kept or dropped, joined word), printed by main() for checking


def join_hyphen_breaks(lines, vocab):
    """Move the second half of a word split at a line end back up.

    A line ending in "-" is either a split word ("transduc-" + "tion") or a
    real compound that happened to break there ("position-" + "wise"). The
    line alone can't tell them apart, so ask the paper: if it spells the
    word without a hyphen somewhere else, drop the hyphen; otherwise keep
    it. Keeping it is the safer default, because "sequence-aligned" stays
    readable while "sequencealigned" is a word nobody would search for.
    """
    lines = list(lines)
    for i in range(len(lines) - 1):
        if lines[i] is None or lines[i + 1] is None:
            continue
        line, nxt = lines[i].rstrip(), lines[i + 1].lstrip()
        if re.search(r"[A-Za-z]-$", line) and nxt[:1].isalpha():
            first, _, rest = nxt.partition(" ")
            stem = WORD.findall(line)[-1]  # the word before the hyphen, e.g. "transduc"
            plain = stem + re.match(r"[A-Za-z]*", first).group()
            drop = nxt[:1].islower() and "-" not in first and plain.lower() in vocab
            lines[i] = (line[:-1] if drop else line) + first
            lines[i + 1] = rest if rest else None  # None: the whole line moved up
            HYPHEN_LOG.append(("dropped" if drop else "kept", WORD.findall(lines[i])[-1]))
    return [line for line in lines if line is not None]


def references_start(pages):
    """Page where the "References" heading appears, or None.

    Nothing is deleted. Step 2 uses this to label those chunks
    section="References" so retrieval can leave them out by default.
    """
    for p in pages:
        if any(line.strip().lower() == "references" for line in p["text"].splitlines()):
            return p["page"]
    return None


def repeated_lines(pages, min_share=0.5):
    """Lines that appear on at least half the pages.

    These are usually running headers or footers. If they stay in, they get
    embedded into every chunk and add noise to search.
    """
    counts = Counter()
    for p in pages:
        counts.update({line.strip() for line in p["text"].splitlines() if line.strip()})
    cutoff = max(2, int(len(pages) * min_share))
    return [(line, n) for line, n in counts.most_common() if n >= cutoff]


def hyphen_breaks(pages):
    """Lines ending in '-', where a word was split across two lines."""
    return sum(
        1
        for p in pages
        for line in p["text"].splitlines()
        if line.rstrip().endswith("-") and len(line.rstrip()) > 1
    )


def show_blocks(pdf_path, page_number):
    """Print each layout block with its position on the page.

    Two columns show up as two groups of x values, for example blocks starting
    near x=50 and others near x=310. Compare this order with the text order.
    """
    page = pymupdf.open(pdf_path)[page_number - 1]
    print(f"page {page_number}, width {page.rect.width:.0f}, height {page.rect.height:.0f}")
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        preview = " ".join(text.split())[:70]
        print(f"x={x0:5.0f}-{x1:<5.0f} y={y0:5.0f}  {preview}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])

    if "--blocks" in sys.argv:
        show_blocks(path, int(sys.argv[sys.argv.index("--blocks") + 1]))
        return

    pages = parse(path)
    cleaned = clean(pages)
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{path.stem}.txt"
    clean_file = out_dir / f"{path.stem}.clean.txt"
    for target, version in [(out_file, pages), (clean_file, cleaned)]:
        target.write_text(
            "\n".join(f"=== page {p['page']} ===\n{p['text']}"
                      + (f"--- footnotes ---\n{p['footnotes']}" if p["footnotes"] else "")
                      for p in version),
            encoding="utf-8",
        )

    print(f"{path.name}: {len(pages)} pages, {sum(len(p['text']) for p in pages):,} characters")
    print("characters per page:", [len(p["text"]) for p in pages])
    print(f"lines ending in a hyphen: {hyphen_breaks(pages)}")
    print(f"after cleaning: {sum(len(p['text']) for p in cleaned):,} characters,"
          f" {hyphen_breaks(cleaned)} lines ending in a hyphen")
    print("characters per page after cleaning:", [len(p["text"]) for p in cleaned])
    print(f"references start on page: {references_start(cleaned)}")
    print("hyphen decisions:")
    for verdict, word in HYPHEN_LOG:
        print(f"  {verdict:>7}  {word}")

    repeats = repeated_lines(pages)
    if repeats:
        print("lines repeated on many pages (likely headers or footers):")
        for line, n in repeats[:8]:
            print(f"  {n:>3}x  {line[:80]}")
    else:
        print("no lines repeated across pages")

    print(f"\nraw text written to {out_file}, cleaned text to {clean_file}")


if __name__ == "__main__":
    main()
