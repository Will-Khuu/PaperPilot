"""Step 2: cut one paper into chunks at section headings and look at them.

Usage:
    python step2_chunk.py papers/attention.pdf                  # 220 tokens, 80 overlap
    python step2_chunk.py papers/attention.pdf --size 600 --overlap 80

Prints every chunk (id, section, pages, size, start of text) and writes them
all to out/<name>.chunks.jsonl so you can read whole chunks.
"""
import json
import re
import sys
from pathlib import Path

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from tokenizers import Tokenizer

from step1_parse import HEADING_NUMBER, clean, parse

# Numbered headings after Step 1's cleaning: "3.2.1 Scaled Dot-Product Attention".
NUMBERED_HEADING = re.compile(HEADING_NUMBER.pattern[:-1] + r" [A-Z][A-Za-z&\- ]*$")

# Headings papers use without a number. Matched as the whole line.
NAMED_HEADINGS = {"abstract", "acknowledgements", "acknowledgments", "references", "appendix"}

# Count with the embedding model's own tokenizer, because that's what the 256
# limit is measured in. Step 2 first used words x 1.3, which undercounted
# citations, numbers and symbols by up to 2x (15 of 49 chunks were over).
EMBEDDING_MODEL = ONNXMiniLM_L6_V2()
EMBEDDING_MODEL(["initialize tokenizer"])  # embedding once downloads the model; .tokenizer alone doesn't
# This is a copy with truncation off, so long text counts in full and
# Chroma's own tokenizer (which truncates at 256) is left untouched.
TOKENIZER = Tokenizer.from_str(EMBEDDING_MODEL.tokenizer.to_str())
TOKENIZER.no_truncation()
TOKENIZER.no_padding()


def count_tokens(text):
    """Word pieces the model will see, including its two markers ([CLS], [SEP])."""
    return len(TOKENIZER.encode(text).ids)


def word_tokens(word):
    """Word pieces for one word. A text's count is the sum of its words' counts + 2."""
    return len(TOKENIZER.encode(word, add_special_tokens=False).ids)


def is_heading(line, first_on_page, after_references):
    line = line.strip()
    if len(line.split()) > 6:
        return False
    if NUMBERED_HEADING.match(line) or line.lower() in NAMED_HEADINGS:
        return True
    # Appendices come after the references and start on a new page with a
    # short title ("Attention Visualizations"). Without this, they'd be
    # labeled as references and filtered out with them.
    return after_references and first_on_page and line.istitle() and not line.endswith(".")


def is_footnote_start(line):
    """Footnotes marked with a symbol: "∗Equal contribution", "†Work performed..."."""
    return line[:1] in "∗†‡" and line[1:2].isalpha()


def split_sections(pages):
    """Group the paper's words into sections.

    Returns [{"section": str, "words": [(word, page), ...]}]. Every word keeps
    its page number, so a chunk knows exactly which pages it came from.
    Text before the first heading (title, authors) becomes "Front matter".

    Footnotes sit at the bottom of a page, after whatever section was
    running, so they'd be glued onto it (page 1's author credits onto the
    Abstract). From a footnote mark to the end of that page goes into a
    "Footnotes" section instead, and the next page picks the running
    section back up.
    """
    sections = {"Front matter": []}  # section name -> words, in reading order
    current = "Front matter"
    after_references = False
    for p in pages:
        lines = [line for line in p["text"].splitlines() if line.strip()]
        in_footnotes = False
        for i, line in enumerate(lines):
            if not in_footnotes and is_footnote_start(line):
                in_footnotes = True
            if in_footnotes:
                sections.setdefault("Footnotes", []).extend((w, p["page"]) for w in line.split())
            elif is_heading(line, i == 0, after_references):
                current = line.strip()
                sections.setdefault(current, [])
                after_references = after_references or current.lower() == "references"
            else:
                sections[current].extend((w, p["page"]) for w in line.split())
    return [{"section": name, "words": words} for name, words in sections.items() if words]


# The one place the chunk settings are decided; step3 imports these.
# 220 fits the embedding model's 256 limit with margin. Overlap 80 won Step 6's
# sweep by the pre-set rule (+2 answers at rank 1, none lost from the top 5),
# a marginal gain at the noise level of 20 questions.
CHUNK_SIZE = 220
CHUNK_OVERLAP = 80


def chunk(pages, paper_id, title, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut each section into windows of about `size` tokens.

    A chunk never crosses a section boundary. A long section is cut into
    several windows, and each window repeats the last `overlap` tokens of
    the one before it, so a sentence cut at a boundary is whole in at least
    one of them. Every chunk starts with its section heading, so a window
    from the middle of a section still says what it's about.

    Why 220: Chroma's default embedding model (all-MiniLM-L6-v2) reads only
    the first 256 word pieces and silently drops the rest.
    """
    chunks = []
    for s in split_sections(pages):
        words = s["words"]
        costs = [word_tokens(w) for w, _ in words]
        budget = size - count_tokens(s["section"])  # the heading and markers come first
        start = 0
        while True:
            # Take words until the next one would go over the budget.
            end, used = start, 0
            while end < len(words) and used + costs[end] <= budget:
                used += costs[end]
                end += 1
            end = max(end, start + 1)  # always move forward, even past one giant "word"
            piece = words[start:end]
            text = s["section"] + "\n" + " ".join(w for w, _ in piece)
            chunks.append({
                "id": f"{paper_id}-{len(chunks):03d}",
                "text": text,
                "paper_id": paper_id,
                "title": title,
                "section": s["section"],
                "page": piece[0][1],
                "page_end": piece[-1][1],
                "chunk_index": len(chunks),
            })
            if end >= len(words):
                break
            # The next window starts `overlap` tokens before this one ended.
            back = 0
            start_next = end
            while start_next > start + 1 and back + costs[start_next - 1] <= overlap:
                start_next -= 1
                back += costs[start_next]
            start = start_next
    return chunks


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else CHUNK_SIZE
    overlap = int(sys.argv[sys.argv.index("--overlap") + 1]) if "--overlap" in sys.argv else CHUNK_OVERLAP

    pages = clean(parse(path))
    # The PDF's title metadata is often empty (it is for this paper), so use
    # the first line of page 1, which Step 1's cleaning made the title.
    title = pages[0]["text"].splitlines()[0].strip()
    chunks = chunk(pages, paper_id=path.stem, title=title, size=size, overlap=overlap)

    out_file = Path("out") / f"{path.stem}.chunks.jsonl"
    out_file.parent.mkdir(exist_ok=True)
    out_file.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")

    print(f'"{title}": {len(chunks)} chunks (size {size}, overlap {overlap} tokens)\n')
    for c in chunks:
        pages_label = f"p{c['page']}" if c["page"] == c["page_end"] else f"p{c['page']}-{c['page_end']}"
        body = " ".join(c["text"].split("\n", 1)[1].split())
        print(f"{c['id']}  {pages_label:<7} {count_tokens(c['text']):>4} tok  {c['section']}")
        print(f"    {body[:110]}...")
    sizes = [count_tokens(c["text"]) for c in chunks]
    print(f"\nsizes: min {min(sizes)}, max {max(sizes)}, mean {sum(sizes) // len(sizes)} tokens")
    print(f"over the model's 256 limit: {sum(n > 256 for n in sizes)}")
    print(f"chunks spanning two pages: {sum(c['page'] != c['page_end'] for c in chunks)}")
    print(f"full chunks written to {out_file}")


if __name__ == "__main__":
    main()
