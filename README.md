# PaperPilot

Ask questions about a research paper and get answers that cite the exact section and page.

PaperPilot is a retrieval-augmented generation (RAG) pipeline built one stage at a time, so each stage can be inspected,
measured and swapped on its own. It currently works on *Attention Is All You Need* (Vaswani et al., 2017).

```
$ python step4_answer.py ask "How many attention heads does the Transformer use?"

RETRIEVED
  [1] attention-016  d=0.412  p. 5  3.2.2 Multi-Head Attention
  ...

ANSWER
The Transformer uses h = 8 parallel attention layers, or heads [1]. For each head,
dk = dv = dmodel/h = 64 [1]. ...

SOURCES
  [1] Attention Is All You Need, 3.2.2 Multi-Head Attention, p. 5  (attention-016)
```

## How it works

| Stage | Script | What it does |
| --- | --- | --- |
| 1. Parse | `step1_parse.py` | Extracts text page by page with PyMuPDF, then applies narrow cleaning rules: drops page numbers and figure noise, rejoins split words and section headings. |
| 2. Chunk | `step2_chunk.py` | Splits at section headings, then into overlapping windows of at most 220 tokens (40 overlap). Each chunk keeps its section and page range. |
| 3. Embed and retrieve | `step3_embed.py` | Embeds chunks with all-MiniLM-L6-v2 into a local Chroma collection (cosine distance) and returns the closest chunks to a question. |
| 4. Answer | `step4_answer.py` | Sends the top 5 chunks to Claude as numbered excerpts. The answer cites `[n]`, and each `[n]` is mapped back to its section and page. |
| 5. Evaluate | `step5_eval.py` | Scores retrieval on 20 labeled questions: hit rate at 1, 5 and 10, by page and by answer phrase, per question type. |

## Design decisions

- **Chunks never cross a section heading.** A chunk about one topic gives a focused embedding and an accurate section
  label for citations.
- **Chunk size is set by the embedding model's limit.** all-MiniLM-L6-v2 silently truncates input after 256 word
  pieces. Counting with the model's own tokenizer (a word-count estimate was off by up to 2x) keeps every chunk under it.
- **Citations map by position, not by the model's memory.** `[1]` is the first excerpt in the prompt, so the page comes
  from metadata attached at ingest, never from the LLM.
- **The paper's own citations are rewritten before generation.** The source text contains bracketed references like
  `[2]`; left as-is, the model could copy one and it would point to the wrong excerpt. They become `(ref 2)`.
- **The model must admit what the excerpts don't cover.** It answers with one fixed sentence when the excerpts are
  insufficient, and any answer without citations is flagged as likely outside knowledge.
- **Retrieval is tested before generation.** A bad answer can then be traced to either the search or the LLM.

## Results so far

- **Retrieval** on 20 labeled questions (4 each: fact, paraphrase, table number, concept, who/relation), at
  220-token chunks with 40 overlap:

  | | Hit rate at 1 | Hit rate at 5 | Hit rate at 10 |
  | --- | --- | --- | --- |
  | Right page retrieved | 60% | 95% | 100% |
  | Retrieved chunk contains the answer | 45% | 90% | 100% |

  The only top-5 miss is a paraphrased question ("How does the model know the order of the words?" vs. the paper's
  "positional encoding"). Scoring the answer phrase as well as the page matters: for one table question, a chunk from
  the right page ranked first while the chunk holding the number ranked ninth. The questions were written with the
  paper in hand, so these numbers are optimistic; the main gap is getting the answer chunk ranked first.
- **Answers:** all 8 answered; all 17 cited claims were checked against the cited text and matched. For "What is BERT?"
  (not in the paper) the model declined instead of answering from training data.
- **Known limits:** numbered footnotes can land in the wrong section's chunk, and table text loses its row and column
  structure.

## Run it

Requires Python 3.9+ and an Anthropic API key for stage 4.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p papers
curl -L -o papers/attention.pdf https://arxiv.org/pdf/1706.03762

python step1_parse.py papers/attention.pdf           # parsed and cleaned text in out/
python step2_chunk.py papers/attention.pdf           # chunks in out/attention.chunks.jsonl
python step3_embed.py ingest papers/attention.pdf    # build the Chroma index
python step3_embed.py ask "What is scaled dot-product attention?"

export ANTHROPIC_API_KEY=...                         # or load it from your OS keychain
python step4_answer.py ask "What is scaled dot-product attention?"
python step4_answer.py ask "..." --dry-run           # print the exact prompt, no API call
python step4_answer.py selftest                      # offline checks, no API key needed

python step5_eval.py run --label baseline            # retrieval scores, saved to out/eval/baseline.json
```

## Roadmap

- [x] Parse, chunk, embed, retrieve
- [x] Answers with page-level citations
- [x] Evaluation set of 20 questions with known answer pages; hit rate at 1, 5 and 10
- [ ] Tune chunk size and overlap against that baseline, aiming at rank-1 precision
- [ ] FastAPI backend and Streamlit chat UI
- [ ] Multiple papers and a compare mode

## License

MIT
