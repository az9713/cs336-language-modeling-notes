# Stanford CS336 — Language Modeling from Scratch

Independent, transcript-grounded study notes for **Stanford CS336, Spring 2026**, following all **18 recordings** in the Stanford Online playlist.

**[Read the live study library](https://az9713.github.io/stanford-cs336/)** · [Interactive math lab](https://az9713.github.io/stanford-cs336/lab.html) · [Primary references](https://az9713.github.io/stanford-cs336/references.html) · [Timestamp source map](https://az9713.github.io/stanford-cs336/source-map.html) · [Validation and limitations](https://az9713.github.io/stanford-cs336/validation.html)

## Stanford course and source recordings

The course, lectures, and original instructional ideas belong to their respective Stanford instructors and contributors. The course is taught by **Percy Liang and Tatsunori Hashimoto**, with a guest lecture by **Dan Fu**.

- [Official Stanford CS336 course website](https://cs336.stanford.edu/)
- [Stanford Online: complete Spring 2026 YouTube playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [Source YouTube video: Lecture 1 — Overview, Tokenization](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [Official Stanford lecture materials](https://github.com/stanford-cs336/lectures), cross-checked against [this pinned revision](https://github.com/stanford-cs336/lectures/tree/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e).

This is an independent educational companion, not an official Stanford publication or an endorsement by Stanford. Recordings are linked to their original source. Raw captions, downloaded slides, and research-page copies are not redistributed here.

## Our contribution beyond the transcript

The recordings provide the topic sequence and motivating questions. Our contribution is the independently authored teaching layer that turns those topics into a navigable study collection:

- **Mathematical development:** explicit definitions, assumptions, units, derivations, numerical checks, and explanations of where a model stops applying.
- **Worked teaching cases:** original examples that connect abstract mechanisms to tokenization, attention, memory, compute, scaling, and evaluation decisions.
- **Researched extensions:** primary papers, official documentation, and implementations integrated with the relevant claims, including comparisons and limitations beyond the lecture discussion.
- **Runnable experiments:** a cumulative CPU Python lab, with PyTorch where useful, covering byte-pair encoding, model resource estimates, rotary positions, gated delta memory, and streaming attention.
- **Exercises and worked solutions:** derivation, computation, assumption checking, and design questions with reasoning exposed in expandable solutions.
- **Reading tools:** timestamped recording links, a section-level source map, searchable chapter navigation, locally bundled mathematical rendering, interactive calculators, responsive pages, and print styles.

These are educational elaborations, not claims of new scientific discoveries. A timestamp identifies the nearby lecture discussion; it does not imply that the instructor supplied every derivation, example, or extension in that section. Research findings remain attributed to their cited authors.

## Current edition and live chapters

**All 18 recordings have notes. Chapters 1–5 have completed the expanded textbook rewrite; chapters 6–18 remain in the shorter study-note edition.** The later chapters' full-depth rewrites are pending. The collection currently contains approximately 48,400 words and 127 timestamped section links.

Each chapter link below opens the rendered HTML on GitHub Pages. GitHub's README does not execute embedded HTML applications; the live links open the complete pages with mathematics and interactive controls.

| Playlist position | Live chapter | Source recording | Edition |
|---|---|---|---|
| 01 | [Overview and tokenization](https://az9713.github.io/stanford-cs336/lecture-01.html) | [Watch](https://www.youtube.com/watch?v=JuoVZkPBiKk) | Expanded textbook chapter |
| 02 | [Tensors and resource accounting](https://az9713.github.io/stanford-cs336/lecture-02.html) | [Watch](https://www.youtube.com/watch?v=kuYAsz7zspQ) | Expanded textbook chapter |
| 03 | [Transformer architectures](https://az9713.github.io/stanford-cs336/lecture-03.html) | [Watch](https://www.youtube.com/watch?v=lVynu4bo1rY) | Expanded textbook chapter |
| 04 | [Attention alternatives and MoE](https://az9713.github.io/stanford-cs336/lecture-04.html) | [Watch](https://www.youtube.com/watch?v=cKSwj_qZ8Jg) | Expanded textbook chapter |
| 05 | [GPUs, TPUs, and data movement](https://az9713.github.io/stanford-cs336/lecture-05.html) | [Watch](https://www.youtube.com/watch?v=izZba4UA7iY) | Expanded textbook chapter |
| 06 | [Kernels and Triton](https://az9713.github.io/stanford-cs336/lecture-06.html) | [Watch](https://www.youtube.com/watch?v=xnDHaNUvHBg) | Shorter study notes |
| 07 | [Collectives and parallel training](https://az9713.github.io/stanford-cs336/lecture-07.html) | [Watch](https://www.youtube.com/watch?v=SzpOcwdIL0Y) | Shorter study notes |
| 08 | [Sharding and hybrid parallelism](https://az9713.github.io/stanford-cs336/lecture-08.html) | [Watch](https://www.youtube.com/watch?v=6-cXp-aOmdg) | Shorter study notes |
| 09 | [Scaling laws](https://az9713.github.io/stanford-cs336/lecture-09.html) | [Watch](https://www.youtube.com/watch?v=Q15rhEWZPQ4) | Shorter study notes |
| 10 | [Inference](https://az9713.github.io/stanford-cs336/lecture-10.html) | [Watch](https://www.youtube.com/watch?v=EfM546A79aM) | Shorter study notes |
| 11 | [Scaling recipes and optimizers](https://az9713.github.io/stanford-cs336/lecture-11.html) | [Watch](https://www.youtube.com/watch?v=vTfEyOyzV9E) | Shorter study notes |
| 12 | [Evaluation](https://az9713.github.io/stanford-cs336/lecture-12.html) | [Watch](https://www.youtube.com/watch?v=JpAxdTWQJxM) | Shorter study notes |
| 13 | [Data sources and provenance](https://az9713.github.io/stanford-cs336/lecture-13.html) | [Watch](https://www.youtube.com/watch?v=-qm0ln33G24) | Shorter study notes |
| 14 | [Filtering and data mixtures](https://az9713.github.io/stanford-cs336/lecture-14.html) | [Watch](https://www.youtube.com/watch?v=5sxHosTLPF8) | Shorter study notes |
| 15 | [Mid-training, SFT, and preferences](https://az9713.github.io/stanford-cs336/lecture-15.html) | [Watch](https://www.youtube.com/watch?v=2oH6PWPrYFo) | Shorter study notes |
| 16 | [RL with verifiable rewards](https://az9713.github.io/stanford-cs336/lecture-16.html) | [Watch](https://www.youtube.com/watch?v=dIFAi87Ws4E) | Shorter study notes |
| 17 | [Multimodality](https://az9713.github.io/stanford-cs336/lecture-17.html) | [Watch](https://www.youtube.com/watch?v=26FtD08ZpOU) | Shorter study notes |
| 18 | [Dan Fu: inference and looped models](https://az9713.github.io/stanford-cs336/lecture-18.html) | [Watch](https://www.youtube.com/watch?v=9EEm4iMAF5s) | Shorter study notes |

The guest lecture is numbered 18 by playlist position. Chapter titles emphasize the material actually discussed; the source map records differences from the playlist's broader labels.

## Run the CPU examples

Clone the repository and run commands from its root:

```sh
git clone https://github.com/az9713/stanford-cs336.git
cd stanford-cs336
python -m pip install torch
python -m unittest discover -s tests -v
```

The public lab has **14 unit tests**. The examples use CPU tensors and do not require a GPU. PyTorch 2.11.0 was used for the recorded checks. Other dependencies in the public lab are from Python's standard library. Code listings in the expanded chapters explain their inputs, assumptions, and expected results.

To serve the pages locally:

```sh
python -m http.server 8000
```

Then open [the local reading library](http://localhost:8000/). The HTML and bundled MathJax also support local reading; recordings and external references require an internet connection.

## Publication and validation

GitHub Pages serves the static files from the root of the `main` branch. `.nojekyll` preserves the static asset layout. This repository includes the rendered HTML, editable chapter Markdown, CPU lab, and lab tests. Private research inputs and authoring audit files are excluded.

Validation covers caption provenance and playlist order during authoring, timestamp bounds, local links and anchors, selected numerical examples, executable listings, and desktop/mobile browser behavior. The [live validation report](https://az9713.github.io/stanford-cs336/validation.html) describes the scope and limits. No large-scale GPU training run or independent replication of the cited empirical research is claimed.

Stanford materials and linked papers retain their respective rights. The bundled MathJax renderer retains its [upstream license](assets/mathjax-LICENSE). This repository does not relicense third-party course content.
