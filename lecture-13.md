# Lecture 13 — Data sources, datasets, and provenance

The lecture shifts attention from how to optimize a model to what it should learn from. It follows data from live services through crawling and extraction to curated pretraining mixtures, then surveys major dataset-building approaches. The central idea is that a dataset is the result of many choices, not a neutral sample of “the internet.”

## 1. The data distribution is part of the model specification [00:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=0s)

Maximum-likelihood training minimizes expected loss under a training distribution $P_{\rm train}$. Deployment quality is evaluated under another distribution $P_{\rm use}$. Even perfect optimization of the first objective does not guarantee optimal performance on the second.

For source distributions $P_1,\ldots,P_K$ and mixture weights $w_i\ge0$ summing to one,

$$P_{\rm train}=\sum_{i=1}^K w_iP_i.$$

The weights determine how often the model sees code, scientific prose, conversation, different languages, and other content. Filtering and deduplication further change each $P_i$. Data engineering therefore changes the statistical problem itself.

**Worked example.** A corpus contains 90% general web text and 10% mathematical solutions by tokens. A model trained for one billion token presentations sees about 100 million mathematics tokens under proportional sampling. Reweighting mathematics to 30% triples exposure, but it also displaces other material and may repeat a small math subset many times. The useful quantity is both exposure and unique coverage.

## 2. A live service is not a static text collection [04:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=240s)

Modern websites can generate content dynamically, require interaction, personalize output, or expose only part of a document at a URL. A crawl captures a particular view at a particular time. Coverage depends on discovery, access, rendering, rate limits, and the crawler's choices.

Common Crawl provides snapshots and associated artifacts rather than a complete census of the web. Its [official overview](https://commoncrawl.org/overview) describes the public crawl resource. WARC records preserve fetched web responses, WAT files provide extracted metadata, and WET files contain extracted text. A downstream pipeline may prefer raw records when it needs a different extraction strategy.

**Worked example.** A page's HTML may contain navigation, cookie notices, a short visible article, and a script that loads the rest. A simple text extractor can retain the navigation while missing the body. The resulting training document exists, tokenizes, and passes basic length checks, but it does not represent the intended source.

Store crawl date, source URL, content hash, extraction version, and processing decisions. A later model failure may otherwise be impossible to trace back to the actual bytes used for training.

## 3. Access, licensing, and use are separate questions [09:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=540s)

The recording spends substantial time on legal and institutional context. Its case descriptions are historical lecture content, not current legal conclusions. Technical accessibility does not establish permission to reuse or redistribute a work. A source's license, access conditions, and the intended use require separate consideration.

In the United States, [17 U.S.C. §107](https://www.copyright.gov/title17/92chap1.html#107) identifies multiple factors for fair-use analysis, including purpose, the work's nature, the portion used, and market effects. That framework is fact-specific; it does not provide a blanket answer for all model training. This chapter does not repeat the recording's simplified copyright-duration claim or attempt to update individual lawsuits.

For a dataset pipeline, record the actual license evidence and acquisition route rather than inferring permission from a file extension or hosting site. Provenance is useful regardless of the final legal assessment because it makes the dataset auditable and allows later corrections.

## 4. Different sources preserve different kinds of structure [32:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=1920s)

Web pages, encyclopedia dumps, scientific papers, code repositories, and question–answer forums have different strengths and extraction problems. An encyclopedia has revision history and templates. A scientific paper may have LaTeX source as well as PDF, with equations and references whose order can be damaged by extraction. A repository contains code, generated files, vendored dependencies, binaries, licenses, and tests.

A useful document model is more than a text field. It can include source identifier, timestamp, document type, language, license, structural boundaries, quality scores, and relationships to other documents. For code, repository and commit identity can matter more than a flat list of files.

**Worked example.** Training on a Python source file and its generated minified JavaScript bundle as though they were equivalent prose wastes capacity and can distort the source mixture. Similarly, extracting a two-column PDF in the wrong reading order can combine unrelated sentences while preserving every word. Token count alone cannot detect either problem.

The lecture also discusses data poisoning through public sources. The narrow engineering implication is to preserve source identity and inspect unusual changes or suspicious repeated content. Broad ingestion is not automatically trustworthy simply because it is large.

## 5. Dataset history reveals changing selection criteria [45:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=2700s)

The recording surveys datasets and recipes including WebText-style selection, CCNet, C4, The Pile, RefinedWeb, Dolma, FineWeb, and DCLM. Their important differences are not only size. They use different source selections, extraction rules, language policies, filters, deduplication methods, and quality proxies.

A filter designed for prose can accidentally exclude valuable code. A quality classifier trained on encyclopedia-like positives can favor encyclopedia style rather than all useful knowledge. A corpus assembled from many named sources can improve diversity while creating uneven duplication and licensing metadata.

The [FineWeb paper](https://arxiv.org/abs/2406.17557) is useful because it documents and ablates processing choices. The [Dolma paper](https://arxiv.org/abs/2402.00159) provides another primary treatment of an open pretraining corpus and its toolkit. Compare their stated construction and evaluation procedures rather than assuming a larger headline token count means a better training distribution.

**Worked example.** Corpus A contains 100 billion unique tokens, while corpus B contains 200 billion tokens but half are repeated copies. Their token counts differ by two, but their unique coverage may be similar. Their training effects can still differ because repetition changes sampling weights and optimization. Neither raw size nor a deduplicated size alone fully predicts quality.

## 6. Instruction-like structure already exists inside pretraining data [55:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=3300s)

Question–answer forums, tutorials, worked proofs, issue discussions, and code reviews resemble downstream interactions more closely than arbitrary prose. The distinction between pretraining and instruction training is therefore partly about formatting, selection, and weighting, not simply whether text contains questions.

This helps explain why a small source can have disproportionate value for a target use. A carefully selected set of worked solutions may expose reasoning patterns absent from a much larger generic crawl. But optimizing only for a narrow benchmark can reduce broader capability or contaminate evaluation.

**Worked example.** A code assistant needs more than function bodies. Repository issues teach problem descriptions; patches teach changes; tests provide behavioral constraints; discussions contain design rationale. Flattening all four into independent text fragments may lose the relationships that make the material useful for software engineering.

## 7. Code and permissively licensed corpora change the data boundary [70:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=4200s)

The lecture closes with source-specific collections and a more explicit licensing boundary. For code, useful processing includes identifying generated or binary artifacts, repository-level duplication, language, secrets or personal information, and the relationship between issues and changes. These are different tasks with different error costs.

[The Common Pile](https://arxiv.org/abs/2506.05209) investigates a corpus built from public-domain and openly licensed text. Its value here is as a documented alternative data-construction experiment. It does not establish that every item found on the same hosting sites has the same permissions or that an upstream synthetic-data generator's training provenance is fully resolved.

Distinguish the license of a dataset's packaging or code from the rights and conditions associated with its underlying documents. Likewise, a model's weight license and its training-data provenance are separate metadata categories.

## 8. Researched extension — provenance can be represented as a transformation graph

An independently useful design is a directed graph whose nodes are immutable source snapshots and derived artifacts. Edges represent transformations such as extraction, normalization, filtering, deduplication, and tokenization. Each edge records code version, parameters, and a hash of its input and output.

For example, a source document hash leads to an extracted-text hash, then a normalized-text hash, then a tokenized shard. A deduplication decision links the removed document to its retained representative. This lets an auditor ask not only “where did this shard come from?” but “which rule removed this document?”

The graph also helps with rollback. If an extractor corrupts equations, identify affected descendants rather than rebuilding unrelated data. If a source must be removed, locate derived shards that include it. The graph does not solve every retraining or unlearning problem, but it makes the data dependency explicit.

## 9. Exercises and worked solutions

**Exercise 1.** A source has 5 billion unique tokens and receives 20% of a 100-billion-token training run. How many average passes over that source occur?

<details><summary>Solution</summary>

Exposure is 20 billion tokens, so the average is four passes, assuming sampling is uniform within the source and all documents are eligible. This is an expectation; actual per-document repetition depends on the sampler.

</details>

**Exercise 2.** What metadata would you preserve for an arXiv-derived training document?

<details><summary>Solution guidance</summary>

Preserve identifier and version, acquisition date, source representation (PDF or LaTeX), license evidence, extraction tool/version, original and extracted hashes, and structural information such as sections and equation boundaries. A PDF and its source are not interchangeable text extractions.

</details>

**Exercise 3.** Why does a high-quality source classifier still need manual inspection?

<details><summary>Solution</summary>

Its label reflects the chosen positives and negatives. It may learn style, language, length, or website identity as proxies. Inspect accepted and rejected samples, including minority domains and languages, and evaluate the downstream consequence of the selection.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=-qm0ln33G24) and [official lecture](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_13.py).
- [Common Crawl documentation](https://commoncrawl.org/overview): crawl artifacts and scope.
- [FineWeb](https://arxiv.org/abs/2406.17557), [Dolma](https://arxiv.org/abs/2402.00159), and [Common Pile](https://arxiv.org/abs/2506.05209): dataset construction studies.
- [U.S. Copyright Office, statutory text](https://www.copyright.gov/title17/92chap1.html#107): the limited legal framework described above.
