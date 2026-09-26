# Lecture 17 — Multimodal representations and vision–language models

Despite the playlist label “Alignment – Multimodality,” the recording is primarily a multimodality lecture. It progresses from image–text representation learning through CLIP, vision transformers, SigLIP, visual instruction tuning, higher-resolution and video models, and interleaved multimodal generation. The recurring problem is how to compress very different signals into representations a common model can use.

## 1. A modality needs a representation and a learning signal [00:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=0s)

Text arrives as discrete symbols, while images and audio begin as dense signals. A multimodal model must choose which information to retain, how to encode position and time, and which objective connects the modalities. “Everything becomes tokens” is an interface description, not a guarantee that all modalities have equivalent statistics.

Let $f_\theta(I)$ encode image $I$ and $g_\phi(T)$ encode text $T$. These functions can produce a single global vector, a sequence of local vectors, or discrete codes. A global vector is convenient for retrieval but can discard small spatial details. A long patch sequence retains more information but increases computation and context use.

**Worked example.** A document page may be easy to classify as an invoice from a global image representation while its invoice number remains unreadable. Image understanding is not one task; the representation needed for broad semantics can be insufficient for precise OCR or spatial relationships.

## 2. CLIP learns matched image–text geometry [05:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=300s)

For a batch of $N$ matched image–text pairs, let normalized embeddings be $u_i$ and $v_i$. Define scores $s_{ij}=u_i^\top v_j/\tau$ with temperature $\tau>0$. The image-to-text loss is

$$\mathcal L_{I\to T}=-\frac1N\sum_i\log\frac{e^{s_{ii}}}{\sum_je^{s_{ij}}}.$$

A symmetric text-to-image loss swaps the roles, and CLIP averages the two. The diagonal pairs are positives; off-diagonal pairs act as negatives. The loss encourages a matched description to score above other descriptions in the batch.

**Worked example.** For one image with correct score 2 and one incorrect score 0, the correct probability is $e^2/(e^2+1)\approx0.881$, giving loss about 0.127. Lower temperature sharpens a fixed cosine-score gap, but it also changes gradients and sensitivity. Learned scaling requires stability controls.

False negatives are possible: two different captions in a batch may both describe the same kind of image. Contrastive training learns from the constructed pairing protocol, not a complete ontology of semantic equivalence.

The [CLIP paper](https://arxiv.org/abs/2103.00020) demonstrates transferring image representations through natural-language supervision. For zero-shot classification, encode textual class descriptions and choose the one most similar to the image. Prompt wording and class descriptions are part of the evaluation protocol.

## 3. Vision transformers turn images into patch sequences [12:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=720s)

For an image of height $H$, width $W$, and $C$ channels, divide it into nonoverlapping $p\times p$ patches. When dimensions divide evenly, the patch count is $N_p=HW/p^2$. Each flattened patch has $p^2C$ entries and is linearly projected into model width $d$.

Position embeddings tell the transformer where patches came from. Without positional information, a standard attention stack largely treats the patch set independently of its spatial ordering, apart from other architectural signals.

**Worked example.** A $336\times336$ RGB image with $14\times14$ patches produces $24^2=576$ patches, each with 588 raw scalar values. Doubling both image dimensions produces 2304 patches. Dense attention over patches then has sixteen times as many pairwise positions, even though image pixel count grows only fourfold.

The [Vision Transformer paper](https://arxiv.org/abs/2010.11929) supplies the patch-based formulation. Resolution is thus both an information choice and a resource choice. Downsampling may erase text or small objects; keeping all detail can be expensive.

## 4. SigLIP changes the coupling between pairs [22:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1320s)

Instead of normalizing one image's scores across all texts with softmax, a sigmoid loss treats each pair as a binary matching problem. Let $z_{ij}=1$ for a matched pair and $-1$ otherwise, and let score $a_{ij}=t\,u_i^\top v_j+b$ include scale and bias. A representative objective is

$$\mathcal L=-\frac1N\sum_{i,j}\log\sigma(z_{ij}a_{ij}).$$

Normalizing constants and negative-sampling details are implementation choices. The important structural difference is that each pair's logistic loss can be evaluated without a global softmax denominator over all texts. This enables different distributed blockwise strategies, including exchanging embeddings and accumulating pair losses.

**Worked example.** A negative pair with score 2 incurs loss $-\log\sigma(-2)\approx2.127$, strongly penalizing a confident false match. A negative score of -2 incurs about 0.127. The bias can account for the large imbalance between positive and negative pairs.

The [SigLIP paper](https://arxiv.org/abs/2303.15343) is the primary source. Removing the softmax denominator does not make negative examples irrelevant or guarantee quality independent of data and batch construction.

## 5. Visual instruction tuning connects an encoder to a language model [29:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1740s)

A common vision–language model has three parts: a vision encoder, a projector or adapter, and a pretrained language decoder. If visual features have width $d_v$ and the language model expects width $d_l$, a simple projector $W\in\mathbb R^{d_v\times d_l}$ maps each feature into the language embedding space.

The combined sequence contains visual embeddings and text embeddings. The decoder predicts an answer conditioned on both. The loss can remain ordinary next-token likelihood on answer text:

$$\mathcal L=-\sum_t\log p_\theta(y_t\mid I,x,y_{<t}).$$

The original [LLaVA work](https://arxiv.org/abs/2304.08485) provides an influential concrete example of connecting a vision encoder and a language model with visual instruction data. Different training stages freeze or update different components.

**Worked example.** Freezing both pretrained backbones while fitting only the projector limits trainable parameters and preserves existing representations. It also constrains what can be learned: if the encoder has discarded tiny text, no linear projector can reconstruct it from absent information. Later unfreezing can adapt features but costs more and can disturb existing capabilities.

## 6. High resolution and video expose the token-budget problem [36:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2160s)

Tiling an image into crops preserves local resolution while a downsampled global view provides context. Combining them creates more visual tokens and requires position or crop metadata. Pooling or token compression reduces cost but can discard details.

For video, sampling $F$ frames with $N_p$ tokens each gives roughly $FN_p$ visual tokens before compression. At 32 frames and 576 patches per frame, that is 18,432 tokens. Full dense interactions over the resulting sequence are substantial, and frame sampling can miss brief events.

The recording discusses increasingly specialized data and training pipelines for documents, charts, multiple images, and video. Many impressive examples depend on task-specific supervision. Generalization should be tested beyond the training task templates rather than inferred from a broad model label.

## 7. Multidimensional position and deeper fusion add structure [50:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3000s)

Text has a natural one-dimensional order. Images have height and width; video adds time. Multimodal RoPE variants allocate coordinate pairs or frequency bands to these axes. The construction must preserve the intended geometry and handle changes in resolution and frame spacing.

An explanatory formulation applies rotations based on coordinates $(t,h,w)$ to different embedding subspaces. Relative-position dot products then depend on the relevant coordinate differences. How frequencies are distributed among axes affects what positional scales each axis can represent.

Deeper fusion can inject features from multiple vision-encoder layers into the language model rather than providing only the final visual representation at the input. This can retain lower-level detail alongside semantic features. The recording's model-specific architectures should be consulted in their official reports for exact placement and training schedules; the general mechanism does not identify the architecture of closed models.

## 8. Generating images requires more than accepting image inputs [66:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3960s)

A VLM that consumes images and emits text is different from a model that generates images. One route to unified autoregressive generation quantizes image features into a learned discrete codebook. If encoder output is $z_e$ and codebook vectors are $e_1,\ldots,e_K$, nearest-code assignment chooses

$$k^*=\arg\min_k\|z_e-e_k\|_2^2.$$

A decoder reconstructs the image from selected codes. A multimodal sequence model can then predict text and image codes in a common discrete sequence. The [VQ-VAE paper](https://arxiv.org/abs/1711.00937) develops discrete representation learning; [Chameleon](https://arxiv.org/abs/2405.09818) studies mixed-modal generation.

Discretization introduces reconstruction loss and codebook-learning challenges. Image-token entropy and local structure differ from text. A shared vocabulary interface does not remove those differences, and training stability can require modality-aware choices. The lecture's comments about unpublished closed-model designs are speculation and are not asserted here as facts.

## 9. Researched extension — semantic sufficiency is task-dependent

An independent way to reason about visual compression is to ask whether representation $Z=f(I)$ preserves information needed for target $Y$. A representation adequate for broad classification may be inadequate for exact text extraction. Increasing resolution or retaining intermediate features changes the information available to the decoder, but no decoder can reliably recover arbitrary details that were destroyed upstream.

A useful experiment pairs tasks on the same images: coarse category, object count, small-text reading, and spatial relation. Vary patch count and compression while holding the language decoder fixed. The resulting curves show which tasks fail first as visual information is reduced. This is more informative than one aggregate “vision score.”

## 10. Exercises and worked solutions

**Exercise 1.** How many $16\times16$ patches are in a $512\times512$ image, and what happens to dense attention pair count at $1024\times1024$?

<details><summary>Solution</summary>

There are $32^2=1024$ patches initially and $64^2=4096$ at the larger resolution. Token count grows fourfold; pair count grows sixteenfold, before architectural changes or compression.

</details>

**Exercise 2.** Why can a visually fluent description still be poorly grounded?

<details><summary>Solution</summary>

The language model can generate plausible descriptions from textual priors even when the visual representation is ambiguous or incomplete. Test image-sensitive contrasts, small changes in visual evidence, and questions whose answer cannot be inferred reliably from common context alone.

</details>

**Exercise 3.** Contrast a continuous visual adapter with a discrete image tokenizer.

<details><summary>Solution</summary>

The adapter maps continuous encoder features into a language model's input space, often for text output. A discrete tokenizer maps image content to code indices that can be predicted and decoded back into images. Their losses, information bottlenecks, and generation roles differ.

</details>

## Primary sources

- [Recording](https://www.youtube.com/watch?v=26FtD08ZpOU) and [official lecture](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_17.py).
- [CLIP](https://arxiv.org/abs/2103.00020), [ViT](https://arxiv.org/abs/2010.11929), [SigLIP](https://arxiv.org/abs/2303.15343), and [LLaVA](https://arxiv.org/abs/2304.08485).
- [VQ-VAE](https://arxiv.org/abs/1711.00937) and [Chameleon](https://arxiv.org/abs/2405.09818): discrete visual representations and mixed-modal generation.
