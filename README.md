# simba :lion:

Similarity measures from Babylon Health.

## Installation

```bash
$ pip install simba
```

You can also checkout this repository and install from the root folder:
```bash
$ pip install .
```

Many of the similarity measures in simba rely on pre-trained embeddings.
If you don't have your own encoding logic already, you can register your
embedding files to use them easily with simba, as long as they're in the
standard text format for word vectors (as described [here](https://fasttext.cc/docs/en/english-vectors.html)).
For example, if you want to use fastText vectors that you've saved to `/path/to/fasttext`,
you can just run
```bash
$ simba embs register --name fasttext --path /path/to/fasttext
```
and simba will recognise them under the name `fasttext`.

You can do something similar for frequencies files (like [these](https://github.com/PrincetonML/SIF/blob/master/auxiliary_data/enwiki_vocab_min200.txt)):
```bash
$ simba freqs register --name wiki --path /path/to/wiki/counts
```

## Usage
```python
from simba.similarities import dynamax_jaccard
from simba.core import embed

sentences = ('The king has returned', 'Change is good')

# Assuming you've registered fasttext embeddings as described above
x, y = embed([s.split() for s in sentences], embedding='fasttext')
sim = dynamax_jaccard(x, y)
```
There are more examples, including comparing different similarity metrics on a dataset
of pairs, in the `examples` directory.

## Similarity Measures

This library contains implementations of the following methods in `simba.similarities`.
Please consider citing the corresponding papers in your work if you find them useful.

| Method | Description | Paper |
| - | - | - |
| `avg_cosine` | Average vector compared with cosine similarity | - |
| `batch_avg_pca`  | Average vector with principal component removal | [1] |
| `fbow_jaccard_factory` | Factory method for general fuzzy bag-of-words given a universe matrix | [2] |
| `max_jaccard` | Max-pooled vectors compared with Jaccard coefficient | [2] |
| `dynamax_{jaccard, otsuka, dice}` | DynaMax using Jaccard, Otsuka-Ochiai, and Dice coefficients | [2] |
| `gaussian_correction_{tic, aic}` | Takeuchi and Akaike Information Criteria (TIC and AIC) for Gaussian likelihood | [3] |
| `spherical_gaussian_correction_{tic, aic}` | TIC and AIC for spherical Gaussian likelihood | [3] |
| `von_mises_correction_{tic, aic}` | TIC and AIC for von Mises Fisher likelihood | [3] |
| `avg_{pearson, spearman, kendall}` | Average vector compared with Pearson, Spearman, and Kendall correlation | [4] |
| `max_spearman` | Max-pooled vectors compared with Spearman correlation | [5] |
| `cka_factory` | Factory method for general Centered Kernel Alignment (CKA) | [5] |
| `cka_{linear, gaussian}`| CKA with linear and Gaussian kernels | [5] |
| `dcorr` | CKA with distance kernel (distance correlation) | [5] |

Papers:
1. [Arora et al., ICLR 2017. *A Simple but Tough-to-Beat Baseline for Sentence Embeddings*](https://openreview.net/forum?id=SyK00v5xx)
2. [Zhelezniak et al., ICLR 2019. *Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors*](https://openreview.net/forum?id=SkxXg2C5FX)
3. [Vargas et al., ICML 2019. *Model Comparison for Semantic Grouping*](http://proceedings.mlr.press/v97/vargas19a.html)
4. [Zhelezniak et al., NAACL-HLT 2019. *Correlation Coefficients and Semantic Textual Similarity*](https://www.aclweb.org/anthology/N19-1100/)
5. [Zhelezniak et al., EMNLP-IJCNLP 2019. *Correlations between Word Vector Sets*](https://arxiv.org/abs/1910.02902)

## Contact
* [April Shen](https://github.com/apriltuesday)
* [Sasho Savkov](https://github.com/savkov)
* [Vitalii Zhelezniak](https://github.com/ironvital)
