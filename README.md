# simba :lion:

Similarity measures from Babylon Health.

## Installation

```bash
$ pip install simba
```

You can also checkout this repository and install from the root folder
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

sentences = ('she likes cats', 'he loves dogs')

# Assuming you've registered fasttext embeddings as described above
x, y = embed(sentences, embedding='fasttext')
sim = dynamax_jaccard(x, y)
```
There are more examples, including comparing different similarity metrics on a dataset
of pairs, in the `examples` directory.

## Contact
* April Shen
* Vitalii Zhelezniak
* Sasho Savokov
