import numpy as np

from simba.similarities import dynamax_jaccard, avg_cosine
from simba.evaluation import evaluate, evaluate_multiple
from simba.core import embed

EMBED_PATH_LARGE = "tests/fixtures/test_embed_large.txt"

SENTENCES1 = ["In the jungle the mighty jungle",
              "The lion sleeps tonight",
              "Hush my darling do not fear my darling",
              "The lion sleeps tonight"]

SENTENCES2 = ["Near the village the peaceful village",
              "The lion sleeps tonight",
              "My little darling",
              "Do not fear my little darling"]


def patch_get_path(embedding, EMB_MAP):
    return EMBED_PATH_LARGE


def test_end_to_end(monkeypatch):
    monkeypatch.setattr("simba.core.get_path", patch_get_path)

    sentences = ["In the jungle the mighty jungle", "The lion sleeps tonight"]
    x, y = embed([s.split() for s in sentences], embedding='test_large')
    assert np.isclose(dynamax_jaccard(x, y), 0.47254264, atol=1e-8)


def test_evaluate(monkeypatch):
    monkeypatch.setattr("simba.core.get_path", patch_get_path)
    embeddings1 = embed([s.split() for s in SENTENCES1],
                        embedding='test_large')
    embeddings2 = embed([s.split() for s in SENTENCES2],
                        embedding='test_large')
    expected, expected_cor = [[0.53112996, 1.0, 0.65611832, 0.40853999], None]
    output_, output_cor = evaluate(embeddings1, embeddings2, dynamax_jaccard)
    assert np.allclose(expected, output_)
    assert expected_cor == output_cor


def test_evaluate_gold(monkeypatch):
    monkeypatch.setattr("simba.core.get_path", patch_get_path)
    gold_labels = [3, 5, 4, 1]
    embeddings1 = embed([s.split() for s in SENTENCES1],
                        embedding='test_large')
    embeddings2 = embed([s.split() for s in SENTENCES2],
                        embedding='test_large')
    expected, expected_cor = [[0.53112996, 1.0, 0.65611832, 0.40853999],
                              0.91116447]
    output_, output_cor = evaluate(embeddings1, embeddings2, dynamax_jaccard,
                                   gold_scores=gold_labels)
    assert np.allclose(expected, output_)
    assert np.allclose(expected_cor, output_cor)


def test_evaluate_multiple(monkeypatch):
    monkeypatch.setattr("simba.core.get_path", patch_get_path)
    embeddings1 = embed([s.split() for s in SENTENCES1],
                        embedding='test_large')
    embeddings2 = embed([s.split() for s in SENTENCES2],
                        embedding='test_large')
    scores = evaluate_multiple(embeddings1, embeddings2,
                               [dynamax_jaccard, avg_cosine])
    exp1 = ([0.73374721, 1., 0.84151215, 0.66572127], None)
    exp2 = ([0.53112996, 1., 0.65611832, 0.40853999], None)
    assert np.allclose(scores['avg_cosine'][0], exp1[0])
    assert np.allclose(scores['dynamax_jaccard'][0], exp2[0])
