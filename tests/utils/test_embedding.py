import numpy as np
import pytest

from simba.utils.embedding import (_create_dictionary, _get_token_weight,
                                   get_embedding_map, get_token_freq_map,
                                   load_embedding_matrix)

EMBED_PATH = "tests/fixtures/test_embed.txt"
FREQ_PATH = "tests/fixtures/test_freq.txt"


@pytest.mark.parametrize(
    'test_pairs',
    [
        ([['Hakuna', 'Matata']],
         (['Hakuna', 'Matata'], {'Hakuna': 0, 'Matata': 1})),
        ([], ([], {})),
        ([['Hakuna', 'Matata'],
          ['What', 'a', 'wonderful', 'phrase'],
          ['Hakuna', 'Matata']],
         (['Hakuna', 'Matata', 'What', 'a', 'wonderful', 'phrase'],
          {'Hakuna': 0, 'Matata': 1, 'What': 2, 'a': 3,
           'wonderful': 4, 'phrase': 5})),
    ]
)
def test_create_dictionary(test_pairs):
    input_, expected = test_pairs
    output_ = _create_dictionary(input_)
    assert expected == output_


def test_get_embedding_map():
    expected = ({'hakuna': np.linspace(0.1, 0.5, 5),
                 'matata': np.linspace(0.5, 0.1, 5)},
                5)
    seq = [["hakuna", "matata"], ["matata", "hakuna"]]
    output_ = get_embedding_map(EMBED_PATH, seq)
    assert str(expected) == str(output_)


def test_get_embedding_map_wrong_dim():
    seq = [["hakuna", "matata"], ["matata", "hakuna"]]
    with pytest.raises(AssertionError):
        get_embedding_map("tests/fixtures/test_embed_wrong_dim.txt", seq)


def test_get_embedding_map_OOV():
    expected = ({}, 5)
    seq = [["problem-free", "philosophy"]]
    output_ = get_embedding_map(EMBED_PATH, seq)
    assert str(expected) == str(output_)


def test_get_embedding_map_norm():
    expected = ({'hakuna': np.linspace(0.1/(0.55)**0.5, 0.5/(0.55)**0.5, 5),
                 'matata': np.linspace(0.5/(0.55)**0.5, 0.1/(0.55)**0.5, 5)},
                5)
    seq = [["hakuna", "matata"], ["matata", "hakuna"]]
    output_ = get_embedding_map(EMBED_PATH, seq, norm=True)
    assert str(expected) == str(output_)


def test_get_embedding_map_freq():
    expected = ({'hakuna': np.array([0.00013316, 0.00026631, 0.00039947,
                                     0.00053262, 0.00066578]),
                 'matata': np.array([0.00199203, 0.00159363, 0.00119522,
                                     0.00079681, 0.00039841])},
                5)
    seq = [["hakuna", "matata"], ["matata", "hakuna"]]
    output_ = get_embedding_map(EMBED_PATH, seq, path_to_counts=FREQ_PATH)
    assert str(expected) == str(output_)


def test_get_token_freq_map():
    expected = {'hakuna': 0.75, 'matata': 0.25}
    output_ = get_token_freq_map(FREQ_PATH)
    assert expected == output_


@pytest.mark.parametrize(
    'test_pairs',
    [('hakuna', 0.0013315579227696406),
     ('matata', 0.00398406374501992)]
)
def test_get_token_weight(test_pairs):
    input_, expected = test_pairs
    freq_map = get_token_freq_map(FREQ_PATH)
    output_ = _get_token_weight(input_, freq_map)
    assert expected == output_


@pytest.mark.parametrize(
    'test_pairs',
    [('problem-free', 1.0),
     ('philosophy', 1.0)]
)
def test_get_token_weight_OOV(test_pairs):
    input_, expected = test_pairs
    freq_map = get_token_freq_map(FREQ_PATH)
    output_ = _get_token_weight(input_, freq_map)
    assert expected == output_


def test_load_embedding_matrix(monkeypatch):
    def patch_get_path(embedding, EMB_MAP):
        return EMBED_PATH
    monkeypatch.setattr("simba.utils.embedding.get_path", patch_get_path)
    expected = np.array([np.linspace(0.1, 0.5, 5),
                         np.linspace(0.5, 0.1, 5)])
    output_ = load_embedding_matrix("test")
    assert str(expected) == str(output_)
