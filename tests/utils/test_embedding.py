import pytest
from simba.utils.embedding import _create_dictionary


@pytest.mark.parametrize(
    'test_pairs',
    [
        ([['Hakuna', 'Matata']], (['Hakuna', 'Matata'], {'Hakuna': 0, 'Matata': 1})),
        ([], ([], {})),
        ([['Hakuna', 'Matata'], ['What', 'a', 'wonderful', 'phrase'], ['Hakuna', 'Matata']],
         (['Hakuna', 'Matata', 'What', 'a', 'wonderful', 'phrase'], {'Hakuna': 0, 'Matata': 1, 'What': 2, 'a': 3, 'wonderful': 4, 'phrase': 5})),
    ]
)
def test_create_dictionary(test_pairs):
    input_, expected = test_pairs
    output_ = _create_dictionary(input_)
    assert expected == output_
