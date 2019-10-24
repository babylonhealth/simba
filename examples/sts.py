# Run STS tasks from SentEval (https://github.com/facebookresearch/SentEval).
# Note that in general we recommend reproducing paper results using our forks
# of the official SentEval repository, rather than this script.
import os

import numpy as np
from collections import defaultdict

from simba.evaluation import evaluate_multiple
from simba.similarities import (
    avg_cosine, dynamax_jaccard, max_jaccard,
    avg_spearman, avg_kendall, max_spearman,
    gaussian_correction_aic, von_mises_correction_aic,
    cka_linear, cka_gaussian, dcorr
)
from simba.core import embed


STS_DIR = '/Users/april.shen/DATA/STS'


def get_sts_data(sts_dir=STS_DIR):
    print('Loading data...')
    task_dict = {}

    for sts_task in os.listdir(sts_dir):
        task_dir = os.path.join(sts_dir, sts_task)
        if 'Benchmark' in sts_task or not os.path.isdir(task_dir):
            continue
        print(sts_task)
        task_dict[sts_task] = {
            'gold_scores': [],
            'sentences1': [],
            'sentences2': [],
        }
        gs = []
        sents1 = []
        sents2 = []
        for filename in sorted(os.listdir(task_dir)):
            if (
                '.gs.' in filename and '.ALL.' not in filename
                and '.SMT.' not in filename  # not publicly available
            ):
                with open(os.path.join(task_dir, filename), 'r') as f:
                    gs.append([x.strip() for x in f.readlines()])
            if 'input' in filename and 'txt' in filename:
                subtask1 = []
                subtask2 = []
                with open(os.path.join(task_dir, filename), 'r') as f:
                    for x in f.readlines():
                        if '\t' not in x:
                            continue
                        s1, s2 = x.strip().split('\t')
                        subtask1.append(s1.split())
                        subtask2.append(s2.split())
                sents1.append(subtask1)
                sents2.append(subtask2)

        for subtask_gs, subtask1, subtask2 in zip(gs, sents1, sents2):
            subtask_gs = np.array(subtask_gs)
            subtask1 = np.array(subtask1)
            subtask2 = np.array(subtask2)

            # Because STS16 is special
            not_empty_idx = subtask_gs != ''
            subtask1 = subtask1[not_empty_idx]
            subtask2 = subtask2[not_empty_idx]
            subtask_gs = [float(x) for x in subtask_gs[not_empty_idx]]

            assert len(subtask1) == len(subtask2) == len(subtask_gs)
            task_dict[sts_task]['gold_scores'].append(subtask_gs)
            task_dict[sts_task]['sentences1'].append(subtask1)
            task_dict[sts_task]['sentences2'].append(subtask2)
    return task_dict


def get_results(methods, task_dict):
    for task in sorted(task_dict):
        print(task)
        subtask_gs = task_dict[task]['gold_scores']
        subtask1 = task_dict[task]['sentences1']
        subtask2 = task_dict[task]['sentences2']
        results = defaultdict(list)

        for sents1, sents2, gs in zip(subtask1, subtask2, subtask_gs):
            # Get word embeddings.
            if von_mises_correction_aic in methods:
                embeddings1 = embed(sents1, embedding='fasttext',
                                    norm=True, pad_token='.')
                embeddings2 = embed(sents2, embedding='fasttext',
                                    norm=True, pad_token='.')
            else:
                embeddings1 = embed(sents1, embedding='fasttext')
                embeddings2 = embed(sents2, embedding='fasttext')

            # Get results for all methods on this subtask.
            all_scores = evaluate_multiple(
                embeddings1,
                embeddings2,
                methods,
                gs,
            )
            for method, result in all_scores.items():
                results[method].append(result[0])

        # Print (unweighted) mean Pearson score for each method.
        for method in results:
            print(f'{method}: {np.mean(results[method])}')
        print()


if __name__ == '__main__':
    task_dict = get_sts_data()
    papers = {
        'ICLR 2019': [avg_cosine, dynamax_jaccard, max_jaccard],
        'ICML 2019': [gaussian_correction_aic, von_mises_correction_aic],
        'NAACL 2019': [avg_spearman, avg_kendall],
        'EMNLP 2019': [max_spearman, cka_linear, cka_gaussian, dcorr],
    }
    for p in papers:
        print(f'\n===== {p} =====\n')
        get_results(papers[p], task_dict)
