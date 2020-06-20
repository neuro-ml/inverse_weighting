import os
from os.path import join as jp
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.measure import label

from dpipe.io import load_json, save_json, load_pred
from dpipe.medim.metrics import dice_score, fraction
from iw.utils import get_pred, volume2diameter, np_sigmoid
from dpipe.commands import load_from_folder


def get_intersection_stat_dice_id(cc_mask, one_cc, pred=None, logit=None):
    """Returns max local dice and corresponding stat to this hit component.
    If ``pred`` is ``None``, ``cc_mask`` treated as ground truth and stat sets to be 1."""
    hit_components = np.unique(cc_mask[one_cc])
    hit_components = hit_components[hit_components != 0]

    hit_stats = dict(zip(['hit_max', 'hit_median', 'hit_q95', 'hit_logit'], [[], [], [], []]))
    hit_dice, hit_id = [], []

    for n in hit_components:
        cc_mask_hit_one = cc_mask == n
        hit_dice.append(dice_score(cc_mask_hit_one, one_cc))
        hit_id.append(n)

        hit_stats['hit_max'].append(1. if pred is None else np.max(pred[cc_mask_hit_one]))
        hit_stats['hit_median'].append(1. if pred is None else np.median(pred[cc_mask_hit_one]))
        hit_stats['hit_q95'].append(1. if pred is None else np.percentile(pred[cc_mask_hit_one], q=95))
        hit_stats['hit_logit'].append(np.inf if logit is None else np.max(logit[cc_mask_hit_one]))

    if len(hit_dice) == 0:
        return dict(zip(['hit_max', 'hit_median', 'hit_q95', 'hit_logit'], [0., 0., 0., -np.inf])), 0., None
    else:
        max_idx = np.argmax(hit_dice)
        hit_id = np.array(hit_id)[max_idx]
        hit_stats['hit_max'] = np.array(hit_stats['hit_max'])[max_idx]
        hit_stats['hit_median'] = np.array(hit_stats['hit_median'])[max_idx]
        hit_stats['hit_q95'] = np.array(hit_stats['hit_q95'])[max_idx]
        hit_stats['hit_logit'] = np.array(hit_stats['hit_logit'])[max_idx]
        return hit_stats, np.max(hit_dice), hit_id


def prc_records(segm, pred, logit):
    segm_split, segm_n_splits = label(get_pred(segm), return_num=True)
    pred_split, pred_n_splits = label(get_pred(pred), return_num=True)

    records = []

    for n in range(1, segm_n_splits + 1):
        record = {}
        segm_cc = segm_split == n

        record['obj'] = f'tum_{n}'
        record['is_tum'] = True
        record['diameter'] = volume2diameter(np.sum(segm_cc))
        stats, dice, hit_id = get_intersection_stat_dice_id(cc_mask=pred_split, one_cc=segm_cc,
                                                            pred=pred[0], logit=logit[0])
        record['hit_dice'] = dice
        record['hit_max'], record['hit_median'], record['hit_q95'], record['hit_logit'] = stats.values()
        record['hit_stat'] = record['hit_max']  # backward compatibility
        record['hit_obj'] = f'pred_{hit_id}'
        record['self_stat'] = 1.
        record['self_logit'] = np.inf

        records.append(record)

    for n in range(1, pred_n_splits + 1):
        record = {}
        pred_cc = pred_split == n

        record['obj'] = f'pred_{n}'
        record['is_tum'] = False
        record['diameter'] = volume2diameter(np.sum(pred_cc))
        stats, dice, hit_id = get_intersection_stat_dice_id(cc_mask=segm_split, one_cc=pred_cc)
        record['hit_dice'] = dice
        record['hit_max'], record['hit_median'], record['hit_q95'], record['hit_logit'] = stats.values()
        record['hit_stat'] = record['hit_max']  # backward compatibility
        record['hit_obj'] = f'tum_{hit_id}'
        record['self_stat'] = np.max(pred[0][pred_cc])
        record['self_logit'] = np.max(logit[0][pred_cc])

        records.append(record)

    return records


def exp2prc_df(exp_path, n_val=5, specific_ids=None):
    """Constructs pandas DataFrame with prc data from all predictions in ``exp_path``."""
    dfs = []
    for n in range(n_val):
        prc_path = jp(exp_path, f'experiment_{n}', 'test_metrics', 'prc_records.json')
        prc_dicts = load_json(prc_path)

        for _id in prc_dicts.keys():
            if specific_ids is None:
                [d.update({'id': _id}) for d in prc_dicts[_id]]
                dfs.append(pd.DataFrame.from_records(prc_dicts[_id]))
            else:
                if _id in specific_ids:
                    [d.update({'id': _id}) for d in prc_dicts[_id]]
                    dfs.append(pd.DataFrame.from_records(prc_dicts[_id]))

    df = pd.concat(dfs)
    return df


def get_size_df(df, size='small'):
    """Takes rows from DataFrame with specified lesion size"""
    if size == 'total':
        return df
    else:
        target_df = df[df['is_tum']]
        pred_df = df[~df['is_tum']]

        target_size_df = target_df[target_df['size'] == size]
        pred_size_df = pred_df[pred_df['size'] == size]

        size_df = pd.concat([target_size_df, pred_size_df])

        for index in target_size_df.index:
            _id, obj, hit_obj = target_size_df[['id', 'obj', 'hit_obj']].loc[index]

            if hit_obj:
                linked_predict = df[(df.id == _id) & (df.hit_obj == obj)]
                size_df = pd.concat([size_df, linked_predict])

        return size_df


def get_prc(df, thresholds=None, dice_th=0, hit_stat='hit_stat', self_stat='self_stat'):
    """Collects necessary data for building prc for experiments"""
    if thresholds is None:
        thresholds = np_sigmoid(np.linspace(0, 5, num=51))

    precision, recall, total_fp, avg_dice, std_dice = [], [], [], [], []

    for th in thresholds:
        conf_dict = {'tp': 0, 'fp': 0, 'fn': 0}

        th_df = df[df[self_stat] >= th]
        target_df = th_df[th_df['is_tum']]
        pred_df = th_df[~th_df['is_tum']]

        conf_dict['fp'] = len(pred_df[(pred_df['hit_dice'] <= dice_th) & (pred_df[self_stat] > th)])

        conf_dict['tp'] = len(target_df[(target_df['hit_dice'] > dice_th) & (target_df[hit_stat] >= th)])

        conf_dict['fn'] = len(target_df[(target_df['hit_dice'] <= dice_th) | (target_df[hit_stat] < th)])

        local_dices = target_df['hit_dice'][(target_df['hit_dice'] > dice_th) & (target_df[hit_stat] >= th)]

        precision.append(fraction(conf_dict['tp'], conf_dict['tp'] + conf_dict['fp']))
        recall.append(fraction(conf_dict['tp'], conf_dict['tp'] + conf_dict['fn']))
        total_fp.append(conf_dict['fp'])
        avg_dice.append(np.mean(local_dices))
        std_dice.append(np.std(local_dices))

    return {'precision': precision, 'recall': recall, 'totalFP': total_fp, 'avg_dice': avg_dice, 'std_dice': std_dice}


def evaluate_individual_metrics_with_prc(load_y_true, metrics: dict,
                                         predictions_path, logits_path, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        target = load_y_true(identifier)

        for metric_name, metric in metrics.items():
            if metric_name == 'prc_records':
                logit = load_pred(identifier, logits_path)
                results[metric_name][identifier] = metric(target, prediction, logit)
            else:
                results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
