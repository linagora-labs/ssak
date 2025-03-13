#!/usr/bin/env python3

import csv
import os
import re

import jiwer
import numpy as np
import tqdm

DEFAULT_INTERVAL_TYPE = "errorbar"  # "boxplot"
# LABELS_INS_DEL_SUBS = ("Insertion", "Deletion", "Substitution")
LABELS_INS_DEL_SUBS = ("Ins.", "Del.", "Subs.")


def normalize_line(line):
    return re.sub(r"\s+", " ", line).strip()


def parse_text_with_ids(file_name):
    with open(file_name, encoding="utf-8") as f:
        res_dict = {}
        for line in f:
            line = normalize_line(line).split(maxsplit=1)
            id = line[0]
            text = line[1] if len(line) > 1 else ""
            if id in res_dict and res_dict[id] != text:
                raise ValueError(f"Id {id} is not unique in {file_name}")
            res_dict[id] = text
    return res_dict


def parse_text_without_ids(file_name):
    return dict(enumerate([normalize_line(l) for l in open(file_name, encoding="utf-8").readlines()]))


def compute_wer(
    refs,
    preds,
    use_ids=False,
    normalization=None,
    character_level=False,
    use_percents=False,
    alignment=False,
    include_correct_in_alignement=False,
    words_list=None,
    words_blacklist=None,
    replacements_ref=None,
    replacements_pred=None,
    details_words_list=False,
    bootstrapping=False,
):
    """
    Compute WER between two files.
    :param refs: path to the reference file, or dictionary {"id": "text..."}, or list of texts
    :param preds: path to the prediction file, or dictionary {"id": "text..."}, or list of texts.
                  Must be of the same type as refs.
    :param use_ids: (for files) whether reference and prediction files includes id as a first field
    :param normalization: None or a language code ("fr", "ar", ...).
        Use suffix '+' (ex: 'fr+', 'ar+', ...) to remove all non-alpha-num characters (apostrophes, dashes, ...)
    :param alignment: if True, print alignment information. If string, write alignment information to the file.
    :param include_correct_in_alignement: whether to include correct words in the alignment
    :param words_list: list of words to focus on
    :param words_blacklist: list of words to exclude all the examples where the reference include such a word
    :param replacements_ref: dictionary of replacements to perform in the reference
    :param replacements_pred: dictionary of replacements to perform in the hypothesis
    :param details_words_list: whether to output information about words that are well recognized (among the specified words_list)
    :param bootstrapping: whether to compute bootstrapping intervals (might be slow)
    """
    # Open the test dataset human translation file
    if isinstance(refs, str):
        assert os.path.isfile(refs), f"Reference file {refs} doesn't exist"
        assert isinstance(preds, str) and os.path.isfile(preds), f"Prediction file {preds} doesn't exist"
        if use_ids:
            refs = parse_text_with_ids(refs)
            preds = parse_text_with_ids(preds)
        else:
            refs = parse_text_without_ids(refs)
            preds = parse_text_without_ids(preds)

    if isinstance(refs, dict):
        assert isinstance(preds, dict)

        # Reconstruct two lists of pred/ref with the intersection of ids
        ids = [id for id in refs.keys() if id in preds]

        if len(ids) == 0:
            if len(refs) == 0:
                raise ValueError("Reference file is empty")
            if len(preds) == 0:
                raise ValueError("Prediction file is empty")
            raise ValueError("No common ids between reference and prediction files")
        if len(ids) != len(refs) or len(ids) != len(preds):
            print("WARNING: ids in reference and/or prediction files are missing or different.")

        refs = [refs[id] for id in ids]
        preds = [preds[id] for id in ids]

    assert isinstance(refs, list)
    assert isinstance(preds, list)
    assert len(refs) == len(preds)

    if words_blacklist:
        # Remove examples where the reference includes a word from the blacklist
        for i in range(len(refs) - 1, -1, -1):
            for w in words_blacklist:
                if re.search(r"\b" + w + r"\b", refs[i]):
                    del refs[i]
                    del preds[i]
                    break

    # Replacements BEFORE normalization
    if replacements_ref:
        for k, v in replacements_ref.items():
            for i, ref in enumerate(refs):
                if k in ref:
                    refs[i] = re.sub(r"\b" + k + r"\b", v, refs[i])
            if words_list:
                for i, w in enumerate(words_list):
                    if k in w:
                        words_list[i] = re.sub(r"\b" + k + r"\b", v, w)
    if replacements_pred:
        for k, v in replacements_pred.items():
            for i, pred in enumerate(preds):
                if k in pred:
                    preds[i] = re.sub(r"\b" + k + r"\b", v, preds[i])

    if normalization:
        from ssak.utils.text import format_text

        strong_normalization = normalization.endswith("+")
        if strong_normalization:
            normalization = normalization[:-1]
        very_strong_normalization = normalization.endswith("+")
        if very_strong_normalization:
            normalization = normalization[:-1]

        normalize_funcs = []
        if normalization.startswith("ar"):
            kwargs = {
                "keep_latin_chars": True,
                "normalize_dialect_words": True if normalization.endswith("tn") else False,
            }
            normalize_funcs.append(lambda x: format_text(x, language=normalization, **kwargs))
        else:
            normalize_funcs.append(lambda x: format_text(x, language=normalization))

        if normalization == "fr":

            def further_normalize(s):
                # Fix masculine / feminine for un ("1 fois" / "une fois" -> "un fois")
                return re.sub(r"\bune?\b", "1", s)

            normalize_funcs.append(further_normalize)

        if strong_normalization:
            from ssak.utils.text_basic import collapse_whitespace
            def remove_not_words(s):
                # Remove any character that is not alpha-numeric (e.g. apostrophes, dashes, ...)
                return collapse_whitespace(re.sub(r"[^\w]", " ", s))

            normalize_funcs.append(remove_not_words)

        if very_strong_normalization:

            def remove_ending_s(s):
                # Remove "s" at the end of words, like "les" -> "le"
                return re.sub(r"(\w)s\b", r"\1", s)

            normalize_funcs.append(remove_ending_s)

        def normalize_func(s):
            for f in normalize_funcs:
                s = f(s)
            return s

        refs = [normalize_func(ref) for ref in tqdm.tqdm(refs, desc="Normalizing references", leave=False)]
        preds = [normalize_func(pred) for pred in tqdm.tqdm(preds, desc="Normalizing predictions", leave=False)]
        if words_list:
            words_list = [normalize_func(w) for w in tqdm.tqdm(words_list, desc="Normalizing special words", leave=False)]
            words_list = [w for w in words_list if w]

        # Replacements AFTER normalization
        if replacements_ref:
            replacements_ref = {normalize_func(k): normalize_func(v) for k, v in replacements_ref.items()}
        if replacements_pred:
            replacements_pred = {normalize_func(k): normalize_func(v) for k, v in replacements_pred.items()}
        if replacements_ref:
            for k, v in replacements_ref.items():
                for i, ref in enumerate(refs):
                    if k in ref:
                        refs[i] = re.sub(r"\b" + k + r"\b", v, refs[i])
                if words_list:
                    for i, w in enumerate(words_list):
                        if k in w:
                            words_list[i] = re.sub(r"\b" + k + r"\b", v, w)
        if replacements_pred:
            for k, v in replacements_pred.items():
                for i, pred in enumerate(preds):
                    if k in pred:
                        preds[i] = re.sub(r"\b" + k + r"\b", v, preds[i])

    refs, preds, hits_bias = ensure_not_empty_reference(refs, preds, character_level)

    # Calculate WER for the whole corpus
    kwargs = {}
    if character_level:
        cer_transform = jiwer.transforms.Compose(
            [
                jiwer.transforms.RemoveMultipleSpaces(),
                jiwer.transforms.Strip(),
                jiwer.transforms.ReduceToSingleSentence(""),
                jiwer.transforms.ReduceToListOfListOfChars(),
            ]
        )
        kwargs = dict(
            truth_transform=cer_transform,
            hypothesis_transform=cer_transform,
        )
    if bootstrapping:
        measures_list = [jiwer.compute_measures([r], [p], **kwargs) for r, p in zip(refs, preds)]
        measures = {}
        for i, meas in enumerate(measures_list):
            for k, v in meas.items():
                if k not in measures:
                    measures[k] = []
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                measures[k].append(v)

        # Compute confidence intervals
        for stat in "substitutions", "deletions", "hits", "insertions":
            measures[stat + "_list"] = measures.pop(stat)
            measures[stat] = np.sum(measures[stat + "_list"])

    else:
        measures = jiwer.compute_measures(refs, preds, **kwargs)

    extra = {}
    if alignment:
        with open(alignment, "w+", encoding="utf-8") if isinstance(alignment, str) else open("/dev/stdout", "w") as f:
            output = jiwer.process_words(refs, preds, reference_transform=cer_transform, hypothesis_transform=cer_transform) if character_level else jiwer.process_words(refs, preds)
            s = jiwer.visualize_alignment(output, show_measures=True, skip_correct=not include_correct_in_alignement)
            f.write(s)
            extra = {
                "alignment": s,
                "raw_alignement": output,
            }

    if words_list:
        TP_list = []
        FP_list = []
        FN_list = []
        if details_words_list:
            detailed_tp = {w: 0 for w in words_list}
            detailed_fp = {w: 0 for w in words_list}
            detailed_fn = {w: 0 for w in words_list}
            detailed_total = {w: 0 for w in words_list}
        for r, p in zip(refs, preds):
            TP_list.append(0)
            FP_list.append(0)
            FN_list.append(0)
            for w in words_list:
                if w in r:
                    num_in_ref = len(re.findall(r"\b" + w + r"\b", r))
                else:
                    num_in_ref = 0
                if w in p:
                    num_in_pred = len(re.findall(r"\b" + w + r"\b", p))
                else:
                    num_in_pred = 0
                tp = min(num_in_ref, num_in_pred)
                fp = max(0, num_in_pred - num_in_ref)
                fn = max(0, num_in_ref - num_in_pred)
                TP_list[-1] += tp
                FP_list[-1] += fp
                FN_list[-1] += fn
                if details_words_list:
                    detailed_total[w] += num_in_ref
                    detailed_tp[w] += tp
                    detailed_fp[w] += fp
                    detailed_fn[w] += fn
        TP = sum(TP_list)
        FP = sum(FP_list)
        FN = sum(FN_list)
        extra.update(
            {
                "FP": FP,
                "FN": FN,
                "TP": TP,
            }
        )
        extra.update(aggregate_f1_recall_precision(extra))
        if bootstrapping:
            extra.update(
                {
                    "TP_list": TP_list,
                    "FP_list": FP_list,
                    "FN_list": FN_list,
                }
            )
        if details_words_list:
            words_list_recall = {w: detailed_tp[w] / (detailed_tp[w] + detailed_fp[w]) if (detailed_tp[w] + detailed_fp[w]) > 0 else 0 for w in words_list}
            words_list_precision = {w: detailed_tp[w] / (detailed_tp[w] + detailed_fn[w]) if (detailed_tp[w] + detailed_fn[w]) > 0 else 0 for w in words_list}
            words_list_F1 = {w: 2 * words_list_precision[w] * words_list_recall[w] / (words_list_precision[w] + words_list_recall[w]) if (words_list_precision[w] + words_list_recall[w]) > 0 else 0 for w in words_list}
            with open(details_words_list, "w+") if isinstance(details_words_list, str) else open("/dev/stdout", "w") as f:
                csv_writer = csv.writer(f, delimiter=",")
                max_length_words = max([len(w) for w in words_list])
                csv_writer.writerow(
                    [
                        "Word" + " " * (max_length_words - 4),
                        "F1%    ",
                        "Recall%",
                        "Precision%",
                        "Total",
                        "TP   ",
                        "FN   ",
                        "FP   ",
                    ]
                )
                for w in sorted(detailed_total.keys(), key=lambda w: (-detailed_total[w], w), reverse=False):
                    if not detailed_total[w] and not detailed_fp[w]:
                        # Ignore words that are not involved at all
                        continue
                    csv_writer.writerow(
                        [
                            f"{w: <{max_length_words}}",
                            f"{round(words_list_F1[w]*100, 1):<7}",
                            f"{round(words_list_recall[w]*100, 1):<7}",
                            f"{round(words_list_precision[w]*100, 1):<10}",
                            f"{detailed_total[w]:<5}",
                            f"{detailed_tp[w]:<5}",
                            f"{detailed_fn[w]:<5}",
                            f"{detailed_fp[w]:<5}",
                        ]
                    )

    scale = 100 if use_percents else 1

    sub_score = measures["substitutions"]
    del_score = measures["deletions"]
    hits_score = measures["hits"]
    ins_score = measures["insertions"]

    if bootstrapping:
        for stat in "substitutions", "deletions", "hits", "insertions":
            vals = measures[stat + "_list"]
            if scale != 1:
                vals = [float(v) * scale for v in vals]
            key = {
                "deletions": "del",
                "insertions": "ins",
                "substitutions": "sub",
            }.get(stat, stat)
            extra[key + "_list"] = vals

    hits_score -= hits_bias
    count = hits_score + del_score + sub_score

    assert "del" not in extra

    if count == 0:  # This can happen if all references are empty
        return {
            "wer": scale if ins_score else 0,
            "del": 0,
            "ins": scale if ins_score else 0,
            "sub": 0,
            "hits": 0,
            "count": 0,
        } | extra

    res = {
        "del": (float(del_score) * scale / count),
        "ins": (float(ins_score) * scale / count),
        "sub": (float(sub_score) * scale / count),
        "hits": (float(hits_score) * scale / count),
        "count": count,
    } | extra

    res.update(aggregate_wer(res, scale=scale, count=count))

    if bootstrapping:
        intervals = list_to_confidence_intervals(res)
        res.update(intervals)

    return res


def compute_wer_differences(refs, preds1, preds2, **kwargs):
    """
    Compute the difference of WER between two predictions and the reference.
    """
    kwargs["alignment"] = True
    out1 = compute_wer(refs, preds1, **kwargs)
    out2 = compute_wer(refs, preds2, **kwargs)

    def collect_errors(out):
        data = out["raw_alignement"]
        alignments = data.alignments
        hypotheses = data.hypotheses
        references = data.references
        dels = []
        ins = []
        subs = []
        for alignement, hyp, ref in zip(alignments, hypotheses, references):
            dels.append(set())
            ins.append([])
            subs.append(set())
            for chunk in alignement:
                if chunk.type == "insert":
                    # An insertion is characterized by the inserted word(s)
                    for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                        ins[-1].append(hyp[chunk.hyp_start_idx])
                elif chunk.type == "delete":
                    # An deletion is characterized by the indices of the deleted word(s) -> unique
                    for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                        dels[-1].add(i)
                elif chunk.type == "substitute":
                    # A substitution is characterized by the indices of the substituted word(s) -> unique
                    len_hyp = chunk.hyp_end_idx - chunk.hyp_start_idx
                    len_ref = chunk.ref_end_idx - chunk.ref_start_idx
                    assert len_hyp == len_ref
                    for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                        subs[-1].add(i)
                else:
                    assert chunk.type == "equal"
        return dels, ins, subs

    dels1, ins1, subs1 = collect_errors(out1)
    dels2, ins2, subs2 = collect_errors(out2)

    def count_differences(set1, set2):
        if isinstance(set1, list):
            # This is a quick approximation
            # (the correct implementation would be to removed common elements one by one and count the rest)
            return count_differences(set(set1), set(set2))
        elif isinstance(set1, set):
            return {
                "removed": len(set1 - set2),
                "added": len(set2 - set1),
            }
        else:
            raise ValueError(f"Invalid type {type(set1)}")

    def count_differences_batch(list1, list2):
        assert len(list1) == len(list2)
        res = {}
        for s1, s2 in zip(list1, list2):
            diff = count_differences(s1, s2)
            for k, v in diff.items():
                if k not in res:
                    res[k] = 0
                res[k] += v
        return res

    count = out1["count"]
    assert count == out2["count"]

    res = {}
    for what, o1, o2 in [
        ("del", dels1, dels2),
        ("ins", ins1, ins2),
        ("sub", subs1, subs2),
    ]:
        diffs = count_differences_batch(o1, o2)
        for k, v in diffs.items():
            res[what + "_" + k] = v / count
    return res


def ensure_not_empty_reference(refs, preds, character_level):
    """
    This is a workaround to avoid error from jiwer.compute_measures when the reference is empty.
        ValueError: one or more groundtruths are empty strings
        ValueError: truth should be a list of list of strings after transform which are non-empty
    """
    refs_stripped = [r.strip() for r in refs]
    hits_bias = 0
    while "" in refs_stripped:
        hits_bias += 1
        i = refs_stripped.index("")
        refs_stripped[i] = refs[i] = "A"
        if character_level:
            preds[i] = "A" + preds[i]
        else:
            preds[i] = "A " + preds[i]
    return refs, preds, hits_bias


def str2bool(string):
    str2val = {"true": True, "false": False}
    string = string.lower()
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError("Expected True or False")


def list_to_confidence_intervals(measures, n_bootstraps=10000, max_samples=1000):
    keys_to_sum = [k for k in measures.keys() if k.endswith("_list")]
    assert len(keys_to_sum)
    n = None
    for k in keys_to_sum:
        if n is None:
            n = len(measures[k])
        assert n == len(measures[k]), f"Length mismatch for {k} : {n} != {len(measures[k])}"
    n_samples = min(max_samples, n)

    # bootstrap
    samples = []
    np.random.seed(51)
    for _ in tqdm.tqdm(range(n_bootstraps), desc="Bootstrapping (sampling)"):
        indices = np.random.choice(n, n_samples)
        sample = {k: [measures[k][i] for i in indices] for k in keys_to_sum}
        sample = {k[:-5]: np.sum(v) for k, v in sample.items()}
        sample.update(aggregate_wer(sample, norm_rates=True))
        sample.update(aggregate_f1_recall_precision(sample))
        samples.append(sample)

    assert len(samples)
    keys = samples[0].keys()
    intervals = {}
    for k in keys:
        if k not in ["wer", "F1", "recall", "precision", "del", "ins", "sub", "hits"]:
            continue
        vals = [s[k] for s in samples]
        intervals[k + "_stdev"] = np.std(vals)
        intervals[k + "_median"] = np.median(vals)
        intervals[k + "_mean"] = np.mean(vals)
        intervals[k + "_low"] = np.percentile(vals, 5)
        intervals[k + "_high"] = np.percentile(vals, 95)
        intervals[k + "_samples"] = vals

    return intervals


def aggregate_f1_recall_precision(measures):
    TP = measures["TP"]
    FN = measures["FN"]
    FP = measures["FP"]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "F1": F1,
    }


def aggregate_wer(measures, scale=1, count=None, norm_rates=False):
    if count is None:
        count = measures.get("count")
    c_scale = count if count else 1
    del_count = measures["del"] * c_scale / scale
    if abs(round(del_count) - del_count) < 0.0001:  # avoid problems with float operations like having 1.9999999997
        del_count = round(del_count)
    ins_count = measures["ins"] * c_scale / scale
    if abs(round(ins_count) - ins_count) < 0.0001:
        ins_count = round(ins_count)
    hits_count = measures["hits"] * c_scale / scale
    if abs(round(hits_count) - hits_count) < 0.0001:
        hits_count = round(hits_count)
    sub_count = measures["sub"] * c_scale / scale
    if abs(round(sub_count) - sub_count) < 0.0001:
        sub_count = round(sub_count)
    if count is None:
        count = hits_count + del_count + sub_count
    assert del_count % 1 == 0, f"{del_count=} ({count=}, {measures['count']=}, {measures['del']=}, {measures['hits']=}, {measures['ins']=}, {measures['sub']=} {scale=})"
    assert ins_count % 1 == 0, f"{ins_count=} ({count=}, {measures['count']=}, {measures['del']=}, {measures['hits']=}, {measures['ins']=}, {measures['sub']=} {scale=})"
    assert sub_count % 1 == 0, f"{sub_count=} ({count=}, {measures['count']=}, {measures['del']=}, {measures['hits']=}, {measures['ins']=}, {measures['sub']=} {scale=})"
    wer = float(del_count + ins_count + sub_count) / count
    if "wer" in measures:
        assert abs(measures["wer"] - wer) < 0.0001
    res = {"count": count, "wer": wer * scale}
    if norm_rates:
        res = res | {
            "del": float(del_count) / count,
            "ins": float(ins_count) / count,
            "hits": float(hits_count) / count,
            "sub": float(sub_count) / count,
        }
    return res


def plot_wer(
    wer_dict,
    label=True,
    legend=True,
    show=True,
    sort_best=-1,
    small_hatch=False,
    title=None,
    label_rotation=15,
    label_fontdict={"weight": "bold", "size": 12},
    ymin=0,
    ymax=None,
    interval_type=DEFAULT_INTERVAL_TYPE,
    show_axisnames=True,
    x_axisname=None,
    colors=None,
    use_colors=None,
    legend_hatches=True,
    scale=100,
    add_percents_in_ticks=True,
    **kwargs,
):
    """
    Plot WER statistics.
    :param wer_dict: dictionary of results, or a list of results, or a dictionary of results,
        where a result is a dictionary as returned by compute_wer, or a list of such dictionaries
    :param label: whether to add a label to the bars (as xticks)
    :param legend: whether to add a legend (Deletion/Substition/Insertion)
    :param show: whether to show the plot (if True) or save it to the given file name (if string)
    :param sort_best: whether to sort the results by best WER
    :param small_hatch: whether to use small hatches for the bars
    :param colors: list of colors
    :param legend_hatches: True, False, "before", "after"
    :param interval_type: "none", "boxplot", "violinplot" or "errorbar"
    :param **kwargs: additional arguments to pass to matplotlib.pyplot.bar
    """
    import matplotlib.pyplot as plt

    if colors is None:
        # rainbow colors
        num_colors = 8
        colors = [plt.cm.gist_rainbow(i / num_colors) for i in range(num_colors)]
        colors = colors[::2] + colors[1::2]
        colors = colors[3:] + colors[:3]
    if isinstance(wer_dict, list) and min([check_result(w) for w in wer_dict]):
        wer_dict = dict(enumerate(wer_dict))
    elif check_result(wer_dict):
        wer_dict = {"Evaluation": wer_dict}
    elif isinstance(wer_dict, dict) and min([check_result(w) for w in wer_dict.values()]):
        pass
    else:
        raise ValueError(
            "Invalid input (expecting a dictionary of results, a list of results, or a dictionary of results, \
where a result is a dictionary as returned by compute_wer, or a list of such dictionaries)"
        )

    if use_colors is None:
        use_colors = len(wer_dict) > 1

    plt.cla()

    kwargs.update(width=0.8, edgecolor="black")
    kwargs_ins = kwargs.copy()
    kwargs_del = kwargs.copy()
    kwargs_sub = kwargs.copy()
    if "color" not in kwargs:
        if not use_colors:
            kwargs_ins["color"] = "gold"
            kwargs_del["color"] = "white"
            kwargs_sub["color"] = "orangered"
    n = 2 if small_hatch else 1
    kwargs_ins["hatch"] = "*" * n
    kwargs_del["hatch"] = "O" * n
    kwargs_sub["hatch"] = "x" * n

    keys = list(wer_dict.keys())
    if sort_best:
        keys = sorted(keys, key=lambda k: get_stat_average(wer_dict[k]), reverse=sort_best < 0)
    positions = range(len(keys))
    D = [get_stat_average(wer_dict[k], "del") * scale for k in keys]
    I = [get_stat_average(wer_dict[k], "ins") * scale for k in keys]
    S = [get_stat_average(wer_dict[k], "sub") * scale for k in keys]
    W = [get_stat_average(wer_dict[k], "wer") * scale for k in keys]

    all_vals = None
    compute_intervals = max([len(get_stat_list(v, "wer") if "wer_samples" not in v else v["wer_samples"]) for v in wer_dict.values()]) > 1
    if compute_intervals:
        all_vals = []
        for k in keys:
            val_list = []
            if "wer_samples" in wer_dict[k]:
                for l in get_stat_list(wer_dict[k], "wer_samples"):
                    val_list.extend(l)
            else:
                val_list.extend(get_stat_list(wer_dict[k], "wer"))
            all_vals.append([v * scale for v in val_list])

    def do_legend_hatches():
        add_opts_legend = add_opts | {"color": "white"}
        if not small_hatch:
            kwargs_ins_legend = kwargs_ins | {"hatch": kwargs_ins["hatch"] * 2}
            kwargs_del_legend = kwargs_del | {"hatch": kwargs_del["hatch"] * 2}
            kwargs_sub_legend = kwargs_sub | {"hatch": kwargs_sub["hatch"] * 2}
        plt.bar([pos], [0], bottom=[d + s], label=LABELS_INS_DEL_SUBS[0], **kwargs_ins_legend, **add_opts_legend)
        plt.bar([pos], [0], bottom=[s], label=LABELS_INS_DEL_SUBS[1], **kwargs_del_legend, **add_opts_legend)
        plt.bar([pos], [0], label=LABELS_INS_DEL_SUBS[2], **kwargs_sub_legend, **add_opts_legend)

    for i_x, (pos, d, i, s, w) in enumerate(zip(positions, D, I, S, W)):
        assert abs(w - (d + i + s)) < 0.0001, f"{w=} != {d + i + s} = {d=} + {i=} + {s=}"
        complete_label = label and i_x == 0
        add_opts = {}
        label_ins = label_del = label_sub = None
        if complete_label:
            (label_ins, label_del, label_sub) = LABELS_INS_DEL_SUBS
        if use_colors:
            add_opts["color"] = colors[i_x % len(colors)]
            add_opts["alpha"] = 0.5
            if complete_label and legend_hatches == "before":
                do_legend_hatches()
            if label:
                system_label = keys[i_x]
                if system_label in [None, ""]:
                    system_label = "_"
                kwargs_ins_color = kwargs_ins.copy()
                kwargs_ins_color.pop("hatch")
                plt.bar([pos], [0], bottom=[d + s], label=system_label, **kwargs_ins_color, **add_opts)
                label_ins = label_del = label_sub = None
        plt.bar([pos], [i], bottom=[d + s], label=label_ins, **kwargs_ins, **add_opts)
        plt.bar([pos], [d], bottom=[s], label=label_del, **kwargs_del, **add_opts)
        plt.bar([pos], [s], label=label_sub, **kwargs_sub, **add_opts)

    if use_colors and legend_hatches and legend_hatches != "before":
        add_opts["color"] = "white"
        kwargs_ins_color["hatch"] = ""
        kwargs_ins_color["edgecolor"] = "white"
        # Add empty label to have Ins/Del/Sub alone in the last column
        # - at least 3 elements in the first column
        if len(keys) <= 2:
            for n in range(3 - len(keys)):
                plt.bar([pos], [0], bottom=[d + s], label=" ", **kwargs_ins_color, **add_opts)
        do_legend_hatches()
        # - same elements in the second column
        for n in range(len(keys) - 3):
            plt.bar([pos], [0], bottom=[d + s], label=" ", **kwargs_ins_color, **add_opts)

    if all_vals and all_vals[0] and len(all_vals[0]) > 1:
        if interval_type == "violinplot":
            if use_colors:
                for i_x in range(len(all_vals)):
                    plot_violinplot([all_vals[i_x]], positions=[positions[i_x]], color=colors[i_x % len(colors)], alpha=0.5)
            else:
                plot_violinplot(all_vals, positions=positions, alpha=0.5)
        elif interval_type == "boxplot":
            plt.boxplot(all_vals, positions=positions, whis=100)
        elif interval_type == "errorbar":
            lows, medians, highs = find_interval_around_median(all_vals)
            plt.errorbar(
                positions,
                medians,
                yerr=[medians - lows, highs - medians],
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1,
                capsize=2,
            )
        elif interval_type in ["none", None]:
            pass
        else:
            raise ValueError(f"Invalid interval_type {interval_type}")

    if False:  # not use_colors:
        plt.xticks(range(len(keys)), keys, rotation=label_rotation, fontdict=label_fontdict, ha="right")
        func_ylabel = plt.ylabel
    else:
        # Remove xticks
        plt.xticks([])

        def func_ylabel(title, *args, **kwargs):
            middle = (len(positions) - 1) / 2
            plt.xticks([middle], [title], rotation=0, fontdict=label_fontdict, ha="center")
            plt.tick_params(axis="x", length=0)

    label_size = label_fontdict.get("size")
    label_weight = label_fontdict.get("weight")
    plt.yticks(fontsize=label_size)
    if ymax is None:
        _, maxi = plt.ylim()
        plt.ylim(bottom=ymin, top=min(100, maxi))
    else:
        plt.ylim(bottom=ymin, top=ymax)
    if legend:
        (y_min, y_max) = plt.ylim()
        plt.ylim(y_min, (y_max - y_min) * 1.2 + y_min)
        plt.legend(
            fontsize=label_size,
            ncols=2,
            loc="best",
        )
    if show_axisnames:
        use_percent = scale == 100
        label_wer = "WER (%)" if (use_percent and not add_percents_in_ticks) else "WER"
        if use_percent and add_percents_in_ticks:
            yticks, yticks_labels = plt.yticks()
            yticks_labels = [f"${y.get_text()}\\%$" for y in yticks_labels]
            plt.yticks(yticks, labels=yticks_labels)
        func_ylabel(label_wer, fontsize=label_size, weight=label_weight)
        if x_axisname:
            plt.xlabel(x_axisname, fontsize=label_size, weight=label_weight)
    if title:
        plt.title(title, fontsize=label_size, weight=label_weight)
    if isinstance(show, str):
        plt.savefig(show, bbox_inches="tight")
    elif show:
        plt.show()


def check_result(wer_stats):
    if isinstance(wer_stats, dict):
        return min([k in wer_stats and isinstance(wer_stats[k], (int, float)) for k in ("wer", "del", "ins", "sub")])
    if isinstance(wer_stats, list):
        return min([check_result(w) for w in wer_stats])
    return False


def get_stat_list(wer_stats, key="wer"):
    if isinstance(wer_stats, dict):
        return [wer_stats[key]]
    if isinstance(wer_stats, list):
        return [w[key] for w in wer_stats]
    raise ValueError(f"Invalid type {type(wer_stats)}")


def get_stat_average(wer_stats, key="wer"):
    return np.mean(get_stat_list(wer_stats, key))


def plot_violinplot(data, positions=None, color="red", showquartiles=True, showmedians=True, **kwargs):
    import matplotlib.pyplot as plt

    if positions is None:
        positions = range(1, len(data) + 1)

    if isinstance(color, list):
        assert len(color) == len(data), f"{len(color)=} {len(data)=}"
        assert len(color) == len(positions)
        for x, y, c in zip(positions, data, color):
            if not len(y):
                continue
            plot_violinplot([y], positions=[x], color=c, showquartiles=showquartiles, showmedians=showmedians, **kwargs)
        return

    alpha = kwargs.pop("alpha", 1)

    parts = plt.violinplot(
        data,
        positions=positions,
        showmedians=showquartiles,
        showmeans=False,
        quantiles=([[0.25, 0.75]] * len(data)) if showquartiles else [],
        showextrema=showquartiles,
        **kwargs,
    )

    for pc in parts["bodies"]:
        # pc.set_facecolor('#D43F3A')
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.5 * alpha)

    if not showquartiles and not showmedians:
        return parts

    means = [np.mean(d) for d in data]
    quartiles = [np.percentile(d, [25, 50, 75]) for d in data]
    quartile1, medians, quartile3 = zip(*quartiles)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    plt.scatter(positions, means, marker="o", color=color, s=30, zorder=3)
    plt.scatter(positions, medians, marker="o", color="k", s=30, zorder=3)

    if not showquartiles:
        return parts

    plt.vlines(positions, quartile1, quartile3, color="k", linestyle="-", lw=5)
    plt.vlines(positions, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_f1_scores(
    wer_dict,
    label=True,
    legend=True,
    show=True,
    sort_best=-1,
    small_hatch=False,
    title=None,
    label_fontdict={"weight": "bold", "size": 12},
    ymin=0,
    ymax=None,
    interval_type=DEFAULT_INTERVAL_TYPE,
    x_axisname=None,
    colors=None,
    scale=100,
    add_percents_in_ticks=True,
    **kwargs,
):
    """
    Plot F1/Recall/Precision statistics.
    :param wer_dict: dictionary of results, or a list of results, or a dictionary of results,
        where a result is a dictionary as returned by compute_wer, or a list of such dictionaries
    :param label: whether to add a label to the bars (as xticks)
    :param legend: whether to add a legend (Deletion/Substition/Insertion)
    :param show: whether to show the plot (if True) or save it to the given file name (if string)
    :param sort_best: whether to sort the results by best WER
    :param small_hatch: whether to use small hatches for the bars
    :param colors: list of colors
    :param interval_type: "none", "boxplot", "violinplot" or "errorbar"
    :param **kwargs: additional arguments to pass to matplotlib.pyplot.bar
    """
    import matplotlib.pyplot as plt

    if colors is None:
        # rainbow colors
        num_colors = 8
        colors = [plt.cm.gist_rainbow(i / num_colors) for i in range(num_colors)]
        colors = colors[::2] + colors[1::2]
        colors = colors[3:] + colors[:3]
    if isinstance(wer_dict, list) and min([check_result(w) for w in wer_dict]):
        wer_dict = dict(enumerate(wer_dict))
    elif check_result(wer_dict):
        wer_dict = {"Evaluation": wer_dict}
    elif isinstance(wer_dict, dict) and min([check_result(w) for w in wer_dict.values()]):
        pass
    else:
        raise ValueError(
            "Invalid input (expecting a dictionary of results, a list of results, or a dictionary of results, \
where a result is a dictionary as returned by compute_wer, or a list of such dictionaries)"
        )

    plt.cla()

    kwargs.update(width=0.8, edgecolor="black")
    kwargs_f1 = kwargs.copy()
    kwargs_recall = kwargs.copy()
    kwargs_precision = kwargs.copy()
    n = 2 if small_hatch else 1
    kwargs_f1["hatch"] = "" * n
    kwargs_recall["hatch"] = "/" * n
    kwargs_precision["hatch"] = "\\" * n

    keys = list(wer_dict.keys())
    if sort_best:
        keys = sorted(keys, key=lambda k: get_stat_average(wer_dict[k]), reverse=not (sort_best < 0))
    positions = range(len(keys))

    offset_recall = len(positions) + 1
    offset_precision = 2 * offset_recall

    F1_list = [get_stat_average(wer_dict[k], "F1") * scale for k in keys]
    recall_list = [get_stat_average(wer_dict[k], "recall") * scale for k in keys]
    precision_list = [get_stat_average(wer_dict[k], "precision") * scale for k in keys]

    all_vals = None
    all_positions = None
    compute_intervals = max([len(get_stat_list(v, "F1") if "F1_samples" not in v else v["F1_samples"]) for v in wer_dict.values()]) > 1
    if compute_intervals:
        all_vals = []
        all_positions = []
        for offset, stat in [(0, "F1"), (offset_recall, "recall"), (offset_precision, "precision")]:
            for pos, k in zip(positions, keys):
                val_list = []
                if f"{stat}_samples" in wer_dict[k]:
                    for l in get_stat_list(wer_dict[k], f"{stat}_samples"):
                        val_list.extend(l)
                else:
                    val_list.extend(get_stat_list(wer_dict[k], stat))
                all_vals.append([v * scale for v in val_list])
                all_positions.append(pos + offset)

    for i_x, (pos, f1, recall, precision) in enumerate(zip(positions, F1_list, recall_list, precision_list)):
        add_opts = {}
        label_f1 = label_recall = label_prec = None
        add_opts["color"] = colors[i_x % len(colors)]
        add_opts["alpha"] = 0.5
        if label:
            system_label = keys[i_x]
            if system_label in [None, ""]:
                system_label = "_"
            label_f1 = system_label
            label_recall = label_prec = None
        plt.bar([pos], [f1], label=label_f1, **kwargs_f1, **add_opts)
        plt.bar([pos + offset_recall], [recall], label=label_recall, **kwargs_recall, **add_opts)
        plt.bar([pos + offset_precision], [precision], label=label_prec, **kwargs_precision, **add_opts)

    # # Add legend F1/Recall/Precision
    # add_opts["color"] = "white"
    # if not small_hatch:
    #     kwargs_f1["hatch"] *= 2
    #     kwargs_recall["hatch"] *= 2
    #     kwargs_precision["hatch"] *= 2
    # plt.bar([pos], [0], label="F1", **kwargs_f1, **add_opts)
    # plt.bar([pos], [0], label="Recall", **kwargs_recall, **add_opts)
    # plt.bar([pos], [0], label="Precision", **kwargs_precision, **add_opts)

    if all_vals and all_vals[0] and len(all_vals[0]) > 1:
        if interval_type == "violinplot":
            for i_x in range(len(all_vals)):
                plot_violinplot([all_vals[i_x]], positions=[all_positions[i_x]], color=colors[(i_x // 3) % len(colors)], alpha=0.5)
        elif interval_type == "boxplot":
            plt.boxplot(all_vals, positions=all_positions, whis=100)
        elif interval_type == "errorbar":
            lows, medians, highs = find_interval_around_median(all_vals)
            plt.errorbar(
                all_positions,
                medians,
                yerr=[medians - lows, highs - medians],
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1,
                capsize=2,
            )
        elif interval_type in ["none", None]:
            pass
        else:
            raise ValueError(f"Invalid interval_type {interval_type}")

    middle = (len(positions) - 1) / 2
    perf_names = ["F1", "Recall", "Prec."]
    use_percent = scale == 100
    if use_percent:
        if not add_percents_in_ticks:
            perf_names = [f"{p} (%)" for p in perf_names]
        else:
            yticks, yticks_labels = plt.yticks()
            yticks_labels = [f"${y.get_text()}\\%$" for y in yticks_labels]
            plt.yticks(yticks, labels=yticks_labels)
    plt.xticks(
        (middle, middle + offset_recall, middle + offset_precision),
        perf_names,
        rotation=0,
        fontdict=label_fontdict,
        ha="center",
    )
    plt.tick_params(axis="x", length=0)
    label_size = label_fontdict.get("size")
    plt.yticks(fontsize=label_size)
    if ymax is None:
        _, maxi = plt.ylim()
        plt.ylim(bottom=ymin, top=min(100, maxi))
    else:
        plt.ylim(bottom=ymin, top=ymax)
    if legend:
        plt.legend(fontsize=label_size)
    if x_axisname:
        plt.xlabel(x_axisname, fontsize=label_size)
    if title:
        plt.title(title, fontsize=label_size)
    if isinstance(show, str):
        plt.savefig(show, bbox_inches="tight")
    elif show:
        plt.show()


def find_interval_around_median(vals, coverage=0.95, symmetric=False):
    tensor2d = hasattr(vals[0], "__len__")
    if tensor2d:
        low_median_high = [find_interval_around_median(v, coverage=coverage, symmetric=symmetric) for v in vals]
        return (np.array(x) for x in zip(*low_median_high))
    median = np.median(vals)
    if symmetric:
        precision = 1e-9
        min_step = 5 if len(vals) >= 1000 else 1
        vals = np.array(sorted(vals))
        low = high = median
        current_ratio = -1
        current_interval = 0
        progress_bar = tqdm.tqdm(
            total=int(coverage * 100),
            desc=f"Finding symmetric interval (for {coverage * 100:.2f} % of {len(vals)} values)",
            leave=False,
        )
        while current_ratio < coverage:
            if current_ratio >= 0:
                dist_to_min = [v for v in np.clip(low - vals, 0, None) if v > precision]
                dist_to_max = [v for v in np.clip(vals - high, 0, None) if v > precision]
                all_dist = sorted(dist_to_min + dist_to_max)
                if len(all_dist):
                    min_dist = all_dist[min(min_step - 1, len(all_dist) - 1)]
                    assert min_dist > 0
                    current_interval += min_dist + precision
                else:
                    current_interval += precision
            low = median - current_interval
            high = median + current_interval
            num_on_interval = np.sum((vals >= low) & (vals <= high))
            previous_current_ratio = current_ratio
            current_ratio = float(num_on_interval) / len(vals)
            progress_bar.n = int(current_ratio * 100)
            progress_bar.refresh()
            # Sanity checks to avoid infinite loops
            assert current_ratio > previous_current_ratio, f"{current_ratio=} {previous_current_ratio=} ({median=} {current_interval=} {low=} {high=} {num_on_interval=})"
        progress_bar.close()
        return low, median, high
    else:
        low = np.percentile(vals, 50 - coverage * 50)
        high = np.percentile(vals, 50 + coverage * 50)
        return low, median, high


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("references", help="File with reference text lines (ground-truth)", type=str)
    parser.add_argument("predictions", help="File with predicted text lines (by an ASR system)", type=str, nargs="+")
    parser.add_argument(
        "--use_ids",
        help="Whether reference and prediction files includes id as a first field",
        default=True,
        type=str2bool,
        metavar="True/False",
    )
    parser.add_argument(
        "--alignment",
        "--debug",
        help="Output file to save debug information, or True / False",
        type=str,
        default=False,
        metavar="FILENAME/True/False",
    )
    parser.add_argument("--intervals", help="Add confidence intervals", default=False, action="store_true")
    parser.add_argument("--names", help="System names", default=[], nargs="+")
    parser.add_argument("--plot", help="See plots", default=False, action="store_true")
    parser.add_argument(
        "--fuse_plots",
        help="Fuse plots into the same figure when there are several plots (WER and F1/Recall/Precision, when --words_list is given)",
        default=False,
        action="store_true",
    )
    parser.add_argument("--include_correct_in_alignement", help="To also give correct alignement", action="store_true", default=False)
    parser.add_argument(
        "--norm",
        help="Language to use for text normalization ('fr', 'ar', ...). Use suffix '+' (ex: 'fr+', 'ar+', ...) to remove all non-alpha-num characters (apostrophes, dashes, ...)",
        default=None,
    )
    parser.add_argument("--char", default=False, action="store_true", help="For character-level error rate (CER)")
    parser.add_argument("--words_list", help="Files with list of words to focus on", default=None)
    parser.add_argument(
        "--words_blacklist",
        help="Files with list of words to exclude all the examples where the reference include such a word",
        default=None,
    )
    parser.add_argument(
        "--details_words_list",
        help="Output file to save information about words that are well recognized",
        type=str,
        default=False,
        metavar="FILENAME/True/False",
    )
    parser.add_argument(
        "--replacements",
        help="Files with list of replacements to perform in both reference and hypothesis",
        default=None,
    )
    parser.add_argument("--replacements_ref", help="Files with list of replacements to perform in references only", default=None)
    parser.add_argument("--replacements_pred", help="Files with list of replacements to perform in predicions only", default=None)
    args = parser.parse_args()

    targets = args.references
    predictions = args.predictions

    if not os.path.isfile(targets):
        assert not os.path.isfile(predictions), f"File {predictions} exists but {targets} doesn't"
        if " " not in targets and " " not in predictions:
            # Assume file instead of isolated word
            assert os.path.isfile(targets), f"File {targets} doesn't exist"
            assert os.path.isfile(predictions), f"File {predictions} doesn't exist"
        targets = [targets]
        predictions = [predictions]

    words_list = None
    if args.words_list:
        assert os.path.isfile(args.words_list), f"File {args.words_list} doesn't exist"
        word_list_name = os.path.splitext(os.path.basename(args.words_list))[0]
        with open(args.words_list) as f:
            words_list = [l.strip() for l in f.readlines()]

    words_blacklist = None
    if args.words_blacklist:
        assert os.path.isfile(args.words_blacklist), f"File {args.words_blacklist} doesn't exist"
        with open(args.words_blacklist) as f:
            words_blacklist = [l.strip() for l in f.readlines()]

    replacements_ref = {}
    replacements_pred = {}

    repl_ref_files = []
    repl_pred_files = []
    if args.replacements_ref:
        repl_ref_files.append(args.replacements_ref)
    if args.replacements_pred:
        repl_pred_files.append(args.replacements_pred)
    if args.replacements:
        repl_ref_files.append(args.replacements)
        repl_pred_files.append(args.replacements)

    for fn in repl_ref_files:
        with open(fn) as f:
            for l in f.readlines():
                trans = l.strip().split()
                if len(trans) != 2:
                    trans = l.strip().split("\t")
                assert len(trans) == 2, f"Invalid line {l}"
                replacements_ref[trans[0]] = trans[1]
    for fn in repl_pred_files:
        with open(fn) as f:
            for l in f.readlines():
                trans = l.strip().split()
                if len(trans) != 2:
                    trans = l.strip().split("\t")
                assert len(trans) == 2, f"Invalid line {l}"
                replacements_pred[trans[0]] = trans[1]

    alignment = args.alignment
    if alignment and alignment.lower() in ["true", "false"]:
        alignment = eval(alignment.title())

    details_words_list = args.details_words_list
    if details_words_list and details_words_list.lower() in ["true", "false"]:
        details_words_list = eval(details_words_list.title())
    use_ids = args.use_ids

    results = {}
    system_names = args.names or [""]
    for i, predictions_system in enumerate(predictions):
        system_name = system_names[i % len(system_names)].strip().replace("_", " ")
        system_name_0 = system_name
        i_0 = 1
        while system_name in results or (not system_name and len(predictions_system) > 1):
            system_name = system_name_0 + f" ({i_0})"
            i_0 += 1
        results[system_name] = compute_wer(
            targets,
            predictions_system,
            use_ids=use_ids,
            normalization=args.norm,
            character_level=args.char,
            alignment=alignment,
            include_correct_in_alignement=args.include_correct_in_alignement,
            words_list=words_list,
            words_blacklist=words_blacklist,
            replacements_ref=replacements_ref,
            replacements_pred=replacements_pred,
            details_words_list=details_words_list,
            bootstrapping=args.intervals,
        )

    for system_name, result in results.items():
        result_str = {}
        for k in "wer", "del", "ins", "sub", "word_err", "F1", "precision", "recall":
            result_str[k] = f"{result.get(k, -1) * 100:.2f} %".replace("-100.00 %", "_")
        for k in "TP", "FN", "FP":
            result_str[k] = str(result.get(k, "_"))
        if args.intervals:
            for k in result_str:
                if k + "_stdev" in result:
                    new_result = result_str[k]
                    add_percent = new_result.endswith("%")
                    if add_percent:
                        new_result = new_result[:-1].strip()
                    if k + "_samples" in result:
                        low, median, high = find_interval_around_median(result[k + "_samples"], symmetric=True)
                        assert abs((high - median) - (median - low)) < 0.0001, f"{low=} {median=} {high=}"
                        interval = (high - median) * 100
                    else:
                        assert k + "_stdev" in result
                        stdev = result[k + "_stdev"]
                        interval = stdev * 1.645 * 100  # Note : the 1.645 is the z-score for 90% confidence interval on a Normal distribution
                    new_result += f" ± {interval:.2f}"
                    if add_percent:
                        new_result += " %"
                    result_str[k] = new_result

        line = f" {'C' if args.char else 'W'}ER: {result_str['wer']} [ del: {result_str['del']} | ins: {result_str['ins']} | subs: {result_str['sub']} ](count: {result['count']})"
        if "word_err" in result:
            line = f" {word_list_name} err: {result_str['word_err']} |" + line

        if system_name:
            print()
            print("=" * len(line))
            print(f"Results for {system_name}:")
        print("-" * len(line))
        print(line)
        print("-" * len(line))
        if words_list:
            extra = f"Details for {len(words_list)} words:\n  "
            extra += " | ".join([f"{w}: {result_str[w]}" for w in ["F1", "precision", "recall"]])
            extra += " | " + " | ".join([f"{w}: {result_str[w]}" for w in ["TP", "FN", "FP"]])
            print(extra)

    if args.plot:
        import matplotlib.pyplot as plt

        kwargs = dict(
            sort_best=0,
        )
        wer_plot = True
        final_show = True  # "WER.pdf" # TODO: add an option to pass a filename
        if words_list:
            if args.fuse_plots:
                plt.subplot(1, 2, 1)
                plot_wer(results, show=False, **kwargs)
                wer_plot = False
                ax = plt.subplot(1, 2, 2)
                # Move y-axis to the right
                ax.yaxis.set_ticks_position("right")
                ax.yaxis.set_label_position("right")
                ax.spines["right"].set_position(("outward", 0))
                ax.spines["left"].set_position(("outward", 0))
            plot_f1_scores(results, show=final_show if args.fuse_plots else False, legend=not args.fuse_plots, **kwargs)
            if not args.fuse_plots:
                plt.figure()

        if wer_plot:
            plot_wer(results, show=final_show, **kwargs)
