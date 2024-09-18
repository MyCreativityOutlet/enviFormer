import sys
import torch
import copy
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import json
from torch.nn import functional as F
from torch import Tensor, tensor, nn
from torch.optim.lr_scheduler import LambdaLR
import math
from py4j.java_gateway import JavaGateway
from tqdm import tqdm
from rdkit.rdBase import BlockLogs
from statistics import mean
from utils.FormatData import precision_recall_threshold, get_workers, canon_smile
from functools import partial


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        pe = self.pe[:, :x.size(1), :]
        x = x + pe
        return self.dropout(x)


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


def seq_metrics(pred_is: Tensor, y: Tensor, pad_id: int) -> tuple[Tensor, Tensor]:
    pred_is[y == pad_id] = pad_id
    seq_accuracy = torch.eq(pred_is, y)
    seq_accuracy = torch.all(seq_accuracy, 1)
    seq_accuracy = torch.count_nonzero(seq_accuracy) / seq_accuracy.nelement()
    pred_is = pred_is.flatten()
    y = y.flatten()
    char_accuracy = torch.eq(pred_is, y)
    padding = torch.nonzero(torch.eq(y, pad_id))
    char_accuracy[padding] = False
    char_accuracy = torch.count_nonzero(char_accuracy) / (char_accuracy.size(0) - padding.flatten().size(0))
    return seq_accuracy, char_accuracy


def save_train_metrics(model_metrics: dict[str, dict], path: str, steps_per_epoch: dict):
    metrics = copy.deepcopy(model_metrics)
    for metric in metrics:
        for s_type in metrics[metric]:
            means = [mean(metrics[metric][s_type][i:i + steps_per_epoch[s_type]]) for i in
                     range(0, len(metrics[metric][s_type]), steps_per_epoch[s_type])]
            metrics[metric][s_type] = means

    with open(f"results/{path}/train_metrics.json", "w") as m_file:
        json.dump(metrics, m_file, indent=4)
    plt.clf()
    fig, axs = plt.subplots(len(metrics), sharex="all")
    fig.set_size_inches(18, 5 * len(metrics))
    plt.suptitle(f"Train Metrics for {path}")
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 26
    axs = axs.flat
    for i, (metric, values) in enumerate(metrics.items()):
        axs[i].set_title(metric, fontsize=BIGGER_SIZE)
        step_num = len(max(values.values(), key=lambda x: len(x)))
        for step_type, value in values.items():
            x_step = step_num // len(value) if len(value) > 0 else 1
            x_coords = list(range(0, step_num, x_step))
            axs[i].plot(x_coords[:len(value)], value, label=step_type)
        axs[i].tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
        axs[i].tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
        axs[i].set_xlabel('Epoch', fontsize=SMALL_SIZE)
        axs[i].legend(list(values.keys()), loc="lower left", fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    plt.savefig(f"results/{path}/train_metrics.png", dpi=120)
    plt.close()


def remove_special_chars(mol_str: str) -> str:
    special_chars = [f"som{i}" for i in range(10)] + ["eom", "som", "[nop]"]
    for char in special_chars:
        mol_str = mol_str.replace(char, "")
    return mol_str


def _sort_beams(mol_strs, log_lhs, all_ll):
    """ Return mols sorted by their log likelihood"""

    assert len(mol_strs) == len(log_lhs)

    sorted_mols = []
    sorted_lls = []
    sorted_all_ll = []

    for mols, lls, all_ls in zip(mol_strs, log_lhs, all_ll):
        mol_lls = sorted(zip(mols, lls, all_ls), reverse=True, key=lambda mol_ll: mol_ll[1])
        mols, lls, all_ls = tuple(zip(*mol_lls))
        sorted_mols.append(torch.stack(mols))
        sorted_lls.append(torch.stack(lls))
        sorted_all_ll.append(torch.stack(all_ls))

    return torch.stack(sorted_mols), torch.stack(sorted_lls), torch.stack(sorted_all_ll)


def beam_step(decode_func, model, tokens, lls):
    output_dist = decode_func(tokens)
    next_token_lls = output_dist[:, -1, :]
    _, vocab_size = next_token_lls.size()
    complete_seq_ll = torch.ones(1, vocab_size,
                                 device=output_dist.device) * -1e5  # Use -1e5 for log softmax or 0 for softmax
    complete_seq_ll[:, model.char_to_i["[nop]"]] = 0.0  # Use 0.0 for log softmax or 1.0 for softmax

    # Use this vector in the output for sequences which are complete
    is_end_token = tokens[:, -1] == model.char_to_i["eom"]
    is_pad_token = tokens[:, -1] == model.char_to_i["[nop]"]
    ll_mask = torch.logical_or(is_end_token, is_pad_token).unsqueeze(1)
    masked_lls = (ll_mask * complete_seq_ll) + (~ll_mask * next_token_lls)

    seq_lls = (lls + masked_lls.T).T
    return seq_lls


def norm_length(seq_lls, mask, length_norm=None):
    """ Normalise log-likelihoods using the length of the constructed sequence
    Equation from:
    Wu, Yonghui, et al.
    "Google's neural machine translation system: Bridging the gap between human and machine translation."
    arXiv preprint arXiv:1609.08144 (2016).

    Args:
        seq_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size] containing log likelihoods for seqs so far
        mask (torch.Tensor): BoolTensor of shape [seq_len, batch_size] containing the padding mask

    Returns:
        norm_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size]
    """

    if length_norm is not None:
        seq_lengths = (~mask).sum(dim=1)
        norm = torch.pow(5 + seq_lengths, length_norm) / pow(6, length_norm)
        norm_lls = (seq_lls.T / norm).T
        return norm_lls

    return seq_lls


def update_beams(i, decode_func, model, token_ids_list, pad_mask_list, lls_list, all_ll, length_norm):
    """Update beam tokens and pad mask in-place using a single decode step

    Updates token ids and pad mask in-place by producing the probability distribution over next tokens
    and choosing the top k (number of beams) log likelihoods to choose the next tokens.
    Sampling is complete if every batch element in every beam has produced an end token.
    """

    assert len(token_ids_list) == len(pad_mask_list) == len(lls_list)

    num_beams = len(token_ids_list)

    ts = [token_ids[:, :i] for token_ids in token_ids_list]
    ms = [pad_mask[:, :i] for pad_mask in pad_mask_list]

    # Apply current seqs to model to get a distribution over next tokens
    # new_lls is a tensor of shape [batch_size, vocab_size * num_beams]
    new_lls = [beam_step(decode_func, model, t, lls) for t, lls in zip(ts, lls_list)]
    norm_lls = [norm_length(lls, mask, length_norm) for lls, mask in zip(new_lls, ms)]

    _, vocab_size = tuple(new_lls[0].shape)
    new_lls = torch.cat(new_lls, dim=1)
    norm_lls = torch.cat(norm_lls, dim=1)

    # Keep lists (of length num_beams) of tensors of shape [batch_size]
    top_lls, top_idxs = torch.topk(norm_lls, num_beams, dim=1)
    all_ll[:, :, i] += top_lls
    new_ids_list = list((top_idxs % vocab_size).T)
    beam_idxs_list = list((top_idxs // vocab_size).T)
    top_lls = [new_lls[b_idx, idx] for b_idx, idx in enumerate(list(top_idxs))]
    top_lls = torch.stack(top_lls).T

    beam_complete = []
    new_ts_list = []
    new_pm_list = []
    new_lls_list = []

    # Set the sampled tokens, pad masks and log likelihoods for each of the new beams
    for new_beam_idx, (new_ids, beam_idxs, lls) in enumerate(zip(new_ids_list, beam_idxs_list, top_lls)):
        # Get the previous sequences corresponding to the new beams
        token_ids = [token_ids_list[beam_idx][b_idx, :] for b_idx, beam_idx in enumerate(beam_idxs)]
        token_ids = torch.stack(token_ids)

        # Generate next elements in the pad mask. An element is padded if:
        # 1. The previous token is an end token
        # 2. The previous token is a pad token
        is_end_token = token_ids[:, i - 1] == model.char_to_i["eom"]
        is_pad_token = token_ids[:, i - 1] == model.char_to_i["[nop]"]
        new_pad_mask = torch.logical_or(is_end_token, is_pad_token)
        beam_complete.append(new_pad_mask.sum().item() == new_pad_mask.numel())

        # Ensure all sequences contain an end token
        if i == model.max_len - 1:
            new_ids[~new_pad_mask] = model.char_to_i["eom"]

        # Set the tokens to pad if an end token as already been produced
        new_ids[new_pad_mask] = model.char_to_i["[nop]"]
        token_ids[:, i] = new_ids

        # Generate full pad mask sequence for new token sequence
        pad_mask = [pad_mask_list[beam_idx][b_idx, :] for b_idx, beam_idx in enumerate(beam_idxs)]
        pad_mask = torch.stack(pad_mask)
        pad_mask[:, i] = new_pad_mask

        # Add tokens, pad mask and lls to list to be updated after all beams have been processed
        new_ts_list.append(token_ids)
        new_pm_list.append(pad_mask)
        new_lls_list.append(lls)

    complete = sum(beam_complete) == len(beam_complete)

    # Update all tokens, pad masks and lls
    if not complete:
        for beam_idx, (ts, pm, lls) in enumerate(zip(new_ts_list, new_pm_list, new_lls_list)):
            token_ids_list[beam_idx] = ts
            pad_mask_list[beam_idx] = pm
            lls_list[beam_idx] = lls

    return complete


def beam_decode(model, x: Tensor, num_beams: int = 8):
    if model.__class__.__name__ == "EnviFormerModel":
        enc_out, src_mask = model.encode(x)
        decode_func = partial(model.decode, enc_out=enc_out, src_mask=src_mask, test=True)
    else:
        raise ValueError(f"Mode of type {model.__class__.__name__} cannot be beam decoded")
    token_ids = [([model.char_to_i["som"]] + ([model.char_to_i["[nop]"]] * (model.max_len - 1)))] * x.size(0)
    token_ids = tensor(token_ids, device=x.device)
    pad_mask = torch.zeros(x.size(0), model.max_len, device=x.device, dtype=torch.bool)
    ts = token_ids[:, :1]
    ll = torch.zeros(x.size(0), device=x.device)
    all_ll = torch.zeros((x.size(0), num_beams, model.max_len), device=x.device)

    first_lls = beam_step(decode_func, model, ts, ll)
    top_lls, top_ids = torch.topk(first_lls, num_beams, dim=-1)
    all_ll[:, :, 1] += top_lls
    top_ids = list(top_ids.T)

    token_ids_list = [token_ids.clone() for _ in range(num_beams)]
    pad_mask_list = [pad_mask.clone() for _ in range(num_beams)]
    lls_list = list(top_lls.T)

    for beam_id, ids in enumerate(top_ids):
        token_ids_list[beam_id][:, 1] = ids
        pad_mask_list[beam_id][:, 1] = False

    for i in range(2, model.max_len):
        complete = update_beams(i, decode_func, model, token_ids_list, pad_mask_list, lls_list, all_ll,
                                model.model_config["length_norm"])
        if complete:
            break

    tokens_tensor = torch.stack(token_ids_list).permute(1, 0, 2)
    log_lhs_tensor = torch.stack(lls_list).permute(1, 0)
    sorted_mols, sorted_lls, sorted_all_lls = _sort_beams(tokens_tensor, log_lhs_tensor, all_ll)

    return sorted_mols, sorted_lls, sorted_all_lls


def decode_mol(array, i_to_char, tokenizer_type):
    if tokenizer_type == "regex":
        decoded = [i_to_char[i] for i in array]
        smiles = "".join(decoded)
        smiles = remove_special_chars(smiles)
    else:
        raise ValueError(f"Unknown tokenizer, {tokenizer_type}")
    mols = smiles.split(".")
    canon_mols = []
    for mol in mols:
        canon = canon_smile(mol)
        if canon is not None and len(canon) > 0:
            canon_mols.append(canon)
    return ".".join(canon_mols)


def process_output(inputs, predict, actual, score, i_to_char, args):
    predict = predict.numpy()
    actual = actual.numpy()
    inputs = inputs.numpy()
    score = score.numpy()
    if predict.ndim == 1:
        m_smiles = [decode_mol(predict, i_to_char, args.tokenizer)]
    else:
        m_smiles = [decode_mol(beam, i_to_char, args.tokenizer)
                    for beam in predict]
    a_smiles = decode_mol(actual, i_to_char, args.tokenizer)
    input_smiles = decode_mol(inputs, i_to_char, args.tokenizer)
    return input_smiles, m_smiles, a_smiles, score.tolist()


def process_seq_test_outputs(outputs: list, path: str, i_to_char: dict, args) -> dict:
    log_blocker = BlockLogs()
    flat_outputs = [[], []]
    flat_scores = []
    flat_inputs = []
    n_jobs = get_workers(args.debug)
    for inputs, (predict, actual, score) in outputs:
        flat_scores.extend(score)
        flat_inputs.extend(inputs)
        flat_outputs[0].extend(predict)
        flat_outputs[1].extend(actual)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_output)(inputs, predict, actual, score, i_to_char, args) for
        inputs, predict, actual, score in tqdm(zip(flat_inputs, flat_outputs[0], flat_outputs[1], flat_scores),
                                               total=len(flat_outputs[0]), desc="Decoding output"))
    results_dict = {}
    accuracy = {}
    invalid_smiles = {}
    if flat_outputs[0][0].ndim == 1:
        top_k = [1]
    else:
        top_k = range(1, len(flat_outputs[0][0]) + 1)
    for input_smiles, m_smiles, a_smiles, score in tqdm(results, desc="Calculating metrics"):
        if len(a_smiles) == 0 or len(input_smiles) == 0:
            continue
        if input_smiles not in results_dict:
            results_dict[input_smiles] = {"actual": [], "scores": [], "predict": {f"top_{k}": [] for k in top_k}}
        elif args.score_all:
            r_numbers = [1]
            for k in results_dict.keys():
                split = k.split(" ")
                if split[0] == input_smiles:
                    if len(split) > 1:
                        r_numbers.append(int(split[-1]) + 1)
            input_smiles += " " + str(max(r_numbers))
            results_dict[input_smiles] = {"actual": [], "scores": [], "predict": {f"top_{k}": [] for k in top_k}}
        # m_smiles = [set(m.split(".")) for m in m_smiles]
        results_dict[input_smiles]["actual"].append(a_smiles)
        results_dict[input_smiles]["scores"].append(score)
        for k in top_k:
            key = f"top_{k}"
            results_dict[input_smiles]["predict"][key].extend(m_smiles[:k])
    for reactant in results_dict:
        real_products = results_dict[reactant]["actual"]
        for k in top_k:
            key = f"top_{k}"
            pred_products = results_dict[reactant]["predict"][key]
            if key not in accuracy:
                accuracy[key] = []
            if key not in invalid_smiles:
                invalid_smiles[key] = []
            invalid_smiles[key].append(sum(int(len(p) == 0) for p in pred_products))
            for real in real_products:
                success = 0
                real = set(real.split("."))
                for predict in pred_products:
                    predict = set(predict.split("."))
                    # if real_inchi_set == pred_inchi_set:  # This requires strict product set equality
                    if len(real - predict) == 0:  # This allows extra products to be predicted
                        success = 1
                        break
                accuracy[key].append(success)
    for key in accuracy:
        accuracy[key] = round(sum(accuracy[key]) / len(accuracy[key]), 4)
        invalid_smiles[key] = sum(invalid_smiles[key])
    recall, precision = precision_recall_threshold(results_dict, path, args)
    save_dict = {"invalid_smiles": invalid_smiles, "accuracy": accuracy,
                 "predictions": results_dict, "recall": recall, "precision": precision}

    with open(f"results/{path}/test_output.json", "w") as out_file:
        json.dump(save_dict, out_file, indent=4)
    return save_dict
