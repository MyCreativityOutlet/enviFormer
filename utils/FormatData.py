import os
import numpy as np
import torch
from py4j.java_gateway import JavaGateway
import re
import random
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import json
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor, tensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import math
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from typing import Iterable


def run_reaction(rule, reactant_smiles, return_smiles=True):
    java_gateway = JavaGateway()
    pred_smiles = java_gateway.entry_point.runReaction(reactant_smiles, rule)
    if len(pred_smiles) == 0:
        return []

    if return_smiles:
        return [pred_smiles]
    else:
        try:
            out_inchi = set()
            for mol in pred_smiles.split("."):
                mol = Chem.MolFromSmiles(mol)
                if mol is not None:
                    out_inchi.add(Chem.MolToInchi(mol))
            return [out_inchi]
        except Exception as e:
            print(e)
            return []


def get_thresholds(value_type=list):
    thresholds = {}
    thresholds.update({i / 100: value_type() for i in range(-2000, -200, 10)})
    thresholds.update({i / 1000: value_type() for i in range(-2000, 0, 10)})
    return thresholds


def regex_tokenizer(smile: str) -> list:
    """Tokenizes a SMILES string using a regular expression.

    :param smile: The input SMILES string.
    :return: A list of tokens extracted from the SMILES string.
    """
    pattern = "(\[|\]|[A-Z][a-z]|[A-Z]|[a-z]|\%[0-9]{2}|:[0-9]{3}|:[0-9]{2}|:[0-9]|[0-9]{2}|[0-9]|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$)"
    regex = re.compile(pattern)
    tokens = regex.findall(smile)
    assert smile == ''.join(tokens), "Regex SMILE is not the same as original SMILE"
    return tokens


def encode_mol(mol: str, tokenizer: dict, args, enclose: bool = True) -> Tensor | None:
    """
    :param dropout: replace some tokens with padding
    :param mol: A string representing the molecule to be encoded.
    :param tokenizer: A dictionary containing the mapping of tokens to indices.
    :param args: Arguments related to the tokenizer.
    :param enclose: A boolean indicating whether to enclose the encoded sequence with special tokens.

    :return: A Tensor containing the encoded sequence, or None if encoding fails.

    This method encodes a given molecule using a specified tokenizer, the tokenizer can be either "selfies" or "regex".
    The encoded sequence is returned as a Tensor with dtype=torch.long. If encoding fails, None is returned.

    If the tokenizer is "selfies", the method uses the selfies library to encode the molecule.
    It first encodes the molecule as a selfies string using sf.encoder(mol), and then converts the selfies string to an
    encoding using sf.selfies_to_encoding(). If enclose is True, the encoded sequence is enclosed with special
    start-of-molecule (som) and end-of-molecule (eom) tokens.

    If the tokenizer is "regex", the method uses the regex_tokenizer() function to tokenize the molecule. Each token is
    then mapped to its corresponding index in the tokenizer dictionary. If enclose is True, the encoded sequence is
    enclosed with special som and eom tokens.
    """
    if args.tokenizer == "regex":
        try:
            tokens = [tokenizer[token] for token in regex_tokenizer(mol)]
        except KeyError as e:
            return None
        if enclose:
            tokens = [tokenizer["som"]] + tokens + [tokenizer["eom"]]
        if len(tokens) > args.max_len or len(tokens) < args.min_len:
            return None
        return tensor(tokens, dtype=torch.long)
    else:
        raise ValueError(f"Invalid tokenizer, {args.tokenizer}")


def encode_reactions(data: list, tokenizer: dict, args, dropout: float = 0.0) -> TensorDataset:
    x = [None for _ in range(len(data))]
    y = [None for _ in range(len(data))]
    failed_encoding = 0
    max_len = 0
    parallel_out = Parallel(n_jobs=get_workers(args.debug), batch_size=2000)(
        delayed(inner)(reaction, tokenizer, args, dropout, i) for i, reaction in enumerate(tqdm(data)))
    for p_out in parallel_out:
        if None not in p_out:
            max_len = max(len(p_out[0]), len(p_out[1]), max_len)
            x[p_out[-1]] = p_out[0]
            y[p_out[-1]] = p_out[1]
        else:
            failed_encoding += p_out.count(None)
    x = [v for v in x if v is not None]
    y = [v for v in y if v is not None]
    print(f"Failed to encode {failed_encoding} SMILES")
    print(f"Longest SMILES is {max_len} tokens.")
    x = pad_sequence(x, padding_value=tokenizer["[nop]"], batch_first=True)
    y = pad_sequence(y, padding_value=tokenizer["[nop]"], batch_first=True)
    data = TensorDataset(x, y)
    return data


def inner(reaction: str, tokenizer: dict, args, dropout, batch_num) -> tuple:
    reactant, product = reaction.split(">>")[:2]
    enc_r = encode_mol(reactant, tokenizer, args, enclose=False, dropout=dropout)
    enc_p = encode_mol(product, tokenizer, args)
    return enc_r, enc_p, reactant, batch_num


def augment_reactions(reactions: list[str], augment_target: int, args) -> list[str]:
    print(f"Augmenting reactions up to {augment_target} times.")
    n_jobs = get_workers(args.debug)
    batch_size = min(2000, math.ceil(len(reactions) / n_jobs))
    results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(delayed(augment_parallel)(reaction, augment_target)
                                                             for reaction in tqdm(reactions))
    output = []
    for result in results:
        output.extend(result)
    return output


def augment_parallel(reaction: str, augment_target: int = 1) -> list[str]:
    log_blocker = BlockLogs()
    reactant, product = reaction.split(">>")[:2]
    reactant_mol = Chem.MolFromSmiles(reactant)
    product_mol = Chem.MolFromSmiles(product)
    list_reactant = [reactant]
    list_product = [product]
    attempts = 0
    random.seed(1)
    while True:
        if len(list_reactant) < augment_target + 1:
            new_reactant = Chem.MolToSmiles(reactant_mol, doRandom=True)
            if new_reactant not in list_reactant:
                list_reactant.append(new_reactant)

        if len(list_product) < augment_target + 1:
            new_product = Chem.MolToSmiles(product_mol, doRandom=True)
            if new_product not in list_product:
                list_product.append(new_product)
        attempts += 1
        if len(list_reactant) >= augment_target + 1 and len(list_product) >= augment_target + 1:
            break

        if attempts > augment_target * 10:
            break
    new_reactions = set()
    for reactant, product in zip(list_reactant, list_product):
        new_reactions.add(reactant + ">>" + product)
    return list(new_reactions)


def canon_smile_rdkit(smile: str, remove_stereochemistry=True) -> str | None:
    log_blocker = BlockLogs()
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    if remove_stereochemistry:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def canon_smirk(smirk, canon_func):
    smirk = smirk.strip()
    reactant, product = smirk.split(">>")[:2]

    def canon_r_side(side):
        canon = []
        for r in side.split(">"):
            c_r = [canon_func(smile) for smile in r.split(".")]
            c_r = [smile for smile in c_r if smile is not None]
            if len(c_r) > 0:
                canon.append(".".join(c_r))
        canon = ">".join(canon)
        if len(canon) == 0:
            return None
        return canon

    canon_reactant = canon_r_side(reactant)
    if canon_reactant is None:
        return None
    canon_product = canon_r_side(product)
    if canon_product is None:
        return None
    return canon_reactant + ">>" + canon_product


def get_uspto_smirks(args) -> list:
    return load_smirks("data/uspto_stereo/stereo_joined.txt", args)


def get_cannon_func(canon_type):
    if canon_type == "rdkit":
        preprocessor = canon_smile_rdkit
    elif canon_type == "envipath":
        preprocessor = canon_smile_envipath
    else:
        raise ValueError(f"{canon_type} is invalid canonical function")
    return preprocessor


def load_smirks(file_name: str, args) -> list:
    with open(file_name) as d_file:
        lines = list(d_file.readlines())
    if args.debug:
        lines = lines[:2000]
    n_jobs = get_workers(args.debug)
    batch_size = min(2000, math.ceil(len(lines) / n_jobs))
    preprocessor = get_cannon_func(args.preprocessor)
    results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(delayed(canon_smirk)(line, preprocessor)
                                                             for line in tqdm(lines, desc="Preprocessing Reactions"))
    result_set = set()
    output = []
    for r in results:
        if r is not None and r not in result_set:
            output.append(r)
            result_set.add(r)
    return output


def split_data(data: list, split: float = None) -> tuple[Iterable, Iterable, Iterable]:
    if split is None:
        split = 0.9
        if len(data) >= 100000:
            split = 0.95
        elif len(data) >= 600000:
            split = 0.98
    train_val_data, test_data = train_test_split(data, train_size=split, shuffle=True, random_state=1)
    train_data, val_data = train_test_split(train_val_data, test_size=len(test_data), shuffle=True, random_state=1)
    return train_data, val_data, test_data


def get_reaction_smirks(package_name: str, args) -> list:
    with open(f"data/envipath/{package_name}.json") as d_file:
        data = json.load(d_file)
    n_jobs = get_workers(args.debug)
    batch_size = 200 if n_jobs > 1 else 1
    canon_func = get_cannon_func(args.preprocessor)
    data = [d["smirks"] for d in data["reactions"]]
    results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(delayed(canon_smirk)(r, canon_func) for r in tqdm(data))
    r_set = set()
    reactions = []
    for r in results:
        if r is not None and r not in r_set:
            r_set.add(r)
            reactions.append(r)
    return reactions


def get_raw_envipath(package_name: str) -> dict:
    with open(f"data/envipath/{package_name}.json") as d_file:
        data = json.load(d_file)
    return data


def get_all_envipath_smirks(args, files: list = None) -> list:
    if files is None:
        files = ["bbd", "soil", "sludge"]
    reactions = []
    for file in files:
        reactions.extend(get_reaction_smirks(file, args))
        if args.debug and len(reactions) > 5000:
            reactions = reactions[:5000]
            break
    return reactions


def get_workers(debug: bool = False) -> int:
    if debug:
        return 1
    return max(1, min(os.cpu_count() - 8, 32))


def get_loaders(train: TensorDataset, val: TensorDataset, test: TensorDataset, batch_size: int | tuple, args):
    """
    Generate data loaders for training, validation, and test sets.

    :param train: The training dataset as a TensorDataset.
    :param val: The validation dataset as a TensorDataset.
    :param test: The test dataset as a TensorDataset.
    :param batch_size: The batch size for the data loaders.
    :param args: Additional arguments.
    :return: A tuple containing the training, validation, and test data loaders.
    """
    num_workers = get_workers(args.debug)
    if type(batch_size) == tuple:
        train_size, val_size, test_size = batch_size
    else:
        train_size, val_size, test_size = batch_size, batch_size, batch_size
    train_loader = DataLoader(train, batch_size=train_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=val_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test, batch_size=test_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=True)
    return train_loader, val_loader, test_loader


def get_data_splits(data: list, train_ratio: float = None) -> tuple[list, list, list]:
    reactants = []
    reactant_set = set()
    for r in data:
        reactant = r.split(">>")[0]
        if reactant not in reactant_set:
            reactants.append(reactant)
            reactant_set.add(reactant)
    mapping = False
    if any(":" in r for r in reactants):
        mapping = True
        reactants = [remove_mapping(r) for r in reactants]
    train, val, test = split_data(reactants, train_ratio)
    train = set(train)
    val = set(val)
    test = set(test)
    if mapping:
        train = [r for r in data if remove_mapping(r.split(">>")[0]) in train]
        val = [r for r in data if remove_mapping(r.split(">>")[0]) in val]
        test = [r for r in data if remove_mapping(r.split(">>")[0]) in test]
    else:
        train = [r for r in data if r.split(">>")[0] in train]
        val = [r for r in data if r.split(">>")[0] in val]
        test = [r for r in data if r.split(">>")[0] in test]
    return train, val, test


def canon_smile_envipath(smile: str, gateway=None) -> str:
    if gateway is None:
        gateway = JavaGateway()
    return gateway.entry_point.standardSmiles(smile)


def get_dataset(args, num_rules: int = 1) -> tuple[list, tuple[dict, dict]]:
    """
    :param num_rules: Number of rules to use if using reduced envipath
    :param args: Argument Parser containing the arguments for getting the dataset.
        - data_name: The name of the dataset to fetch (options: "uspto", "envipath", "pubchem", "baeyer").
        - tokenizer: The type of tokenizer to use (options: "selfies", "regex").
        - debug: (optional) Whether to run in debug mode, limiting the dataset size to 5000.
    :return: A tuple containing the dataset and the tokenizer.
    """
    data_name = args.data_name
    tokenizer = args.tokenizer
    if data_name == "uspto":
        data = get_uspto_smirks(args)
    elif data_name == "envipath":
        data = get_all_envipath_smirks(args)
    elif data_name == "envipath_soil":
        data = get_all_envipath_smirks(args, files=["soil"])
    elif data_name == "envipath_bbd":
        data = get_all_envipath_smirks(args, files=["bbd"])
    elif data_name == "envipath_sludge":
        data = get_all_envipath_smirks(args, files=["sludge"])
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    with open("data/tokenizers_stereo.json") as token_file:
        tokenizers = json.load(token_file)
    try:
        char_to_i = tokenizers[tokenizer]
    except KeyError:
        raise KeyError(f"{tokenizer} is not a valid tokenizer")
    i_to_char = {i: char for char, i in char_to_i.items()}
    tokenizer = (char_to_i, i_to_char)
    return data, tokenizer


def remove_mapping(smiles, canonical=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=canonical)


def threshold_process(threshold, predictions):
    pred_set = set()
    max_score = float("-inf")
    min_score = 0.0
    for input_smiles, prediction in predictions.items():
        input_smiles = input_smiles.split(" ")[0]
        scores = prediction["scores"][0]
        prediction = prediction["predict"]
        product_scores = {}
        top_k = list(prediction.keys())
        top_k = max(top_k, key=lambda x: int(x.split("_")[-1]))
        k_product = prediction[top_k]
        i = int(top_k.split("_")[-1])
        k_product = k_product[:i]
        # k_product = [".".join(p) for p in k_product]
        for i, p in enumerate(k_product):
            if p in product_scores:
                product_scores[p].append(scores[i])
            else:
                product_scores[p] = [scores[i]]
        for product, score in product_scores.items():
            score = max(score)
            max_score = max(max_score, score)
            min_score = min(min_score, score)
            pred_reaction = input_smiles + ">>" + product
            # score = math.exp(score)
            if score >= threshold:
                pred_set.add(pred_reaction)
    return threshold, pred_set, max_score, min_score


def precision_recall_threshold(predictions, path, args, data_name=None, thresholds=None, save=True, extra_pr=None):
    if data_name is None:
        data_name = path.split("/")[-1]
    reactions = set()
    for input_smiles, prediction in predictions.items():
        input_smiles = input_smiles.split(" ")[0]
        actual_p = prediction["actual"]
        for a_p in actual_p:
            reactions.add(input_smiles + ">>" + a_p)
    if thresholds is None:
        thresholds = get_thresholds(value_type=set)
    max_score = float("-inf")
    min_score = 0.0
    results = Parallel(n_jobs=get_workers(args.debug))(delayed(threshold_process)(t, predictions) for t in tqdm(thresholds, desc="Calculating PR curve"))
    for threshold, pred_set, max_s, min_s in results:
        thresholds[threshold] = pred_set
        max_score = max(max_score, max_s)
        min_score = min(min_score, min_s)
    print(f"Minimum score: {min_score}")
    print(f"Maximum score: {max_score}")

    recall = {}
    precision = {}
    for threshold, pred_set in thresholds.items():
        num_correct = len(reactions) - len(reactions - pred_set)
        recall[threshold] = num_correct / len(reactions)
        precision[threshold] = num_correct / len(pred_set) if pred_set else 0.0

    # Sort the values by recall to ensure proper integration
    sorted_recall, sorted_precision, area_under_curve = sort_recall_precision(recall, precision)
    if save:
        plot_pr_curve(sorted_recall, sorted_precision, area_under_curve, data_name, path, extra_pr=extra_pr)
    return recall, precision


def plot_pr_curve(recall, precision, area, data_name, path, file_name=None, extra_pr=None, plot_title=None):
    fig_size = (7, 7)

    # Plot the curve with recall on the x-axis and precision on the y-axis
    plt.figure(figsize=fig_size)
    plt.style.use('seaborn-v0_8-dark-palette')
    if len(recall) > 0:
        plt.plot(recall, precision, label=f'{data_name} (AUC: {area:.4f})')
    if extra_pr is not None:
        for pr in extra_pr:
            plt.plot(pr[0], pr[1], label=pr[2])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add labels and title
    plt.xlabel('Recall', fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'Precision-Recall Curve: {data_name if plot_title is None else plot_title}', fontsize=18)

    # Show the legend
    plt.legend(fontsize=15, loc="upper right")

    # Show the plot or save it
    if ".png" in path:
        plt.savefig(path, dpi=120)
    else:
        for i in range(1, 100):
            save_loc = f"results/{path}/precision_recall_{i if file_name is None else file_name}.png"
            if not os.path.exists(save_loc) or file_name is not None:
                plt.savefig(save_loc, dpi=120)
                break
    plt.clf()
    plt.close()
    return


def reactions_from_pathways(pathways):
    co2 = {"O=C=O", "C(=O)=O"}
    reactions_set = set()
    reactions_list = []
    for pathway in pathways:
        for edge in pathway["links"]:
            reactant = pathway["nodes"][edge["source"]]["smiles"]
            product = pathway["nodes"][edge["target"]]["smiles"]
            if product not in co2:
                reaction = reactant + ">>" + product
                if reaction not in reactions_set:
                    reactions_list.append(reaction)
                    reactions_set.add(reaction)
    return reactions_list


def standardise_pathways(pathways, canon_func):
    for pathway in tqdm(pathways, desc="Fixing pathway edges and canonicalising SMILES"):
        edges_list = pathway["links"]
        nodes_list = pathway["nodes"]
        source_changes = {}
        target_changes = {}
        for i, edge in enumerate(edges_list):
            source = nodes_list[edge["source"]]
            source_i = edge["source"]
            target = nodes_list[edge["target"]]
            target_i = edge["target"]
            while source["depth"] == -1:
                for edge2 in edges_list:
                    if nodes_list[edge2["target"]] == source:
                        source = nodes_list[edge2["source"]]
                        source_i = edge2["source"]
                        break
            while target["depth"] == -1:
                for edge2 in edges_list:
                    if nodes_list[edge2["source"]] == target:
                        target = nodes_list[edge2["target"]]
                        target_i = edge2["target"]
                        break
            if edge["source"] != source_i:
                source_changes[i] = source_i
            if edge["target"] != target_i:
                target_changes[i] = target_i
        for i in source_changes:
            edges_list[i]["source"] = source_changes[i]
        for i in target_changes:
            edges_list[i]["target"] = target_changes[i]
        for node in pathway["nodes"]:
            if "smiles" in node:
                node["smiles"] = canon_func(node["smiles"])

    return pathways


def get_all_pathways(args):
    files = ["soil", "bbd", "sludge"]
    pathways = []
    for file in files:
        with open(f"data/envipath/{file}.json") as file:
            data = json.load(file)
        pathways.extend(data["pathways"])
    canon_func = get_cannon_func(args.preprocessor)
    pathways = standardise_pathways(pathways, canon_func)
    return pathways


def pathways_split(args):
    pathway_file = ""
    if "soil" in args.data_name:
        pathway_file = "soil"
    if "bbd" in args.data_name:
        pathway_file = "bbd"
    if "sludge" in args.data_name:
        pathway_file = "sludge"
    with open(f"data/envipath/{pathway_file}.json") as file:
        data = json.load(file)
    pathways = data["pathways"]

    # Fix the pathways, removing edges with no smiles and depth 0, also canon all SMILES
    canon_func = get_cannon_func(args.preprocessor)
    pathways = standardise_pathways(pathways, canon_func)

    # Create the folds
    folds = []
    for train, test in KFold(n_splits=10, shuffle=True, random_state=1).split(pathways):
        train_pathways = [pathways[i] for i in train]
        test_pathways = [pathways[i] for i in test]

        # Generate the training set
        training_set = reactions_from_pathways(train_pathways)
        test_set = reactions_from_pathways(test_pathways)

        # Remove any reactions from the training set containing molecules in the test set
        to_remove = set()
        for r in training_set:
            reactant, product = r.split(">>")
            for test_r in test_set:
                test_reactant, test_product = test_r.split(">>")
                if test_reactant == reactant or test_product == product:
                    to_remove.add(r)
                    break
        for r in to_remove:
            training_set.remove(r)
        folds.append([training_set, test_pathways, test_set])
    return folds


def sort_recall_precision(recall, precision):
    sorted_recall = sorted(recall.items(), key=lambda x: x[1])
    order = [r[0] for r in sorted_recall]
    sorted_recall = np.array([r[1] for r in sorted_recall])
    sorted_precision = np.array([precision[key] for key in order])
    area_under_curve = auc(sorted_recall, sorted_precision)
    return sorted_recall, sorted_precision, area_under_curve
