from utils.FormatData import get_all_envipath_smirks, get_raw_envipath, run_reaction, encode_mol
from utils.EnviPathDownloader import check_envipath_data
from py4j.java_gateway import JavaGateway
import os
import json
from tqdm import tqdm
import subprocess
from argparse import ArgumentParser


def enviformer_coverage(smiles, dataset, args):
    with open("data/tokenizers_stereo.json") as token_file:
        tokenizers = json.load(token_file)["regex"]
    applicability_count = 0
    for smile in tqdm(smiles):
        smile = smile.split(">>")[0]
        encoded = encode_mol(smile, tokenizers, args)
        if encoded is not None:
            applicability_count += 1
    return applicability_count / len(smiles)


def envirule_coverage(smiles, dataset):
    java_gateway = JavaGateway()
    train_data_path = os.path.abspath(f"data/coverage_temp.txt")
    with open(train_data_path, "w") as file:
        for j, reaction in enumerate(smiles):
            r, p = reaction.split(">>")
            p_in_r = any(p == r1 for r1 in r.split("."))
            if not p_in_r and "*" not in reaction:
                file.write(reaction + "\n")
    rules = java_gateway.entry_point.generateRules(train_data_path, False, True, 1)
    rules = [rule for rule in rules]
    return run_rules(smiles, rules)


def envipath_coverage(smiles, dataset):
    rules = get_raw_envipath(dataset)["rules"]
    rules = [r["smirks"] for r in rules]
    if len(rules) == 0:
        return 0
    return run_rules(smiles, rules)


def run_rules(smiles, rules):
    applicability_count = 0
    for smile in tqdm(smiles):
        smile = smile.split(">>")[0]
        for rule in rules:
            products = run_reaction(rule, smile)
            if len(products) > 0:
                applicability_count += 1
                break
    return applicability_count / len(smiles)


def main(args):
    datasets = ["bbd", "soil", "sludge"]
    check_envipath_data()
    for dataset in datasets:
        smiles = get_all_envipath_smirks(args, files=[dataset])
        print(f"EnviFormer coverage on {dataset}: {enviformer_coverage(smiles, dataset, args):.2%}")
        print(f"EnviPath coverage on {dataset}: {envipath_coverage(smiles, dataset):.2%}")
        print(f"EnviRule coverage on {dataset}: {envirule_coverage(smiles, dataset):.2%}")
    return


if __name__ == "__main__":
    proc = subprocess.Popen(["java", "-jar", "java/envirule-2.6.0-jar-with-dependencies.jar"])
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--tokenizer", type=str, default="regex")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum encoded length to consider")
    parser.add_argument("--min-len", type=int, default=0, help="Minimum encoded length to consider")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as e:
        proc.kill()
        raise e
    proc.kill()
