import torch
from datetime import datetime
import pandas as pd
import json
from tqdm import trange
import os
from argparse import ArgumentParser
from models.TransformerModel import TransformerModel
from EnviRuleExperiments import EnviRuleModel
from utils.FormatData import get_all_envipath_smirks, get_all_pathways
from utils.EvalFunctions import predict_multigen


def run_experiment(args, save_folder):
    if args.model_name == "TransformerModel":
        with open(f"results/TransformerModel/uspto_regex/tokens.json") as t_file:
            tokenizer = json.load(t_file)
        with open(f"models/TransformerModel_config.json") as config_file:
            config = json.load(config_file)
        tokenizer[1] = {int(k): v for k, v in tokenizer[1].items()}
        model = TransformerModel.load_from_checkpoint("results/TransformerModel/soil_regex/checkpoints/epoch=148-step=969838.ckpt",
                                                      map_location="cpu", config=config, vocab=tokenizer, p_args=args)
        model = model.to(args.device)
        model = torch.compile(model, mode="reduce-overhead")
    elif args.model_name == "envirule":
        with open("results/envirule/soil_regex/fold_0_rules.txt") as rule_file:
            rules = rule_file.readlines()
        rules = [r.strip() for r in rules]
        model = EnviRuleModel(rules, "soil_regex", "envirule", 0, args)
        if args.debug:
            model.classifier.num_jobs = 1
    else:
        raise ValueError(f"Can't instantiate model of type {args.model_name}")

    reaction_data = get_all_envipath_smirks(args)
    pathway_data = get_all_pathways(args)
    repeats = 1

    pathway_times = {1: []}
    for _ in range(repeats):
        for batch_size in pathway_times.keys():
            print(f"Testing pathway prediction time on batch size {batch_size}")
            for i in trange(0, len(pathway_data), batch_size):
                pathways = pathway_data[i: i+batch_size]
                if len(pathways) < batch_size:
                    break
                start_time = datetime.now()
                _ = predict_multigen(model, pathways, [float("-inf")], args, evaluate=False, max_depth=5, max_width=3)
                duration = (datetime.now() - start_time).total_seconds()
                pathway_times[batch_size].append(duration)

    reaction_times = {1: [], 10: [], 100: []}
    for _ in range(repeats):
        for batch_size in reaction_times.keys():
            print(f"Testing reaction prediction time on batch size {batch_size}")
            for i in trange(0, len(reaction_data), batch_size):
                reactions = reaction_data[i: i+batch_size]
                reactions = [r.split(">>")[0] for r in reactions]
                if len(reactions) < batch_size:
                    break
                start_time = datetime.now()
                _ = model.smiles_to_smiles_inf(reactions)
                duration = (datetime.now() - start_time).total_seconds()
                reaction_times[batch_size].append(duration)
            with open(os.path.join(save_folder, "times.json"), "w") as r_file:
                json.dump({"reaction": reaction_times, "pathway": pathway_times}, r_file, indent=4)
    return


def plot_results(results, save_folder):
    reaction = results["reaction"]
    pathway = results["pathway"]
    results_file = open(os.path.join(save_folder, "summary.txt"), "w")
    for batch_size, times in reaction.items():
        results_file.write(f"Statistics for reaction times with batch size of {batch_size}\n")
        times = pd.DataFrame(times)
        results_file.write(str(times.describe()) + "\n")
    for batch_size, times in pathway.items():
        results_file.write(f"Statistics for pathway times with batch size of {batch_size}\n")
        times = pd.DataFrame(times)
        results_file.write(str(times.describe()) + "\n")
    results_file.close()
    return


def main(args):
    save_folder = f"results/{args.model_name}/runtime_{args.device}"
    os.makedirs(save_folder, exist_ok=True)
    if not os.path.exists(os.path.join(save_folder, "times.json")):
        run_experiment(args, save_folder)
    with open(os.path.join(save_folder, "times.json")) as r_file:
        results = json.load(r_file)
    plot_results(results, save_folder)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="envirule")
    parser.add_argument("--data-name", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="regex")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum encoded length to consider")
    parser.add_argument("--min-len", type=int, default=0, help="Minimum encoded length to consider")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size to target")
    parser.add_argument("--preprocessor", type=str, default="envipath",
                        help="Type of preprocessing to use, envipath or rdkit")
    parser.add_argument("--device", type=str, default="cpu")
    arguments = parser.parse_args()
    main(arguments)
