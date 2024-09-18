import torch.cuda
from argparse import ArgumentParser
import subprocess
import pytorch_lightning as pl
from PreTrainEnviFormer import setup_model, build_trainer
from utils.EvalFunctions import predict_singlegen, predict_multigen
from utils.FormatData import *


def setup_train(args, train_data):
    print(f"Debug: {args.debug}\nModel: {args.model_name}\nDataset: {args.data_name}\nTokenizer: {args.tokenizer}")
    print("Setting up")
    pl.seed_everything(2)
    torch.set_float32_matmul_precision('high')
    model_class, config = setup_model(args)
    with open(f"results/{args.model_name}/{args.weights_dataset}_{args.tokenizer}/tokens.json") as t_file:
        tokenizer = json.load(t_file)
    tokenizer[1] = {int(k): v for k, v in tokenizer[1].items()}
    model = model_class(config, vocab=tokenizer, p_args=args)
    train_reactants_set = set()
    train_reactants = []
    for r in train_data:
        reactant = r.split(">>")[0]
        if reactant not in train_reactants_set:
            train_reactants_set.add(reactant)
            train_reactants.append(reactant)

    train_reactants, val_reactants = train_test_split(train_reactants, train_size=0.9, shuffle=True, random_state=1)
    train_reactants = set(train_reactants)
    val_reactants = set(val_reactants)
    train = [r for r in train_data if r.split(">>")[0] in train_reactants]
    val = [r for r in train_data if r.split(">>")[0] in val_reactants]
    train_data = encode_reactions(train, tokenizer[0], args)
    val_data = encode_reactions(val, tokenizer[0], args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=get_workers(args.debug),
                              persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=get_workers(args.debug),
                            persistent_workers=True)
    trainer = build_trainer(args, config)
    print("Beginning training")
    ckpt_path = f"results/{args.model_name}/{args.weights_dataset}_{args.tokenizer}_weights.ckpt" \
        if args.weights_dataset else None
    if "baseline" in args.data_name:
        model = model.__class__.load_from_checkpoint(ckpt_path, config=config, vocab=tokenizer, p_args=args)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        model = model.__class__.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, config=config, vocab=tokenizer, p_args=args)
    with open(f"results/{args.model_name}/{args.data_name}_{args.tokenizer}/tokens.json", "w") as file:
        json.dump(tokenizer, file, indent=4)
    with open(f"results/{args.model_name}/{args.data_name}_{args.tokenizer}/config.json", "w") as file:
        json.dump(config, file, indent=4)
    return model, trainer, tokenizer


def mean_dicts(results: list[dict]) -> dict:
    mean_results = {}
    for key in results[0].keys():
        if key == "predictions":
            mean_results["predictions"] = results[0]["predictions"]
            continue
        avg = {}
        for k in results[0][key]:
            top_k = []
            for result in results:
                top_k.append(result[key][k])
            top_k = round(sum(top_k) / len(top_k), 4)
            if key not in avg:
                avg[key] = {}
            avg[key][k] = top_k
        mean_results.update(avg)
    return mean_results


def extract_data(data_list):
    finger_data = {"top_1": [], "top_2": [], "top_3": [], "top_4": [], "top_5": []}
    acc_data = {"top_1": [], "top_2": [], "top_3": [], "top_4": [], "top_5": []}
    for data_dict in data_list:
        for key in finger_data.keys():
            finger_data[key].append(data_dict['fingerprint_similarity'][key])
            acc_data[key].append(data_dict['accuracy'][key])

    return finger_data, acc_data


def load_folds(path):
    fold_paths = [p for p in os.listdir(path) if "splits" in p]
    folds = []
    for p in fold_paths:
        with open(os.path.join(path, p)) as f_file:
            data = json.load(f_file)
            fold = [data["train"], data["test"], data["test_reactions"]]
            folds.append(fold)
    return folds


def train_eval_single_multi(args):
    random.seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_test_output = {}
    single_test_output = {}
    thresholds = get_thresholds(value_type=set)
    results_directory = f"results/{args.model_name}/{args.data_name}_{args.tokenizer}"
    curve_directory = "/".join(results_directory.split("/")[1:])
    extra_data = []
    co2 = {"O=C=O", "C(=O)=O"}
    if os.path.exists(results_directory):
        folds = load_folds(results_directory)
    elif "leave" in args.data_name:
        canon_func = get_cannon_func(args.preprocessor)
        if "soil" in args.data_name:
            train = get_all_envipath_smirks(args, ["bbd", "sludge"])
            test = get_raw_envipath("soil")
        elif "bbd" in args.data_name:
            train = get_all_envipath_smirks(args, ["soil", "sludge"])
            test = get_raw_envipath("bbd")
        elif "sludge" in args.data_name:
            train = get_all_envipath_smirks(args, ["soil", "bbd"])
            test = get_raw_envipath("sludge")
        else:
            raise ValueError(f"Can't use {args.data_name} unknown dataset type")
        test_pathways = standardise_pathways(test["pathways"], canon_func)
        test_reactions = [canon_smirk(r["smirks"], canon_func) for r in test["reactions"]]
        test_reactions = [r for r in test_reactions if r is not None and r.split(">>")[-1] not in co2]
        folds = [[train, test_pathways, test_reactions]] * 1
    elif "add" in args.data_name:
        extra_data_name = args.data_name.split("_")[-1]
        full_data_name = args.data_name
        args.data_name = args.data_name.split("_")[0]
        folds = pathways_split(args)
        args.data_name = full_data_name
        extra_data = get_all_envipath_smirks(args, [extra_data_name])
        extra_data = [r for r in extra_data if r is not None and r.split(">>")[-1] not in co2]
    else:
        folds = pathways_split(args)

    os.makedirs(results_directory, exist_ok=True)
    for count, (train_data, test_pathways, test_data) in enumerate(folds):
        print(f"Training on fold {count}")
        train_data.extend(extra_data)
        train_set = set(train_data)
        to_remove = set()
        for r in test_data:  # Double check nothing in train set that shouldn't be after extra data or from leave out
            if r in train_set:
                to_remove.add(r)
        train_data = [r for r in train_data if r not in to_remove]
        model, trainer, tokenizer = setup_train(args, train_data)
        with open(f"results/{args.model_name}/{args.data_name}_{args.tokenizer}/fold_{count}_splits.json", "w") as file:
            json.dump({"train": train_data, "test": test_pathways, "test_reactions": test_data}, file)
        model = model.to(device)

        print(f"Performing single gen evaluation on fold {count}.")
        single_test_output[count] = predict_singlegen(model, test_data, thresholds)
        with open(f"{results_directory}/test_output_single.json", "w") as out_file:
            json.dump(single_test_output, out_file, indent=4)
        single_recall, single_precision, single_area = sort_recall_precision(single_test_output[count]["recall"],
                                                                             single_test_output[count]["precision"])
        single_pr = [[single_recall, single_precision, f"{args.model_name} AUC: {single_area:.3f}"], ]

        plot_pr_curve([], [], 0, args.data_name, curve_directory,
                      file_name=f"single_fold{count}", extra_pr=single_pr)

        print(f"Performing multi gen evaluation on fold {count}.")
        multi_test_output[count] = predict_multigen(model, test_pathways, thresholds, args)
        multi_recall, multi_precision, multi_area = sort_recall_precision(multi_test_output[count]["recall"],
                                                                          multi_test_output[count]["precision"])
        multi_pr = [[multi_recall, multi_precision, f"{args.model_name} AUC: {multi_area:.3f}"], ]
        plot_pr_curve([], [], 0, args.data_name, curve_directory, extra_pr=multi_pr,
                      file_name=f"multi_fold{count}")
        with open(f"{results_directory}/test_output_multi.json", "w") as out_file:
            json.dump(multi_test_output, out_file, indent=4)
        model.reset_model()

    # Plot mean PR curve for singlegen
    mean_single = mean_dicts(list(single_test_output.values()))
    single_sorted_recall, single_sorted_precision, single_area = sort_recall_precision(mean_single["recall"],
                                                                                       mean_single["precision"])
    single_pr = [[single_sorted_recall, single_sorted_precision, f"{args.model_name} AUC: {single_area:.3f}"], ]
    plot_pr_curve([], [], 0, args.data_name, curve_directory, file_name=f"single_mean",
                  extra_pr=single_pr)

    # Plot mean PR curve for multigen
    mean_multi = mean_dicts(list(multi_test_output.values()))
    multi_sorted_recall, multi_sorted_precision, multi_area = sort_recall_precision(mean_multi["recall"],
                                                                                    mean_multi["precision"])
    multi_pr = [[multi_sorted_recall, multi_sorted_precision, f"{args.model_name} AUC: {multi_area:.3f}"], ]
    plot_pr_curve([], [], 0, args.data_name, curve_directory, extra_pr=multi_pr,
                  file_name=f"multi_mean")
    with open(f"{results_directory}/mean_output.json", "w") as mean_file:
        json.dump({"single": mean_single, "multi": mean_multi}, mean_file)
    torch.cuda.empty_cache()
    return


if __name__ == "__main__":
    proc = subprocess.Popen(["java", "-jar", "java/envirule-2.6.0-jar-with-dependencies.jar"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    parser = ArgumentParser()
    parser.add_argument("data_name", type=str, default="soil", help="Which dataset to use, uspto or envipath")
    parser.add_argument("--model-name", type=str, default="EnviFormerModel",
                        help="Valid models include: EnviFormerModel")
    parser.add_argument("--tokenizer", type=str, default="regex", help="Style of tokenizer, regex")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum encoded length to consider")
    parser.add_argument("--min-len", type=int, default=0, help="Minimum encoded length to consider")
    parser.add_argument("--augment-count", type=int, default=-1, help="How much SMILES augmentation to do, -1 disables")
    parser.add_argument("--debug", action="store_true", default=False, help="Whether to set parameters for debugging")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size to target")
    parser.add_argument("--weights-dataset", type=str, default="", help="Pretrained weights based on what dataset")
    parser.add_argument("--test-mapping", action="store_true",
                        help="Whether to remove atom mapping before calculating accuracy")
    parser.add_argument("--score-all", action="store_true", help="Whether to group same reactants together")
    parser.add_argument("--run-clusters", action="store_true")
    parser.add_argument("--preprocessor", type=str, default="envipath",
                        help="Type of preprocessing to use, envipath or rdkit")
    arguments = parser.parse_args()
    if arguments.data_name == "multi":
        data_names = ["soil_add_sludge", "bbd_add_sludge", "sludge_add_soil", "sludge_add_bbd"]
        for name in data_names:
            arguments.data_name = name
            train_eval_single_multi(arguments)
    else:
        train_eval_single_multi(arguments)
    proc.kill()
