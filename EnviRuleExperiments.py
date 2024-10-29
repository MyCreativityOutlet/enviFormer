"""
EnviFormer a transformer based method for the prediction of biodegradation products and pathways
Copyright (C) 2024  Liam Brydon
Contact at: lbry121@aucklanduni.ac.nz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from sklearn.ensemble import RandomForestClassifier
from utils.EvalFunctions import predict_multigen, predict_singlegen
from utils.FormatData import *
from utils.EnviPathDownloader import check_envipath_data
from argparse import ArgumentParser
from EnviFormerExperiments import mean_dicts
from py4j.java_gateway import JavaGateway
import subprocess
import math
from utils.multilabelClassifiers import EnsembleClassifierChain
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pickle


class EnviRuleModel:

    def __init__(self, rules, dataset_name, model_name, fold, args, rdkit_reaction=False):
        self.classifier = None
        self.rules = rules
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.fold = fold
        self.args = args
        self.rdkit_reaction = rdkit_reaction
        clf_path = f"results/{self.model_name}/{self.dataset_name}/fold_{self.fold}_clf.txt"
        if os.path.exists(clf_path):
            file = open(clf_path, "rb")
            self.classifier = pickle.load(file)
            file.close()

    def train(self, train_reactions):
        # Train the ECC on the Rules
        good_count = 0
        clf_path = f"results/{self.model_name}/{self.dataset_name}/fold_{self.fold}_clf.txt"
        if self.classifier is None:
            x_train = []
            y_train = []
            for reaction in train_reactions:
                x = []
                y = []
                can_predict = False
                reactants, products = reaction.split(">>")
                reactant_one_mol = Chem.MolFromSmiles(reactants)
                x.extend(MACCSkeys.GenMACCSKeys(reactant_one_mol))
                true_products = set(mol for mol in products.split("."))
                for rule in self.rules:
                    pred_products = run_reaction(rule, reactants, return_smiles=True)
                    if len(pred_products) > 0:
                        x.append(1)
                        match = False
                        for candidate_products in pred_products:
                            candidate_products = candidate_products.split(".")
                            candidate_products = {canon_smile(c) for c in candidate_products}
                            pred_set = set(candidate_products)
                            if len(true_products - pred_set) == 0:
                                match = True
                        if match:
                            y.append(1)
                            can_predict = True
                        else:
                            y.append(0)
                    else:
                        x.append(0)
                        y.append(None)
                x_train.append(x)
                good_count += int(can_predict)
                y_train.append(y)

            x_train = np.array(x_train, dtype=float)
            y_train = np.array(y_train, dtype=float)
            base_classifier = RandomForestClassifier(n_jobs=1, max_depth=10, n_estimators=100)
            ecc_classifier = EnsembleClassifierChain(base_classifier, debug=self.args.debug)
            ecc_classifier.fit(x_train, y_train)
            self.classifier = ecc_classifier
            file = open(clf_path, 'wb')
            pickle.dump(ecc_classifier, file)
            file.close()
            print(f"The rules predicted correct products on {good_count} of {len(train_reactions)} reactions.")
        return

    def generate_rule_pred(self, reactants):
        x = []
        pred_y = []
        reactant_one_mol = Chem.MolFromSmiles(reactants)
        if reactant_one_mol is None:
            print(f"Couldn't pass {reactants} to rdkit for maccs")
            return False, x, pred_y
        x.extend(MACCSkeys.GenMACCSKeys(reactant_one_mol))
        for rule in self.rules:
            pred_products = run_reaction(rule, reactants, return_smiles=True)
            if len(pred_products) > 0:
                x.append(1)
                pred_y.append(pred_products)
            else:
                x.append(0)
                pred_y.append(None)
        return True, x, pred_y

    def smiles_to_smiles_inf(self, smiles, **kwargs):
        x_test = []
        y_test_pred = []
        bad_smiles = []
        # Multiprocessing doesn't work due to rdkit functions being Boost.Python which cannot be pickled, need n_jobs=1
        results = Parallel(n_jobs=1, verbose=2)(delayed(self.generate_rule_pred)(smile) for smile in smiles)
        for i, (success, x, pred_y) in enumerate(results):
            if success:
                x_test.append(x)
                y_test_pred.append(pred_y)
            else:
                bad_smiles.append(i)

        pred_smiles = []
        pred_proba = []
        if len(x_test) > 0:
            pred_rules = self.classifier.predict_proba(x_test)
            for i, y_pred in enumerate(y_test_pred):
                prediction = []
                probability = []
                for j in range(len(y_pred)):
                    if y_pred[j] is not None and pred_rules[i, j] > 0:
                        for product_set in y_pred[j]:
                            prediction.append(product_set)
                            probability.append(math.log(pred_rules[i, j]))
                pred_smiles.append(prediction)
                pred_proba.append(probability)
        for bad_i in bad_smiles:
            pred_smiles.insert(bad_i, [""])
            pred_proba.insert(bad_i, [float("-inf")])
        return pred_smiles, pred_proba


def main(args):
    transformer_base = "results/EnviFormerModel/"
    check_envipath_data()
    if args.data_name != "":
        transformer_folders = [args.data_name]
    else:
        # Add experiments to this line to run multiple experiments
        transformer_folders = ["soil"]
    transformer_folders = [transformer_base + t + "_regex/" for t in transformer_folders]

    method_name = "envipath" if args.existing_rules else "envirule"
    java_gateway = JavaGateway()
    for dataset in transformer_folders:
        dataset_name = dataset.split("/")[-2]
        args.data_name = "_".join(dataset_name.split("_")[:-1])
        os.makedirs(f"results/{method_name}/{dataset_name}", exist_ok=True)
        print(f"Processing {dataset}.")
        fold_splits = [file for file in os.listdir(dataset) if "splits" in file]
        multi_test_output = {}
        single_test_output = {}
        failed_reactions = {}
        thresholds = [t for t in get_thresholds().keys()]
        # Run enviRule on every fold for the current dataset
        if args.debug:
            fold_splits = fold_splits[:2]
        for i, fold in enumerate(tqdm(fold_splits, desc="Fold")):
            failed_reactions[i] = []
            with open(dataset + fold) as fold_file:
                fold_data = json.load(fold_file)
            train_data = []
            for smirk in fold_data["train"]:
                smirk = canon_smirk(smirk)
                if smirk is not None:
                    train_data.append(smirk)
            test_pathways = fold_data["test"]
            test_pathways = standardise_pathways(test_pathways)
            test_reactions = []
            for smirk in fold_data["test_reactions"]:
                smirk = canon_smirk(smirk)
                if smirk is not None:
                    test_reactions.append(smirk)
            os.makedirs(f"data/{method_name}/{dataset_name}", exist_ok=True)
            train_data_path = os.path.abspath(f"data/{method_name}/{dataset_name}/fold_{i}_train.txt")
            rules_path = f"results/{method_name}/{dataset_name}/fold_{i}_rules.txt"
            train_data_filtered = []
            # EnviRule needs the reactions saved in a line separated text file
            with open(train_data_path, "w") as file:
                for j, reaction in enumerate(train_data):
                    r, p = reaction.split(">>")
                    p_in_r = any(p == r1 for r1 in r.split("."))
                    if not p_in_r and "*" not in reaction:  # and any(p1 in r for p1 in p.split(".")):
                        file.write(reaction + "\n")
                        train_data_filtered.append(reaction)
                    else:
                        failed_reactions[i].append(reaction)
            train_data = train_data_filtered
            print(f"Ignoring {len(failed_reactions[i])} reactions with issues, dataset {dataset_name}.")
            if not os.path.exists(rules_path) and not args.existing_rules:
                generalizeIgnoreHydrogen = False
                radius = 1
                rules = java_gateway.entry_point.generateRules(train_data_path, generalizeIgnoreHydrogen, True, radius)
                rules = [rule for rule in rules]
                with open(rules_path, "w") as rule_file:
                    for rule in rules:
                        rule_file.write(rule + "\n")
            else:
                if args.existing_rules:
                    rules = []
                    # For the leave experiments these only work on leave_sludge
                    if "bbd" in dataset_name or "leave" in dataset_name:
                        for rule in get_raw_envipath("bbd")["rules"]:
                            if rule["identifier"] == "simple-rule":
                                rules.append(rule["smirks"])
                    if "soil" in dataset_name or "leave" in dataset_name:
                        for rule in get_raw_envipath("soil")["rules"]:
                            if rule["identifier"] == "simple-rule":
                                rules.append(rule["smirks"])
                    if len(rules) == 0:
                        raise ValueError("Length of existing rules is zero.")
                else:
                    with open(rules_path) as rule_file:
                        rules = [rule.strip() for rule in rule_file.readlines()]

            model = EnviRuleModel(rules, dataset_name, method_name, i, args)
            model.train(train_data)
            # Test the ECC on test_set single_gen
            if not args.skip_single:
                print("Testing singlegen performance")
                single_test_output[i] = predict_singlegen(model, test_reactions, thresholds)
                with open(f"results/{method_name}/{dataset_name}/test_output_single.json",
                          "w") as out_file:
                    json.dump(single_test_output, out_file, indent=4)
                sorted_recall, sorted_precision, area = sort_recall_precision(single_test_output[i]["recall"],
                                                                              single_test_output[i]["precision"])
                single_pr = [[sorted_recall, sorted_precision, f"{method_name} AUC: {area:.3f}"], ]
                plot_pr_curve([], [], 0, dataset_name, f"{method_name}/{dataset_name}", file_name=f"single_fold{i}",
                              extra_pr=single_pr)

            # Perform multigen evaluation
            if not args.skip_multi:
                print("Testing multigen performance")
                if args.debug:
                    test_pathways = test_pathways[:1]
                multi_test_output[i] = predict_multigen(model, test_pathways, thresholds, args)
                sorted_recall_e, sorted_precision_e, area_under_curve_e = sort_recall_precision(
                    multi_test_output[i]["recall"], multi_test_output[i]["precision"])
                multi_pr = [[sorted_recall_e, sorted_precision_e, f"{method_name} AUC: {area_under_curve_e:.3f}"], ]
                plot_pr_curve([], [], 0, dataset_name, f"{method_name}/{dataset_name}", extra_pr=multi_pr,
                              file_name=f"multi_fold{i}")
                with open(f"results/{method_name}/{dataset_name}/test_output_multi.json",
                          "w") as out_file:
                    json.dump(multi_test_output, out_file, indent=4)
        mean_output = {}
        # Plot mean PR curve for singlegen
        if not args.skip_single:
            mean_single = mean_dicts(list(single_test_output.values()))
            mean_output["single"] = mean_single
            single_sorted_recall, single_sorted_precision, single_area = sort_recall_precision(mean_single["recall"],
                                                                                               mean_single["precision"])
            single_pr = [[single_sorted_recall, single_sorted_precision, f"{method_name} AUC: {single_area:.3f}"], ]
            plot_pr_curve([], [], 0, dataset_name, f"{method_name}/{dataset_name}", file_name=f"single_mean",
                          extra_pr=single_pr)

        # Plot mean PR curve for multigen
        if not args.skip_multi:
            mean_multi = mean_dicts(list(multi_test_output.values()))
            mean_output["multi"] = mean_multi
            multi_sorted_recall, multi_sorted_precision, multi_area = sort_recall_precision(mean_multi["recall"],
                                                                                            mean_multi["precision"])
            multi_pr = [[multi_sorted_recall, multi_sorted_precision, f"{method_name} AUC: {multi_area:.3f}"], ]
            plot_pr_curve([], [], 0, dataset_name, f"{method_name}/{dataset_name}", extra_pr=multi_pr,
                          file_name=f"multi_mean")
        with open(f"results/{method_name}/{dataset_name}/mean_output.json", "w") as mean_file:
            json.dump(mean_output, mean_file)
    pass


if __name__ == "__main__":
    proc = subprocess.Popen(["java", "-jar", "java/envirule-2.6.0-jar-with-dependencies.jar"])
    parser = ArgumentParser()
    parser.add_argument("-ss", "--skip-single", action="store_true")
    parser.add_argument("-sm", "--skip-multi", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-er", "--existing-rules", action="store_true")
    parser.add_argument("--model-name", type=str, default="envirule")
    parser.add_argument("--data-name", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="regex")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as e:
        proc.kill()
        raise e
    proc.kill()
