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
from tqdm import tqdm
from utils.FormatData import plot_pr_curve, get_workers, sort_recall_precision
import networkx as nx
from joblib import Parallel, delayed
import json
import os


def get_shortest_path(pathway, in_start_node, in_end_node):
    try:
        pred = nx.shortest_path(pathway, source=in_start_node, target=in_end_node)
    except nx.NetworkXNoPath:
        return []
    pred.remove(in_start_node)
    pred.remove(in_end_node)
    return pred


def set_pathway_eval_weight(pathway):
    node_eval_weights = {}
    for node in pathway.nodes:
        # Scale score according to depth level
        node_eval_weights[node] = 1 / (2 ** pathway.nodes[node]["depth"]) if pathway.nodes[node]["depth"] >= 0 else 0
    return node_eval_weights


def get_depth_adjusted_pathway(data_pathway, pred_pathway, intermediates):
    if len(intermediates) < 1:
        return pred_pathway
    root_nodes = pred_pathway.graph["root_nodes"]
    for node in pred_pathway.nodes:
        if node in root_nodes:
            continue
        if node in intermediates and node not in data_pathway:
            pred_pathway.nodes[node]["depth"] = -99
        else:
            shortest_path_list = []
            for root_node in root_nodes:
                shortest_path_nodes = get_shortest_path(pred_pathway, root_node, node)
                if shortest_path_nodes:
                    shortest_path_list.append(shortest_path_nodes)

            if shortest_path_list:

                shortest_path_nodes = min(shortest_path_list, key=len)
                num_ints = sum(1 for shortest_path_node in shortest_path_nodes if
                               shortest_path_node in intermediates)
                pred_pathway.nodes[node]["depth"] -= num_ints
    return pred_pathway


def assign_next_depth(pathway, current_depth):
    new_assigned_nodes = False
    current_depth_nodes = {n for n in pathway.nodes if pathway.nodes[n]["depth"] == current_depth}
    for node in current_depth_nodes:
        successors = pathway.successors(node)
        for s in successors:
            if pathway.nodes[s]["depth"] < 0:
                pathway.nodes[s]["depth"] = current_depth + 1
                new_assigned_nodes = True
    return new_assigned_nodes


def get_pathway_with_depth(pathway):
    """Recalculates depths in the pathway.
    Can fix incorrect depths from json parse if there were multiple nodes with the same SMILES at
    different depths that got merged."""
    current_depth = 0
    for node in pathway.nodes:
        if node in pathway.graph["root_nodes"]:
            pathway.nodes[node]["depth"] = current_depth
        else:
            pathway.nodes[node]["depth"] = -99

    while assign_next_depth(pathway, current_depth):
        current_depth += 1
    return pathway


def initialise_pathway(pathway):
    pathway.graph["root_nodes"] = {n for n in pathway.nodes if pathway.nodes[n]["depth"] == 0}
    pathway = get_pathway_with_depth(pathway)
    return pathway


def compare_pathways(data_pathway, pred_pathway):
    """Compare two pathways for multi-gen evaluation.
    It is assumed the smiles in both pathways have been standardised in the same manner.
    Requires the input pathways to be Networkx graphs,
    this can be created from the graph_from_envipath or graph_from_serializable functions"""
    data_pathway = initialise_pathway(data_pathway)
    pred_pathway = initialise_pathway(pred_pathway)
    intermediates = find_intermediates(data_pathway, pred_pathway)

    if intermediates:
        pred_pathway = get_depth_adjusted_pathway(data_pathway, pred_pathway, intermediates)

    test_pathway_eval_weights = set_pathway_eval_weight(data_pathway)
    pred_pathway_eval_weights = set_pathway_eval_weight(pred_pathway)

    common_nodes = get_common_nodes(pred_pathway, data_pathway)

    data_only_nodes = set(n for n in data_pathway.nodes if n not in common_nodes)
    pred_only_nodes = set(n for n in pred_pathway.nodes if n not in common_nodes)

    score_TP, score_FP, score_FN, final_score, precision, recall = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for node in common_nodes:
        if pred_pathway.nodes[node]["depth"] > 0:
            score_TP += test_pathway_eval_weights[node]

    for node in data_only_nodes:
        if data_pathway.nodes[node]["depth"] > 0:
            score_FN += test_pathway_eval_weights[node]

    for node in pred_only_nodes:
        if pred_pathway.nodes[node]["depth"] > 0:
            score_FP += pred_pathway_eval_weights[node]

    final_score = score_TP / denom if (denom := score_TP + score_FP + score_FN) > 0 else 0.0
    precision = score_TP / denom if (denom := score_TP + score_FP) > 0 else 0.0
    recall = score_TP / denom if (denom := score_TP + score_FN) > 0 else 0.0
    return final_score, precision, recall, intermediates


def find_intermediates(data_pathway, pred_pathway):
    # data_pathway_nodes = data_pathway.nodes
    common_nodes = get_common_nodes(pred_pathway, data_pathway)
    intermediates = set()
    for node in common_nodes:
        down_stream_nodes = data_pathway.successors(node)
        for down_stream_node in down_stream_nodes:
            if down_stream_node in pred_pathway:
                all_ints = get_shortest_path(pred_pathway, node, down_stream_node)
                # intermediates.update(i for i in all_ints if i not in data_pathway_nodes)
                intermediates.update(all_ints)
    return intermediates


def get_common_nodes(pred_pathway, data_pathway):
    common_nodes = set()
    for node in data_pathway.nodes:
        is_pathway_root_node = node in data_pathway.graph["root_nodes"]
        is_this_root_node = node in pred_pathway.graph["root_nodes"]
        if node in pred_pathway.nodes:
            if is_pathway_root_node is False and is_this_root_node is False:
                common_nodes.add(node)
            elif is_pathway_root_node and is_this_root_node:
                common_nodes.add(node)
    return common_nodes


def predict_singlegen(model, test_reactions, thresholds):
    """Using a model with a smiles_to_smiles_inf function calculate the single gen model performance on the given
    test_reactions. It is assumed that the model output and test reactions have been standardised in the same manner."""
    correct = {t: 0 for t in thresholds}
    predicted = {t: 0 for t in thresholds}
    predictions, probabilities = model.smiles_to_smiles_inf([r.split(">>")[0] for r in test_reactions])
    true_dict = {}
    for r in test_reactions:
        reactant, true_product_set = r.split(">>")
        true_product_set = {p for p in true_product_set.split(".")}
        if reactant not in true_dict:
            true_dict[reactant] = []
        true_dict[reactant].append(true_product_set)
    pred_dict = {}
    assert len(test_reactions) == len(predictions)
    for k, (pred_smiles, pred_proba) in enumerate(zip(predictions, probabilities)):
        reactant, true_product = test_reactions[k].split(">>")
        if reactant not in pred_dict:
            pred_dict[reactant] = {"predict": [], "scores": []}
        for smiles, proba in zip(pred_smiles, pred_proba):
            smiles = set(smiles.split("."))
            if smiles not in pred_dict[reactant]["predict"]:
                pred_dict[reactant]["predict"].append(smiles)
                pred_dict[reactant]["scores"].append(proba)
    for threshold in correct:
        for reactant, product_sets in true_dict.items():
            pred_smiles = pred_dict[reactant]["predict"]
            pred_scores = pred_dict[reactant]["scores"]
            pred_smiles = [s for i, s in enumerate(pred_smiles) if pred_scores[i] > threshold]
            predicted[threshold] += len(pred_smiles)
            for true_set in product_sets:
                for pred_set in pred_smiles:
                    if len(true_set - pred_set) == 0:
                        correct[threshold] += 1
                        break
    recall = {k: v / len(test_reactions) for k, v in correct.items()}
    precision = {k: v / predicted[k] if predicted[k] > 0 else 0 for k, v in correct.items()}
    save_predictions = {}
    for reactant, true_products in true_dict.items():
        pred_smiles = pred_dict[reactant]["predict"]
        pred_smiles = [".".join(p) for p in pred_smiles]
        pred_scores = pred_dict[reactant]["scores"]
        true_products = [".".join(t) for t in true_products]
        save_predictions[reactant] = {"actual": true_products, "predict": pred_smiles, "scores": pred_scores}
    single_test_output = {"predictions": save_predictions, "recall": recall, "precision": precision}
    return single_test_output


def predict_multigen(model, test_pathways, thresholds, args, evaluate=True, max_depth=7, max_width=50):
    thresholds = {t: [] for t in thresholds}
    min_threshold = min(thresholds.keys())
    co2 = {"O=C=O", "C(=O)=O"}
    multi_test_output = {"predictions": [], "recall": [], "precision": []}
    if args.debug:
        test_pathways = test_pathways[:2]
    for pathway in tqdm(test_pathways, desc="Testing multigen"):
        # print(f"Predicting pathway: {pathway['name']}")
        graph = nx.DiGraph(name=pathway['name'])
        root_nodes = []
        max_depth = pathway.get("max_depth", max_depth)
        for node in pathway["nodes"]:
            if node["depth"] == 0:
                root_nodes.append(node["smiles"])
        if len(root_nodes) == 0:
            print(f"Can't find root for pathway {pathway['name']}, skipping pathway")
            continue
        if len(root_nodes) > 1:
            print(f"More than one root node for pathway {pathway['name']}, skipping")
            continue
        for node in root_nodes:
            graph.add_node(node, depth=0, probability=0.0, root=True)
        input_mols = root_nodes
        for d in range(1, max_depth):
            next_layer = []
            output_mols, output_probabilities = model.smiles_to_smiles_inf(
                input_mols, num_beams=5)

            if max_width > 0:
                flat_indices = []
                flat_probabilities = []
                for i, probs in enumerate(output_probabilities):
                    for j, prob in enumerate(probs):
                        flat_indices.append([i, j])
                        flat_probabilities.append(prob)
                flat_indices = sorted(zip(flat_indices, flat_probabilities), key=lambda x: x[1], reverse=True)
                sorted_layer = {}
                for indices, _ in flat_indices[:max_width]:
                    i, j = indices
                    sorted_layer[input_mols[i]] = sorted_layer.setdefault(input_mols[i], []) + [[output_mols[i][j], output_probabilities[i][j]]]
                sorted_input, sorted_output, sorted_probs = [], [], []
                for in_mol, pred in sorted_layer.items():
                    sorted_input.append(in_mol)
                    out = []
                    prob = []
                    for p in pred:
                        out.append(p[0])
                        prob.append(p[1])
                    sorted_output.append(out)
                    sorted_probs.append(prob)
                input_mols, output_mols, output_probabilities = sorted_input, sorted_output, sorted_probs

            for input_mol, output_mol, output_probability in zip(input_mols, output_mols, output_probabilities):
                if input_mol not in graph:
                    continue

                for mol_smiles, mol_probability in zip(output_mol, output_probability):
                    conditional_probability = graph.nodes[input_mol]['probability'] + mol_probability

                    for mol in mol_smiles.split("."):
                        mol = mol.replace("*", "")  # Standardize SMILES
                        if conditional_probability >= min_threshold and mol not in co2:
                            if mol not in graph:
                                # Add new molecule to the graph
                                graph.add_node(mol, depth=d, probability=conditional_probability)
                                next_layer.append(mol)
                            # Add edge from input molecule to product
                            if input_mol != mol:
                                graph.add_edge(input_mol, mol, edge_probability=mol_probability)

            if len(next_layer) == 0 or len(graph.nodes) > 1000:
                break
            input_mols = next_layer
        multi_test_output["predictions"].append([pathway, graph])
    if evaluate:
        recall, precision = process_compare_pathways(args, multi_test_output, thresholds, plot_curve=False)
        multi_test_output["precision"] = precision
        multi_test_output["recall"] = recall
    return multi_test_output


def graph_to_serializable(graph):
    """
    Convert a NetworkX graph to a serializable dictionary containing both node and edge attributes.
    """
    return {
        "nodes": {node: graph.nodes[node] for node in graph.nodes()},
        "edges": nx.to_dict_of_dicts(graph)
    }


def graph_from_serializable(data):
    """
    Recreate a NetworkX graph from a serializable dictionary.
    """
    graph = nx.from_dict_of_dicts(data["edges"], create_using=nx.DiGraph)
    nx.set_node_attributes(graph, data["nodes"])
    return graph


def graph_from_envipath(data):
    graph = nx.DiGraph(name=data['name'])
    nodes = data["nodes"]
    co2 = {"O=C=O", "C(=O)=O"}
    for link in data["links"]:
        source = nodes[link["source"]]
        target = nodes[link["target"]]
        if source["smiles"] not in graph:
            graph.add_node(source["smiles"], depth=source["depth"])
        else:
            graph.nodes[source["smiles"]]["depth"] = min(source["depth"], graph.nodes[source["smiles"]]["depth"])
        if target["smiles"] not in graph and target["smiles"] not in co2:
            graph.add_node(target["smiles"], depth=target["depth"])
        elif target["smiles"] not in co2:
            graph.nodes[target["smiles"]]["depth"] = min(target["depth"], graph.nodes[target["smiles"]]["depth"])
        if target["smiles"] not in co2 and target["smiles"] != source["smiles"]:
            graph.add_edge(source["smiles"], target["smiles"])
    return graph


def process_compare_pathways(args, test_output, thresholds, plot_curve=True):
    # Process the predicted pathway tree with varying thresholds
    for i, (true, pred) in enumerate(test_output["predictions"]):
        test_output["predictions"][i][1] = graph_to_serializable(pred)
    results = Parallel(n_jobs=get_workers(args.debug), verbose=2)(delayed(pathway_analysis)(test_output, threshold, args)
                                                                  for threshold in thresholds)
    for threshold, result in results:
        thresholds[threshold] = result
    precision = {}
    recall = {}
    for threshold in thresholds:
        precision_sum = []
        recall_sum = []
        for result in thresholds[threshold]:
            precision_sum.append(result[1])
            recall_sum.append(result[2])
        precision_sum = sum(precision_sum) / len(precision_sum)
        recall_sum = sum(recall_sum) / len(recall_sum)
        precision[threshold] = precision_sum
        recall[threshold] = recall_sum

    if plot_curve:
        sorted_recall, sorted_precision, area_under_curve = sort_recall_precision(recall, precision)
        plot_pr_curve(sorted_recall, sorted_precision, area_under_curve, f"{args.data_name} MultiGen",
                      f"{args.model_name}/{args.data_name}_{args.preprocessor}", "multigen")
    return recall, precision


def pathway_analysis(test_output, threshold, args):
    scores = []
    plot_path = f"results/{args.model_name}/{args.data_name}_{args.tokenizer}/p_plots"
    os.makedirs(plot_path, exist_ok=True)
    for pathway in test_output["predictions"]:
        true, predicted_graph_dict = pathway
        # Convert the predicted tree to a NetworkX directed graph
        predicted_graph = graph_from_serializable(predicted_graph_dict)
        true_graph = graph_from_envipath(true)

        # Prune nodes below the threshold
        prune_graph(predicted_graph, threshold)
        p_way_score, intermediates = compare_pathways(true_graph, predicted_graph)
        scores.append(p_way_score)

    return threshold, scores


def prune_graph(graph, threshold):
    """
    Prunes nodes in the graph based on their precomputed log probabilities.
    Nodes below the threshold are removed.
    """
    while True:
        try:
            cycle = nx.find_cycle(graph)
            graph.remove_edge(*cycle[-1])  # Remove the last edge in the cycle
        except nx.NetworkXNoCycle:
            break

    for node in list(graph.nodes):
        # Check if the node's log probability meets the threshold
        if graph.nodes[node]["probability"] < threshold:  # / max(graph.nodes[node]["depth"], 1):
            graph.remove_node(node)
    for node in list(nx.isolates(graph)):
        if graph.nodes[node]["depth"] != 0:
            graph.remove_node(node)
    root_node = [n for n in graph.nodes if "root" in graph.nodes[n]][0]
    descendants = nx.descendants(graph, root_node)
    descendants.add(root_node)
    for node in list(graph.nodes):
        if node not in descendants:
            graph.remove_node(node)


def multi_gen_examples():
    with open("data/multigen_test_cases.json") as file:
        test_cases = json.load(file)
    score_type = ["Accuracy", "Precision", "Recall"]
    cases = list(test_cases.items())
    for case, data in cases:
        print(f"Testing case {case}")
        true_path = data["true_path"]
        pred_path = data["pred_path"]
        expected_results = data["results"]
        results, intermediates = compare_pathways(true_path, pred_path)
        results = [round(r, 4) for r in results]
        for i, r in enumerate(results):
            pass_metric = "PASS" if r == expected_results[i] else "FAILED"
            print(f"{score_type[i]}: {pass_metric}")
            if pass_metric == "FAILED":
                print(f"Expected {score_type[i]} to be {expected_results[i]}, got {results[i]}")


if __name__ == "__main__":
    multi_gen_examples()
