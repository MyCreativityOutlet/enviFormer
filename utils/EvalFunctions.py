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
from collections import deque
from tqdm import tqdm
from utils.FormatData import plot_pr_curve, get_workers, sort_recall_precision
from nutree import Tree
from joblib import Parallel, delayed
import json


class Node:
    def __init__(self, smiles, depth, probability=float("-inf")):
        self.depth = depth
        self.probability = probability
        self.smiles = smiles


class Edge:
    def __init__(self, sources: set[Node], targets: set[Node]):
        self.sources = sources
        self.targets = targets


class Pathway:
    def __init__(self):
        self.edges = set()
        self.nodes = set()
        self.root_nodes = set()
        self.down_stream_nodes = {}

    @staticmethod
    def from_json(pathway_json, ignore_co2=True):
        """Returns a Pathway object from a json of a reaction tree.
        Notably the returned pathway can be considered a graph, although depth information may be incorrect.
        Depth information is fixed with self.get_pathway_with_depth()"""
        edges_list = pathway_json["links"]
        nodes_list = pathway_json["nodes"]
        co2 = {"O=C=O", "C(=O)=O"}
        pathway = Pathway()
        for edge in edges_list:
            source = nodes_list[edge["source"]]
            target = nodes_list[edge["target"]]
            while source["depth"] == -1:
                for edge2 in edges_list:
                    if nodes_list[edge2["target"]] == source:
                        source = nodes_list[edge2["source"]]
                        break
            while target["depth"] == -1:
                for edge2 in edges_list:
                    if nodes_list[edge2["source"]] == target:
                        target = nodes_list[edge2["target"]]
                        break
            sources = set()
            for smiles in source["smiles"].split("."):
                node = pathway.add_node(Node(smiles, source["depth"]))
                sources.add(node)
            if ignore_co2 and target["smiles"] in co2:
                continue
            targets = set()
            for smiles in target["smiles"].split("."):
                if ignore_co2 and smiles in co2:
                    continue
                node = pathway.add_node(Node(smiles, target["depth"]))
                targets.add(node)
            pathway.edges.add(Edge(sources, targets))
        pathway = pathway.get_pathway_with_depth()
        for node in pathway.nodes:
            pathway.down_stream_nodes[node] = pathway.get_down_stream_nodes(node)
        return pathway

    def add_node(self, node):
        new_node = node
        for n in self.nodes:  # Check if a Node instance with same smiles exists
            if new_node.smiles == n.smiles:
                n.depth = min(new_node.depth, n.depth)
                new_node = n
                break
        self.nodes.add(new_node)  # Set will not add if new_node instance already exists
        if new_node.depth == 0:
            self.root_nodes.add(new_node)
        return new_node

    def get_pathway_with_depth(self):
        """Recalculates depths in the pathway.
        Can fix incorrect depths from json parse if there were multiple nodes with the same SMILES at
        different depths that got merged."""
        new_pathway = Pathway()
        for node in self.nodes:
            new_pathway.add_node(node)
        new_pathway.edges = self.edges
        root_nodes = self.root_nodes
        current_depth = 0
        for node in new_pathway.nodes:
            if node in root_nodes:
                node.depth = current_depth
            else:
                node.depth = -99

        while new_pathway.assign_next_depth(current_depth):
            current_depth += 1
        return new_pathway

    def assign_next_depth(self, current_depth):
        new_assigned_nodes = False
        edges_pw_edges = self.edges
        all_pw_nodes = self.nodes

        current_depth_nodes = set()

        for node in all_pw_nodes:
            if node.depth == current_depth:
                current_depth_nodes.add(node)

        for edge in edges_pw_edges:
            reactants = edge.sources
            products = edge.targets

            for reactant in reactants:
                if reactant in current_depth_nodes:
                    for node in all_pw_nodes:
                        if node.depth < 0 and node in products:
                            node.depth = current_depth + 1
                            new_assigned_nodes = True
                    break

        return new_assigned_nodes

    def get_down_stream_nodes(self, node):
        if node not in self.down_stream_nodes:
            down_stream_nodes = set()

            for edge in self.edges:
                for start_node in edge.sources:
                    if start_node.smiles == node.smiles:
                        down_stream_nodes.update(edge.targets)
                        break
            self.down_stream_nodes[node] = down_stream_nodes
        return self.down_stream_nodes[node]

    def get_shortest_path(self, nodes_in_data, in_start_node, in_end_node):
        # Algorithm taken from here: https://www.geeksforgeeks.org/shortest-path-unweighted-graph/

        # Test if endNode is reachable from startNode
        # Get set of downStreamNodes for each node
        all_down_stream_nodes = {}
        for node in self.nodes:
            down_stream_nodes = self.get_down_stream_nodes(node)
            all_down_stream_nodes[node] = down_stream_nodes

        pred = {}
        start_node = None
        end_node = None

        if len(nodes_in_data) > 0:
            for pred_node, data_node in nodes_in_data.items():
                if data_node.smiles == in_start_node.smiles:
                    start_node = pred_node
                if data_node.smiles == in_end_node.smiles:
                    end_node = pred_node
                if start_node is not None and end_node is not None:
                    break
        else:
            start_node = in_start_node
            end_node = in_end_node

        if not is_reachable(all_down_stream_nodes, start_node, end_node, pred):
            return []

        # LinkedList to store path
        reverse_path = deque()
        crawl = end_node
        reverse_path.append(crawl)
        while pred.get(crawl) is not None:
            reverse_path.append(pred.get(crawl))
            crawl = pred.get(crawl)

        path = []

        for i in range(len(reverse_path) - 2, 0, -1):
            path.append(reverse_path[i])

        return path

    def get_depth_adjusted_pathway(self, intermediates):
        if len(intermediates) < 1:
            return self
        root_nodes = self.root_nodes
        root_nodes_smiles = {n.smiles for n in self.root_nodes}
        intermediates = {n.smiles for n in intermediates}

        dummy_node_map = {}

        for node in self.nodes:
            if node.smiles in root_nodes_smiles:
                continue

            if node.smiles in intermediates:
                node.depth = -99
            else:
                shortest_path_list = []
                max_size = 0

                for root_node in root_nodes:
                    shortest_path_nodes = self.get_shortest_path(dummy_node_map, root_node, node)
                    if shortest_path_nodes:
                        shortest_path_list.append(shortest_path_nodes)
                    max_size = max(max_size, len(shortest_path_nodes))

                if shortest_path_list:
                    shortest_index = 0
                    for i in range(len(shortest_path_list)):
                        if len(shortest_path_list[i]) < max_size:
                            max_size = len(shortest_path_list[i])
                            shortest_index = i

                    shortest_path_nodes = shortest_path_list[shortest_index]
                    num_ints = sum(1 for shortest_path_node in shortest_path_nodes if
                                   shortest_path_node.smiles in intermediates)
                    node.depth -= num_ints

        return self

    def set_pathway_eval_weight(self):
        # This method intends to replace the weight assignment functionality in compare_pathways
        # So that it can work with merge_pathways
        # This method intends to be applied to raw pathways before they are merged

        node_eval_weights = {}

        for node in self.nodes:
            # Scale score according to depth level
            node_eval_weights[node] = 1 / (2 ** node.depth)

        return node_eval_weights


def compare_pathways(data_pathway, pred_pathway, ignore_co2=True):
    """Compare two pathways for multi-gen evaluation.
    It is assumed the smiles in both pathways have been standardised in the same manner."""
    data_pathway_with_depth = Pathway.from_json(data_pathway, ignore_co2)
    pred_pathway_with_depth = Pathway.from_json(pred_pathway, ignore_co2)

    start_and_end_nodes = find_intermediates(data_pathway_with_depth, pred_pathway_with_depth)
    intermediates = {n for n in start_and_end_nodes}

    if intermediates:
        pred_pathway_with_depth = pred_pathway_with_depth.get_depth_adjusted_pathway(intermediates)

    test_pathway_eval_weights = data_pathway_with_depth.set_pathway_eval_weight()
    pred_pathway_eval_weights = pred_pathway_with_depth.set_pathway_eval_weight()

    common_nodes = get_common_nodes(pred_pathway_with_depth, data_pathway_with_depth)

    common_nodes_in_data = set()
    common_nodes_in_pred = set()

    for pred_node, data_node in common_nodes.items():
        common_nodes_in_data.add(data_node)
        common_nodes_in_pred.add(pred_node)

    data_only_nodes = get_unique_nodes(data_pathway_with_depth, common_nodes_in_data)
    pred_only_nodes = get_unique_nodes(pred_pathway_with_depth, common_nodes_in_pred)

    score_test_TP = 0.0
    score_FP = 0.0
    score_FN = 0.0
    final_score = 0.0
    precision = 0.0
    recall = 0.0

    pred_only_remove = set()
    for node in pred_only_nodes:
        if check_if_same_edge(node, common_nodes_in_pred, pred_pathway_with_depth):
            pred_only_remove.add(node)
    for n in pred_only_remove:
        pred_only_nodes.discard(n)

    for pred_node, data_node in common_nodes.items():
        if pred_node.depth > 0:
            score_test_TP += pred_pathway_eval_weights[pred_node]

    for node in data_only_nodes:
        if node.depth > 0:
            score_FN += test_pathway_eval_weights[node]

    for node in pred_only_nodes:
        if node.depth > 0:
            score_FP += pred_pathway_eval_weights[node]

    if (score_test_TP + score_FP + score_FN) > 0:
        final_score = score_test_TP / (score_test_TP + score_FP + score_FN)
    if (score_test_TP + score_FP) > 0:
        precision = score_test_TP / (score_test_TP + score_FP)
    if (score_test_TP + score_FN) > 0:
        recall = score_test_TP / (score_test_TP + score_FN)
    return final_score, precision, recall


def find_intermediates(data_pathway_with_depth, pred_pathway_with_depth):
    data_pathway_nodes = data_pathway_with_depth.nodes
    common_nodes = get_common_nodes(pred_pathway_with_depth, data_pathway_with_depth)

    start_and_end_nodes = {}

    for pred_node, data_node in common_nodes.items():
        down_stream_nodes = data_pathway_with_depth.get_down_stream_nodes(data_node)

        for down_stream_node in down_stream_nodes:
            for pred_ds_node, data_ds_node in common_nodes.items():  # Check downstream node is also a common node
                if down_stream_node.smiles == data_ds_node.smiles:
                    all_ints = pred_pathway_with_depth.get_shortest_path(common_nodes, data_node, down_stream_node)

                    for int_node in all_ints:
                        if not check_if_node_in_set(int_node, data_pathway_nodes):
                            start_and_end_nodes[int_node] = {data_node: down_stream_node}
                    break
    return start_and_end_nodes


def get_common_nodes(pred_pathway, data_pathway):
    common_nodes = {}

    for node in data_pathway.nodes:
        is_pathway_root_node = check_if_node_in_set(node, data_pathway.root_nodes)
        is_this_root_node = check_if_node_in_set(node, pred_pathway.root_nodes)

        for pred_node in pred_pathway.nodes:
            if node.smiles == pred_node.smiles:  # and (node.depth <= pred_node.depth or not check_depth):
                if is_pathway_root_node is False and is_this_root_node is False:
                    common_nodes[pred_node] = node
                elif is_pathway_root_node and is_this_root_node:
                    common_nodes[pred_node] = node

    return common_nodes


def is_reachable(all_down_stream_nodes, start_node, end_node, pred):
    queue = deque()
    visited_nodes = {start_node.smiles}
    queue.append(start_node)
    downstream_items = list(all_down_stream_nodes.items())
    # bfs Algorithm
    while queue:
        u = queue.popleft()
        u_smiles = u.smiles
        for ds_node, ds_nodes_set in downstream_items:
            if ds_node.smiles == u_smiles:
                for down_stream_node in ds_nodes_set:
                    # SMILES is used to avoid inefficient check_if_node_in_set call
                    if down_stream_node.smiles not in visited_nodes:
                        visited_nodes.add(down_stream_node.smiles)
                        pred[down_stream_node] = u
                        queue.append(down_stream_node)
                        if down_stream_node.smiles == end_node.smiles:
                            return True
                break
    return False


def get_unique_nodes(pw, common_nodes):
    pathway_nodes = pw.nodes
    unique_nodes = set()

    for node in pathway_nodes:
        if check_if_node_in_set(node, common_nodes) is False:
            unique_nodes.add(node)

    return unique_nodes


def check_if_node_in_set(node, node_set: set):
    for n in node_set:
        if n.smiles == node.smiles:
            return True
    return False


def check_if_same_edge(node, common_nodes, pw):
    for common_node in common_nodes:
        for edge in pw.edges:
            sum_index = 0
            for product_node in edge.targets:
                if product_node.smiles == common_node.smiles:
                    sum_index += 1
                if product_node.smiles == node.smiles:
                    sum_index += 1
            if sum_index == 2:
                return True
    return False


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


def predict_multigen(model, test_pathways, thresholds, args, evaluate=True, max_depth=7, max_width=-1):
    thresholds = {t: [] for t in thresholds}
    min_threshold = min(thresholds.keys())
    co2 = {"O=C=O", "C(=O)=O"}
    multi_test_output = {"predictions": [], "recall": [], "precision": []}
    if args.debug:
        test_pathways = test_pathways[:3]
    for pathway in tqdm(test_pathways, desc="Testing multigen"):
        print(f"Predicting pathway: {pathway['name']}")
        tree = Tree(pathway["name"], calc_data_id=_calc_id)
        root_nodes = []
        for node in pathway["nodes"]:
            if node["depth"] == 0:
                root_nodes.append(node["smiles"])
        if len(root_nodes) == 0:
            print(f"Can't find root for pathway {pathway['name']}, skipping pathway")
            continue
        for node in root_nodes:
            tree.add(Node(node, 0, 0))
        input_mols = root_nodes
        for d in range(1, max_depth):
            # print(f"Depth {d}. Performing {len(input_mols)} predictions.")
            output_mols, output_probabilities = model.smiles_to_smiles_inf(input_mols, num_beams=5)
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
                    if input_mols[i]not in sorted_layer:
                        sorted_layer[input_mols[i]] = []
                    sorted_layer[input_mols[i]].append([output_mols[i][j], output_probabilities[i][j]])
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

            next_layer = []
            for input_mol, output_mol, output_probability in zip(input_mols, output_mols, output_probabilities):
                input_node = tree.find_all(data_id=input_mol)
                input_node = max(input_node, key=lambda x: x.data.depth)
                input_path = input_node.get_parent_list(add_self=True)
                for mol_smiles, mol_probability in zip(output_mol, output_probability):
                    conditional_probability = mol_probability
                    for node in input_path:
                        conditional_probability += node.data.probability
                    for mol in mol_smiles.split("."):
                        mol = mol.replace("*", "")  # enviPath SMILES standardisation sometimes add erroneous asterix
                        if conditional_probability >= min_threshold and len(
                                mol) > 0 and mol not in co2 and mol not in next_layer:
                            next_layer.append(mol)
                            input_node.add(Node(mol, d, mol_probability))
            if len(next_layer) == 0 or len(tree) > 1000:
                break
            input_mols = next_layer
        multi_test_output["predictions"].append([pathway, tree])
    if evaluate:
        recall, precision = process_compare_pathways(args, multi_test_output, thresholds, plot_curve=False)
        multi_test_output["precision"] = precision
        multi_test_output["recall"] = recall
    return multi_test_output


def _calc_id(tree, data):
    if isinstance(data, Node):
        return data.smiles
    return hash(data)


def serialize_mapper(node, data):
    data["smiles"] = node.data.smiles
    data["depth"] = node.data.depth
    data["probability"] = node.data.probability
    return data


def deserialize_mapper(parent, data):
    data = Node(data["smiles"], data["depth"], data["probability"])
    return data


def process_compare_pathways(args, test_output, thresholds, plot_curve=True):
    # Process the predicted pathway tree with varying thresholds
    for i, (true, pred) in enumerate(test_output["predictions"]):
        test_output["predictions"][i][1] = pred.to_dict_list(mapper=serialize_mapper)
    results = Parallel(n_jobs=get_workers(args.debug), verbose=2)(delayed(pathway_analysis)(test_output, threshold)
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
                      f"{args.model_name}/{args.data_name}_{args.tokenizer}", "multigen")
    return recall, precision


def pathway_analysis(test_output, threshold, ignore_co2=True):
    scores = []
    for pathway in test_output["predictions"]:
        true, predicted_tree = pathway
        predicted_tree = Tree.from_dict(predicted_tree, mapper=deserialize_mapper)
        for root in predicted_tree.children:
            prune_tree(root, 0, threshold)
        new_links = []
        new_nodes = []
        for root_node in predicted_tree.children:
            root_smiles = root_node.data.smiles
            new_node = {"depth": 0, "smiles": root_smiles}
            new_nodes.append(new_node)
            path = [root_smiles]
            for node in root_node:  # Depth first is the default iterator for nutree package
                while path[-1] != node.parent.data.smiles:
                    path.pop()
                node_smiles = node.data.smiles
                new_node = {"depth": len(path), "smiles": node_smiles}
                pred_react = node.parent.data.smiles + ">>" + node_smiles
                source = None
                for i, n in enumerate(new_nodes):
                    if n["smiles"] == node.parent.data.smiles and n["depth"] == len(path) - 1:
                        # Source (or reactant) for the new node is the node with the same smiles
                        # as the parent and with depth one higher
                        source = i
                        break
                new_link = {"smirks": pred_react, "source": source, "target": len(new_nodes)}
                new_links.append(new_link)
                new_nodes.append(new_node)
                path.append(node.data.smiles)
        predicted = {"nodes": new_nodes, "links": new_links}
        # analysis = mainAnalysis(true, predicted)
        # p_way_score = analysis.runAnalysisLocal()
        p_way_score = compare_pathways(true, predicted, ignore_co2)
        scores.append(p_way_score)
    return threshold, scores


def prune_tree(node, log_prob_so_far, threshold):
    # Add the log probability of the current node to the running total
    total_log_prob = log_prob_so_far + node.data.probability

    # If the total log probability is lower than the threshold, prune the node
    if total_log_prob < threshold:
        node.remove()
        return None

    # Otherwise, recursively prune the children
    for child in node.children:
        prune_tree(child, total_log_prob, threshold)

    return node


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
        results = compare_pathways(true_path, pred_path)
        results = [round(r, 4) for r in results]
        for i, r in enumerate(results):
            pass_metric = "PASS" if r == expected_results[i] else "FAILED"
            print(f"{score_type[i]}: {pass_metric}")
            if pass_metric == "FAILED":
                print(f"Expected {score_type[i]} to be {expected_results[i]}, got {results[i]}")


if __name__ == "__main__":
    multi_gen_examples()
