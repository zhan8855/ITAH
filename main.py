import math
import random
import sapiens
import heapq
import pickle
from dataclasses import dataclass
from argparse import ArgumentParser
from tqdm import tqdm

amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--method", default="greedy", choices=["greedy", "mcts", "itah"])
    parser.add_argument("--data_path", default="Hu-mAb_Results.csv", type=str)
    parser.add_argument("--output_path", default="predictions", type=str)
    parser.add_argument("--graft", default="cdr", choices=["cdr", "vernier"])
    parser.add_argument("--dedup", default=0, type=int)
    parser.add_argument("--sort", default=0, type=int)
    parser.add_argument("--step", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)

def calc_lm_score(chain, sequence):
    scores = sapiens.predict_scores(sequence, chain)
    log_prod = sum([math.log(scores.iloc[idx][char]) for idx, char in enumerate(sequence)])
    return log_prod

class MCTSNode:
    def __init__(self, chain, sequence, graft, parent=None, dedup=False, sort=False):
        self.chain = chain
        self.sequence = sequence
        self.graft = graft
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.dedup = dedup
        self.sort = sort

        scores = sapiens.predict_scores(sequence, chain)

        self.score = 0
        self.candidates = []
        for idx, candidate_scores in scores.iterrows():
            char = sequence[idx]
            original_score = candidate_scores[char]
            self.score = self.score + math.log(original_score)
            if not self.graft[idx]:
                self.candidates.extend([(idx, new_char, math.log(candidate_scores[new_char]) - math.log(original_score))
                                        for new_char, score in candidate_scores.items() if new_char != char])
        if sort:
            self.candidates = sorted(self.candidates, key=lambda candidate: candidate[2])
        else:
            random.shuffle(self.candidates)
        self.iter = len(self.candidates) - 1

    def is_fully_expanded(self):
        return self.iter == -1

    def best_child(self, exploration_weight):
        return max(self.children, key=lambda child: child.value / child.visits + exploration_weight * math.sqrt(math.log(self.visits) / child.visits))

    def expand(self):
        idx, new_char, score = self.candidates[self.iter]
        new_sequence = list(self.sequence)
        new_sequence[idx] = new_char
        new_sequence = "".join(new_sequence)
        new_graft = list(self.graft)
        if self.dedup:
            new_graft[idx] = 1
        new_node = MCTSNode(
            chain=self.chain,
            sequence=new_sequence,
            graft=new_graft,
            parent=self,
            dedup=self.dedup,
            sort=self.sort
        )
        self.children.append(new_node)
        self.iter = self.iter - 1
        return new_node

    def update(self, reward):
        self.visits = self.visits + 1
        self.value = self.value + reward

class MCTS:
    def __init__(self, exploration_weight=0.001):
        self.exploration_weight = exploration_weight
    
    def select(self, node):
        while node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node.expand()

    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

def greedy_predictor(chain, sequence, graft_region, step, dedup, sort):
    original_sequence = sequence
    rewards = []
    best_reward = 0
    for step_idx in range(step):
        new_sequence = sapiens.predict_best_score(sequence, chain)
        sequence = "".join([sequence[idx] if graft_region[idx] else new_sequence[idx] for idx in range(len(sequence))])
        reward = calc_lm_score(chain, sequence) - calc_lm_score(chain, original_sequence)
        if reward > best_reward:
            best_reward = reward
            best_sequence = sequence
        rewards.append(best_reward)
    return best_sequence, rewards

def mcts_predictor(chain, sequence, graft_region, step, dedup, sort):
    mcts = MCTS()
    root = MCTSNode(
        chain=chain,
        sequence=sequence,
        graft=graft_region,
        dedup=dedup,
        sort=sort
    )
    rewards = []
    best_reward = 0
    best_node = root
    init_score = root.score
    for idx in tqdm(range(step)):
        node = mcts.select(root)
        reward = node.score - root.score
        mcts.backpropagate(node, reward)
        if reward > best_reward:
            best_reward = reward
            best_node = node
        rewards.append(best_reward)
    return best_node.sequence, rewards

def itah_predictor(chain, sequence, graft_region, step, dedup, sort):
    root = MCTSNode(
        chain=chain,
        sequence=sequence,
        graft=graft_region,
        dedup=dedup,
        sort=sort
    )
    node_list = [root]
    node_dict = {root.sequence: 1}
    node_size = 0
    rewards = []
    best_reward = 0
    best_node = root
    heap = []
    heapq.heappush(heap, (-(root.score + root.candidates[-1][2]), 0))
    for idx in tqdm(range(step)):
        score, node_idx = heapq.heappop(heap)
        node = node_list[node_idx]
        reward = node.score - root.score
        if reward > best_reward:
            best_reward = reward
            best_node = node
        rewards.append(best_reward)

        new_node = node.expand()
        if (new_node.sequence not in node_dict) or (not new_node.dedup):
            node_dict[new_node.sequence] = 1
            node_list.append(new_node)
            node_size = node_size + 1
            heapq.heappush(heap, (-(new_node.score + new_node.candidates[-1][2]), node_size))
        if not node.is_fully_expanded():
            heapq.heappush(heap, (-(node.score + node.candidates[node.iter][2]), node_idx))
    return best_node.sequence, rewards

@dataclass
class AntibodyData:
    index: list
    sequence: str
    label: str
    cdr_region: list
    vernier_region: list

class AntibodyDataset:
    def __init__(self, args):
        self.args = args
        self.dataset = {"H": [], "L": []}

        if args.method == "greedy":
            self.predictor = greedy_predictor
        elif args.method == "mcts":
            self.predictor = mcts_predictor
        elif args.method == "itah":
            self.predictor = itah_predictor
        else:
            raise NotImplementedError

    def append(self, chain, lines):
        index = lines[0].split(",")[1:-1]
        sequence = lines[1].split(",")[1:-1]
        label = lines[3].split(",")[1:-1]

        residual_index = [idx for idx in range(len(sequence)) if sequence[idx] != "-"]
        index = [index[idx] for idx in residual_index]
        sequence = [sequence[idx] for idx in residual_index]
        label = [label[idx] for idx in residual_index]

        sequence = "".join(sequence)
        label = "".join(label)

        # Hu-mAb data uses IMGT numbering.
        num_index = [int(idx) if len(idx) <= 3 else int(idx[:-1]) for idx in index]
        cdr_region = [int((num_idx >= 27 and num_idx <= 38) or (num_idx >= 56 and num_idx <= 65) or (num_idx >= 105 and num_idx <= 117)) for num_idx in num_index]
        vernier_idx = [2, 52, 53, 54, 76, 78, 80, 82, 87, 118] if chain == "H" else [2, 4, 41, 42, 52, 53, 54, 55, 78, 80, 84, 85, 87, 118]
        vernier_region = [int(num_idx in vernier_idx) for num_idx in num_index]

        data = AntibodyData(
            index=index,
            sequence=sequence,
            label=label,
            cdr_region=cdr_region,
            vernier_region=vernier_region,
        )
        self.dataset[chain].append(data)

    def load_data(self):
        file = open(self.args.data_path)
        lines = file.readlines()
        lines = [line for line in lines if len("".join(line.strip().split(",")))]
        vh_offset, vl_offset = 12, 24
        for seq_idx in range(25):
            self.append("H", lines[24 * seq_idx + vh_offset: 24 * seq_idx + vh_offset + 4])
            self.append("L", lines[24 * seq_idx + vl_offset: 24 * seq_idx + vl_offset + 4])
        file.close()

    def predict(self):
        predictions, rewards = {"H": [], "L": []}, {"H": [], "L": []}
        for chain, chain_dataset in self.dataset.items():
            for data in chain_dataset:
                if self.args.graft == "cdr":
                    graft_region = data.cdr_region
                elif self.args.graft == "vernier":
                    graft_region = [cdr_region or vernier_region for cdr_region, vernier_region in zip(data.cdr_region, data.vernier_region)]
                else:
                    raise NotImplementedError

                prediction, reward = self.predictor(
                    chain=chain,
                    sequence=data.sequence,
                    graft_region=graft_region,
                    step=self.args.step,
                    dedup=bool(self.args.dedup),
                    sort=bool(self.args.sort)
                )
                predictions[chain].append(prediction)
                rewards[chain].append(reward)
        self.save_predictions(predictions, rewards)
        return predictions

    def save_predictions(self, predictions, rewards):
        fasta_file = open(f"{self.args.output_path}.fasta", "w")
        for chain in predictions:
            for idx, prediction in enumerate(predictions[chain]):
                print(f">Antibody{idx+1:02} V{chain}\n{prediction}", file=fasta_file)
        fasta_file.close()
        pickle_file = open(f"{self.args.output_path}.pickle", "wb")
        pickle.dump(rewards, file=pickle_file)
        pickle_file.close()

    def evaluate(self, predictions):
        lm_score_improvement, cdr_lm_score_improvement, total_preservation, vernier_preservation, total_precision, vernier_precision = [], [], [], [], [], []
        for chain in self.dataset:
            for data, prediction in zip(self.dataset[chain], predictions[chain]):
                lm_score_improvement.append(calc_lm_score(chain, prediction) - calc_lm_score(chain, data.sequence))
                total_preservation.append(sum([int(raw == pred) for raw, pred in zip(data.sequence, prediction)]) / len(data.sequence))
                vernier_preservation.append(sum([int((raw == pred) and vernier) for raw, pred, vernier in zip(data.sequence, prediction, data.vernier_region)]) / sum(data.vernier_region))
                total_precision.append(sum([int((pred == truth) and (raw != pred)) for raw, pred, truth in zip(data.sequence, prediction, data.label)]) /
                                       sum([int(raw != pred) for raw, pred in zip(data.sequence, prediction)]))
                if sum([int((raw != pred) and vernier) for raw, pred, vernier in zip(data.sequence, prediction, data.vernier_region)]):
                    vernier_precision.append(sum([int((pred == truth) and (raw != pred) and vernier) for raw, pred, truth, vernier in zip(data.sequence, prediction, data.label, data.vernier_region)]) /
                                            sum([int((raw != pred) and vernier) for raw, pred, vernier in zip(data.sequence, prediction, data.vernier_region)]))

        return {
            "lm_score_improvement": sum(lm_score_improvement) / len(lm_score_improvement),
            "total_preservation": sum(total_preservation) / len(total_preservation),
            "vernier_preservation": sum(vernier_preservation) / len(vernier_preservation),
            "total_precision": sum(total_precision) / len(total_precision),
            "vernier_precision": sum(vernier_precision) / len(vernier_precision)
        }

def main(args):
    dataset = AntibodyDataset(args)
    dataset.load_data()
    predictions = dataset.predict()
    results = dataset.evaluate(predictions)
    return results

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    results = main(args)
    print(results)