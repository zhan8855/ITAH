# Improve Antibody Humanization with Monte-Carlo Tree Search

### Course Project for IFT 6162 and IFT 6269, Fall 2024, at Mila and Université de Montréal.

### Introduction

See [introduction.pdf](introduction.pdf).

### Reproduction

```
conda env create -f environment.yml
python main.py --method greedy --output_path greedy_pred
python main.py --method mcts --output_path mcts_pred
python main.py --method itah --output_path ours_pred --dedup 1 --sort 1
python main.py --method itah --output_path ours_pred_no_dedup --dedup 0 --sort 1
python main.py --method itah --output_path ours_pred_no_heurs --dedup 1 --sort 0
python vis.py
```
You can submit the output fasta file to [BioPhi platform](https://biophi.dichlab.org/humanization/humanness/), using bulk upload, IMGT numbering and relaxed threshold.
