# Federated Multi-Armed Bandits Under Byzantine Attacks
Author: Ilker Demirel ([ilkerd1997@gmail.com](mailto:ilkerd1997@gmail.com))

The repository for the manuscript "[Federated Multi-Armed Bandits Under Byzantine Attacks](https://arxiv.org/abs/2205.04134)".

## Simulations

The simulation code for each scenario is named accordingly. Simply run the corresponding `__.py` file to reproduce the results for a particular scenario/algorithm. Please beware that some save file names should be manually arranged inside the `__.py` files (emphasized next to the relevant parameters that determine the save file name) for the `plots.py` to work.

## Plots
`plots.py` reproduces the plots in the "Experimental Results" section. The necessary data is readily available under `./np_arrays/` .

## Citing
If you use this software please cite the paper as follows:
```
@article{demirel2022federated,
  title={Federated Multi-Armed Bandits Under Byzantine Attacks},
  author={Demirel, Ilker and Yildirim, Yigit and Tekin, Cem},
  journal={arXiv preprint arXiv:2205.04134},
  year={2022}
}
```