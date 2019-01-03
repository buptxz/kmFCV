# K-fold-m-step Forward Cross-validation (kmFCV)
K-fold-m-step forward cross-validation. A new approach of evaluating extrapolation performance in materials property prediction.

##  Prerequisites

This package requires:

- [keras](https://keras.io/)
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)
- [matminer](https://hackingmaterials.github.io/matminer/)
- [ase](https://wiki.fysik.dtu.dk/ase/index.html)
- [pybtex](https://pybtex.org/)

## Usage


In directory `kmFCV`, you can evaluate a random forest with magpie feature on Materials Project formation energy dataset:

```bash
python evaluation.py
```

Options:
- --data-path, feature data path
- --demo, to enable the demo mode
- --dataset, dataset name {mp,supercon}
- --property, property to predict {formation_energy,band_gap,Tc}
- --feature, feature name {magpie,composition,ptr}
- --model, model to use {1nn,rf,mlp,cnn,cgcnn}
- --valiation, validation type {cv,fcv}
- -k, k fold value for cv and fcv
- -m, m step value for fcv

For example, you can evaluate Materials Project band gap dataset, using CNN model with PTR feature and 100 fold 2 step forward cross-validation, with demo mode enabled like this: 

```bash
python evaluation.py --demo --dataset mp --feature ptr --model cnn --validation fcv -k 100 -m 2
```

After running, you will get three files in `data/results` directory.

- `.csv`: stores the results in csv.
- `.pkl`: stores the results in pkl.
- `.png`: plots the figure.

Also, MAE, RMSE, R squared and expolation accuracy will be stored in 'data/results/results.csv' file.
