# *K*-fold-*m*-step Forward Cross-validation (*km*FCV) for Materials Discovery
*K*-fold-*m*-step forward cross-validation is a new approach of evaluating extrapolation performance in materials property prediction. THe standard *k*-fold cross-validation falls short on evaluating the prediction performances of models in screening novel materials with desirable properties, wihch usually lie outside the domain of known materials. This project provides a comprehensive benchmarks studies on the extrapolation performances of a variety of prediction models on materials properties. Our results show even though current machine learning models can achieve good results when evaluated with standard cross-validation, their extrapolation power is actually very low as shown by our proposed *km*FCV evaluation method and the proposed extrapolation accuracy.

##  Prerequisites

This package requires:

- [keras](https://keras.io/)
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)
- [matminer](https://hackingmaterials.github.io/matminer/)
- [ase](https://wiki.fysik.dtu.dk/ase/index.html)
- [pybtex](https://pybtex.org/)

In directory `kmFCV`, you can test if all the prerequisites are met and show the help messages by:

```bash
python evaluation.py -h
```

If no error messages show up, it means that the prerequisites are installed properly.

## Usage

In directory `kmFCV`, run `evaluation.py` to do cross-validation or forward cross-validation on benchmark datasets and models. 

For example, without any arguments the pacakge uses the default ones to evaluate a random forest with magpie feature on Materials Project formation energy dataset:

```bash
python evaluation.py
```

All the options:
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

- `.csv`: stores the prediction results in csv.
- `.pkl`: stores the prediction results in pkl.
- `.png`: plots the prediction figure.

Also, the MAE, RMSE, R squared and expolation accuracy metrics will be stored in 'data/results/results.csv' file.
