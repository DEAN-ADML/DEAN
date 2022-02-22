# DEAN
Implementation of Deep Ensemble ANomaly detection. I will refer here to the corresponding paper, as soon as I am allowed to publish it.


base.py implements a single DEAN model. mass.py repeatedly calls it until there are 900 models trained. The outputs of each model are then saved for merge.py to combine them into one ensemble.

hyper.json allows to modify the hyperparameters of our model. representation needs to be either "latent" when using 5 autoencoder as preprocessing (not included here) or "mnist" when using pure mnist features. dataset allows choosing the normal class. When not using dataset "7", you should set the representation to "mnist" as else the autoencoder is trained on anormal datapoints. lr and batch modify learning rate and batch size. depth is the depth of the model, while bag is the amount of features included in each model




In "reason" our interpretability method is applied to each MNIST test sample.

To use the files provided you need to install git-lfs ( https://git-lfs.github.com/ )
