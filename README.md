# DistributedSupervisedLearning
Distributedly training 10 different neural nets to recognize handwritten digits in images. Specifically, we consider a subset of the MNIST data
set containing 5000 images of 10 digits (0-9), of which 2500 are used for training and 2500 are used for testing. Training data are divided among ten agents connected in an undirected
unweighted ring topology. For more details on the algorithm, please see https://arxiv.org/abs/1908.06693
