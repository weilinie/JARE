## JARE 

A new theoretically motivated regularization method to stabilize the GAN training dynamics. 
(Please see more details in the paper: https://arxiv.org/abs/1806.09235)

Dependencies: 

* [Tensorflow 1.4](https://www.tensorflow.org/)
* [Numpy 1.14.1](http://www.numpy.org/)
* [Matplotlib 2.2.0](https://matplotlib.org)
* [Scipy 1.0.0](https://www.scipy.org)

To run experiments on Istropic Gaussian:
```bash
cd src
python3 Affine_GAN.py
```

To run experiments on GMM:
```bash
cd src
python3 GMM_GAN.py
```

To cite this work, please use
```
@INPROCEEDINGS{Nie2019UAI,
  author = {Weili Nie and Ankit Patel},
  title = {Towards a Better Understanding and Regularization of GAN Training Dynamics},
  booktitle = {UAI},
  year = {2019}
}
```
