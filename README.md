## JARE 

A new theoretically motivated regularization method to stabilize the GAN training dynamics. 
(Please see more details in the paper: https://arxiv.org/abs/1806.09235)

#### Dependencies: 

* 64-bit Python 3.6 installation, Tensorflow 1.4 with GPU support 
* Numpy 1.14.1, Matplotlib 2.2.0, Scipy 1.0.0, tqdm 4.12.0, imageio 2.8.0, six 1.13.0, opencv-python 3.4.9.31

#### Synthetic data 

The folder `synthetic` contains the code for experiments on synthetic data.

To run experiments on Isotropic Gaussian:
```bash
cd synthetic
python3 Affine_GAN.py
```

To run experiments on GMM:
```bash
cd synthetic
python3 GMM_GAN.py
```

#### Real data (CIFAR-10)

The folder `real` contains the code for experiments on real data (CIFAR-10).

To run experiments on CIFAR-10, for example, we can do:
```bash
cd real/experiments
python3 jare.py 0 1
```
Here the first argument `0` represents the gpu_id (in the case of using
multiple gpus), and the second argument `1` represents the job_id (0-5), each of 
which means one of six network settings.

 To run baselines, for example [ConOpt](https://arxiv.org/abs/1705.10461), we can 
do: 
```bash
cd real/experiments
python3 conopt.py 0 1
```
Similarly, we can change the job_id in the script to run baselines on 
different network settings.

Note that in order to compute the FID score, we may need to first download
the [`inception_frozen.zip`](https://drive.google.com/file/d/14FgCNvRBh3OffkFFMEyYnrxavPSsvwZi/view?usp=sharing) 
and unzip it into the `inception` folder before training.



### Reference

To cite this work, please use
```
@INPROCEEDINGS{Nie2019UAI,
  author = {Weili Nie and Ankit Patel},
  title = {Towards a Better Understanding and Regularization of GAN Training Dynamics},
  booktitle = {UAI},
  year = {2019}
}
```
