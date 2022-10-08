# PAUCI
An implement of the NeurIPS 2022 paper: [**Asymptotically Unbiased Instance-wise Regularized Partial AUC Optimization: Theory and Algorithm**].

## Environments
* **Ubuntu** 16.04
* **CUDA** 10.2
* **Python** 3.7.3
* **Pytorch** 1.12.1+cu10
* **Numpy** 1.21.4
* **pandas** 1.3.5
* **scikit-learn** 1.0.1

## Data preparation

Download [cifar-10-long-tail]([https://drive.google.com/uc?export=download&id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8](https://drive.google.com/drive/folders/191lqLKQFksMci_Dm1EC-B7M7OfHSO8Fk?usp=sharing)), [cifar-100-long-tail]([https://www.pkuml.org/resources/pku-vehicleid.html](https://drive.google.com/drive/folders/191lqLKQFksMci_Dm1EC-B7M7OfHSO8Fk?usp=sharing)), and [tiny-imagenet-200]([https://github.com/visipedia/inat_comp/tree/master/2018#Data](https://drive.google.com/drive/folders/191lqLKQFksMci_Dm1EC-B7M7OfHSO8Fk?usp=sharing)). Unzip these files and place then in `./data/[dataset]/`.

## Training & Testing

Using the pretrained models  in `./pretrained_models_back`.

Run the following command for training & validation

```shell
python3 train_SPAUCI.py
```

## Example

```shell
With augmentations.
train set has 10415 images
class number:  [9511  904]
test set has 2236 images
class number:  [2042  194]
val set has 2233 images
class number:  [2039  194]
--------------------------------------------------
To balance ratio, add 1474 pos imgs (with replace = True)
--------------------------------------------------
--------------------------------------------------
after complementary the ratio, having 11889 images
--------------------------------------------------
epoch:0 val pauc:0.9366215047631786
epoch:1 val pauc:0.9495861038852849
...
epoch:48 val pauc:0.9628886839452894
epoch:49 val pauc:0.9652434893251374
test pauc:0.7721479724874244
```

## Losses

The following methods are provided in this repository (see Problem Formulation in our paper):

* OPAUC Losses, An implement of the loss function in section 3.1.
* TPAUC Losses, An implement of the loss function in section 3.2.

See `./losses/SPAUCI.py` for usage.

## Optimizer

An implement of the training algorithm in section 4.

See `./optimizer/MinMax.py` for usage.

## References
If this code is helpful to you, please consider citing our paper:
```
@inproceedings{shao2022pauci,
  title={Asymptotically Unbiased Instance-wise Regularized Partial AUC Optimization: Theory and Algorithm},
  author={Shao, Huiyang and Xu, Qianqian and Yang, Zhiyong and Bao, Shilong and Huang, Qingming},
  booktitle={Annual Conference on Neural Information Processing Systems},
  year={2022}
}
```
