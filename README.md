# SPEED+: Next-Generation Dataset for Spacecraft Pose Estimation across Domain Gap

This repository is developed by Tae Ha "Jeff" Park at [Space Rendezvous Laboratory (SLAB)](https://slab.stanford.edu) of Stanford University.

- [2021.12.02] Our paper will be presented at the 2022 IEEE Aerospace Conference! This repository is updated for our latest draft which will soon become available in arXiv.
- [2022.08.13] Citation update.
- [2022.10.12] As announced on the [Kelvins website](https://kelvins.esa.int/pose-estimation-2021/discussion/90/), the post-mortem competition will terminate on 2022/12/31, followed by the full release of the lightbox and sunlamp labels on 2023/01/01. This repository will be updated shortly afterwards to reflect the new availability of the test labels.

## Introduction

This is the official repository of the baseline studies conducted in our paper titled [SPEED+: Next-Generation Dataset for Spacecraft Pose Estimation across Domain Gap](https://ieeexplore.ieee.org/document/9843439). It consists of the official PyTorch implementations of the following CNN models:

- Keypoint Regression Network (KRN) [[arXiv](https://arxiv.org/abs/1909.00392)]
- Spacecraft Pose Network (SPN) [[arXiv](https://arxiv.org/abs/1906.09868)]

The implementation of the SPN model follows from the original work by Sumant Sharma based on Tensorflow and MATLAB. The repository also supports the following algorithms for the KRN model:

- Domain randomization via style augmentation introduced in [Style Augmentation: Data Augmentation via Style Randomization](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Deep%20Vision%20Workshop/Jackson_Style_Augmentation_Data_Augmentation_via_Style_Randomization_CVPRW_2019_paper.pdf) by Jackson et al. (2019). The implementation derives from the [official GitHub repository](https://github.com/philipjackson/style-augmentation).
- Domain adaptation via DANN introduced in [Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/volume17/15-239/15-239.pdf) by Ganin et al. The implementation of Gradient Reversal Layer derives from the following [GitHub repository](https://github.com/jvanvugt/pytorch-domain-adaptation).

## Currently Unavailable Features

The SPEED+ dataset is currently released and used for the [Satellite Pose Estimation Competition (SPEC2021)](https://kelvins.esa.int/pose-estimation-2021/). Following the end of SPEC2021, a post-mortem version is currently on-going. In fairness of the competition, some items that are necessary to reproduce our results are not available at this stage. These include:

- Keypoints data used to train KRN (`src/utils/tangoPoints.mat`)
- CSV files containing bounding box and keypoint labels (KRN) or spacecraft attitude classes (SPN) for all domains of SPEED+

However, you can still create your own data to generate CSV files and train/test the models.

## Installation

The code is developed and tested with python 3.7 on Ubuntu 20.04. It is implemented with PyTorch 1.8.0 and trained on a single NVIDIA GeForce RTX 2080 Ti 12GB GPU.

1. Install [PyTorch](https://pytorch.org/).

2. Clone this repository. Its full path (`$PROJROOT`) should be specified for `--projroot` in `config.py`.

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Download [SPEED+](https://purl.stanford.edu/wv398fc4383). Its full path (`$DATAROOT`) should be specified for `--dataroot` in `config.py`.

5. Place the appropriate CSV files under `$DATAROOT/{domain}/splits_{model}/`. For example, CSV files for synthetic training and validation sets for KRN should be placed under `$DATAROOT/synthetic/splits_krn/`. See below for creating CSV files for yourself.

6. Download the pre-trained AlexNet weights (`bvlc_alexnet.npy`) from [here](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) and place it under `$PROJROOT/checkpoints/pretrained/` to be used for SPN.

## Pre-processing

1. First, recover 11 keypoints as described in this [paper](https://arxiv.org/abs/1909.00392). The order of keypoints does not matter as long as you are consistent with them. Save it as [3 x 11] array under the variable named `tango3Dpoints` and save it under `src/utils/tangoPoints.mat`. If you choose to save it elsewhere, make sure to specify its location w.r.t. `$PROJROOT` at `--keypts_3d_model` in `config.py`.

2. For SPN, the attitude classes are provided at `src/utils/attitudeClasses.mat`.

3. Pre-processing can be done from `preprocess.py`. Specify the below arguments when running the script, which will convert the JSON file at `$DATAROOT/{domain}/{jsonfile}` to `$DATAROOT/{domain}/{outcsvfile}`.

| Argument | Description |
| -------- | ----------- |
| `--model_name` | KRN or SPN (e.g. `krn`) |
| `--domain` | Dataset domain (e.g. `synthetic`)|
| `--jsonfile` | JSON file name to convert (e.g. `train.json`)|
| `--csvfile` | CSV file to write (e.g. `splits_krn/train.csv`)|

For example, to create CSV file of SPEED+ `synthetic` training set for KRN, run
```
python preprocess.py --model_name krn --domain synthetic --jsonfile train.json --csvfile splits_krn/train.csv
```

## Training & Testing

Use below arguments to toggle on/off some settings:

| Argument              | Description |
| --------------------- | ----------- |
| `--no_cuda`           | Disable GPU training |
| `--use_fp16`          | Use mixed-precision training |
| `--randomize_texture` | Perform style augmentation online during training |
| `--perform_dann`      | Perform domain adaptation via DANN |

Note the networks in this repository are not trained with mixed-precision training, but it's recommended if your GPU supports Tensor Cores to expedite the training.

To train KRN on SPEED+ synthetic training set:
```
python train.py --savedir 'checkpoints/krn/synthetic_only' \
                --logdir 'log/krn/synthetic_only' \
                --model_name 'krn' --input_shape 224 224 \
                --batch_size 48 --max_epochs 75 \
                --optimizer 'adamw' --lr 0.001 \
                --weight_decay 0.01 --lr_decay_alpha 0.95 \
                --train_domain 'synthetic' --test_domain 'synthetic' \
                --train_csv 'train.csv' --test_csv 'test.csv'

```

Add `--randomize_texture` to train with style augmentation.

To test KRN on `synthetic` validation images:
```
python test.py --pretrained 'checkpoints/krn/synthetic_only/model_best.pth.tar' \
               --logdir 'log/krn/synthetic_only' --resultfn 'results.txt' \
               --model_name 'krn' --input_shape 224 224 \
               --test_domain 'synthetic' --test_csv 'validation.csv'
```
which will write the test results to `$PROJROOT/log/krn/synthetic_only/results.txt`.

To test KRN on `lightbox` test images with DANN:
```
python adapt.py --savedir 'checkpoints/krn/dann_lightbox' \
                --logdir 'log/krn/dann_lightbox' --resultfn 'results.txt' \
                --model_name 'krn' --input_shape 224 224 \
                --batch_size 16 --max_epochs 750 --test_epoch 50 \
                --optimizer 'adamw' --lr 0.001 \
                --weight_decay 0.01 --lr_decay_alpha 0.95 --lr_decay_step 10 \
                --train_domain 'synthetic' --test_domain 'lightbox' \
                --train_csv 'train.csv' --test_csv 'lightbox.csv' \
                --perform_dann
```
which currently assumes `lightbox.csv` is available with test labels for occasional validation. (You can comment out relevant parts in `adapt.py` to not run testing at all.)
## License

The SPEED+ basline studies repository is released under the MIT License.

## Citation

If you find this repository and the SPEED+ dataset helpful in your research, please cite the paper below along with the dataset itself.
```
@inproceedings{park2022speedplus,
  author={Park, Tae Ha and M{\"a}rtens, Marcus and Lecuyer, Gurvan and Izzo, Dario and D'Amico, Simone},
  booktitle={2022 IEEE Aerospace Conference (AERO)},
  title={SPEED+: Next-Generation Dataset for Spacecraft Pose Estimation across Domain Gap},
  year={2022},
  pages={1-15},
  doi={10.1109/AERO53065.2022.9843439}
}
```

KRN was introduced in the following paper:
```
@inproceedings{park2019krn,
	author={Park, Tae Ha and Sharma, Sumant and D'Amico, Simone},
	booktitle={2019 AAS/AIAA Astrodynamics Specialist Conference, Portland, Maine},
	title={Towards Robust Learning-Based Pose Estimation of Noncooperative Spacecraft},
	year={2019},
	month={August 11-15}
}
```

SPN was introduced in the following paper:
```
@inproceedings{sharma2019spn,
	author={Sharma, Sumant and D'Amico, Simone},
	booktitle={2019 AAS/AIAA Space Flight Mechanics Meeting, Ka'anapali, Maui, HI},
	title={Pose Estimation for Non-Cooperative Spacecraft Rendezvous Using Neural Networks},
	year={2019},
	month={January 13-17}
}
```


