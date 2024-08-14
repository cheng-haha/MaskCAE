<h1 align="center">🌟🌟MaskCAE: Mask Convolutional AutoEncoder for HAR</h1>
<p align="center"><a href="https://cheng-haha.github.io/papers/MaskCAE.pdf">Paper</a></p>
<p align="center"><a href="https://cheng-haha.github.io/">Dongzhou Cheng</a></p>

## 🔆🚀If you use this project, please cite our papers.
```
@article{cheng2024maskcae,
  title={MaskCAE: Masked Convolutional AutoEncoder via Sensor Data Reconstruction for Self-Supervised Human Activity Recognition},
  author={Cheng, Dongzhou and Zhang, Lei and Qin, Lutong and Wang, Shuoyuan and Wu, Hao and Song, Aiguo},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```
```
@article{cheng2023protohar,
  title={Protohar: Prototype guided personalized federated learning for human activity recognition},
  author={Cheng, Dongzhou and Zhang, Lei and Bu, Can and Wang, Xing and Wu, Hao and Song, Aiguo},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={27},
  number={8},
  pages={3900--3911},
  year={2023},
  publisher={IEEE}
}
```
```
@article{cheng2023learning,
  title={Learning hierarchical time series data augmentation invariances via contrastive supervision for human activity recognition},
  author={Cheng, Dongzhou and Zhang, Lei and Bu, Can and Wu, Hao and Song, Aiguo},
  journal={Knowledge-Based Systems},
  volume={276},
  pages={110789},
  year={2023},
  publisher={Elsevier}
}

```


# Pretrain environment
```
conda activate -n maskcae python=3.8

python -m pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

python -m pip install torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

python -m pip uninstall setuptools

python -m pip install setuptools==59.5.0

python -m pip install -r requirements.txt
```
Activate your environment
```
conda activate maskcae
```

## Settings
1. enter the folder
```
cd /MaskCAE
```
2. check the yaml configs:  `pre_trained_path`, this is the path to save the pre-trained weights
3. check the `WINDOWS_SAVE_PATH` and `ROOT_PATH`
   * `WINDOWS_SAVE_PATH` is the path to save the pre-processed data
   * `ROOT_PATH` is the path to save the raw dataset

### Get all pretrained weights
```
python pretrain/script/run.py --dataset ucihar 

python pretrain/script/run.py --dataset uschad 

python pretrain/script/run.py --dataset motion 
```


# Self-Supervised Learning
*FCN Base (without MaskCAE)* 
```
# ucihar LOCV mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaebase.yaml --dataset ucihar --device 0 --times 1

# uschad Given mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaebase.yaml --dataset uschad --device 2 

# motion LOCV mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaebase.yaml --dataset motion --device 1 --times 1
```
*FCN fully finetune*
```
# ucihar LOCV mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaeConv4Net.yaml --dataset ucihar --device 2 --times 1

# uschad Given mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaeConv4Net.yaml --dataset uschad --device 1 

# motion LOCV mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaeConv4Net.yaml --dataset motion --device 2 --times 1
```
*FCN Linear Evalution*
```
# ucihar LOCV mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaeConv4Net_LP.yaml --dataset ucihar --device 0 --times 1

# uschad Given mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaeConv4Net_LP.yaml --dataset uschad --device 1 

# motion LOCV mode
python scripts/MaskCAE/Base.py --config configs/yaml/maskcaeConv4Net_LP.yaml --dataset motion --device 2 --times 1

```
