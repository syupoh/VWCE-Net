# VWCE-Net

## Readme
* This is a PyTorch implementation for paper "Video analysis of small bowel capsule endoscopy using a Transformer network". 

## Environment
* We build this repo from the OpenSource project [MMAction2](https://github.com/open-mmlab/mmaction2).
* The following steps are based on [MMAction2](https://github.com/open-mmlab/mmaction2).  

1. Create a conda virtual environment and install PyTorch and TorchVision: 
```
conda create -n $ENV_NAME python=3.6
conda activate $ENV_NAME
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

2. Install mmaction2 and requirements:
```
pip install openmim 
mim install mmaction2 -f https://github.com/open-mmlab/mmaction2.git
pip install -r requirements.txt
```


## Evaluation
* 
```
 CUDA_VISIBLE_DEVICES=0 python tools/inference.py --evaluate_pth=${pthfile} --dataname=${dataname} --annfile=${annfile}
```

## Citations
* 
```
@inproceedings
```
