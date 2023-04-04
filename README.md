# VWCE-Net

## Readme
* This is a PyTorch implementation for paper "Video analysis of small bowel capsule endoscopy using a Transformer network". We build this model based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).


## Environment
```
pip install -r requirements.txt
```

## Evaluation
* 
```
 CUDA_VISIBLE_DEVICES=0 python tools/inference.py \
            --evaluate_pth=${pthfile} \
            --dataname=${pthfile} --annfile=${annfile}
```

## Citations
* 
```
@inproceedings
```
