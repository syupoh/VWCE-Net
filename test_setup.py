import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

inputimg_path = '/data/syupoh/dataset/endoscopy/videos/220107/SIM_16_00000'
model = init_recognizer(config_file, device=device)

inference_recognizer(model, inputimg_path)

# inference the demo video
# inference_recognizer(model, 'demo/demo.mp4')