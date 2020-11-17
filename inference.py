import torchaudio
import torch
from unet import UNet
import time


def load_model(model, state_dict_file_path=None):
    if state_dict_file_path is not None:
        model.load_state_dict(torch.load(state_dict_file_path))
    return model

model_path = "/home/un270/shvaas/resblock-test.pt"
x = torch.rand(1,3,256,256)
print(x.shape)
model = UNet(3,2,compression_factor=4)
model.eval()
t = 0.0
with torch.no_grad():
     for i in range(100):
         start = time.time()
         out = model(x)
         end = time.time()
         t += float(end-start)
#print("3 block - 1/2")
print(t/100)
