from fastai import datasets
from fastai.vision import *
import pickle, gzip, math, torch, matplotlib as mpl
from torch import tensor, nn
from pathlib import Path
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import DataLoader, Dataset
from run import DataBunch
from datablock import Data,make_rgb,RandomResizedCrop,PilRandomFlip,np_to_float,CenterCrop 

def get_files(path, extensions):
    path = Path(path)
    files = []
   
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in sorted( filter(lambda f: extensions in f, filenames), key=lambda x: int(x.split(".")[-2].split("_")[4]),):
            filepath = os.path.join(dirpath, filename)
            if "mask" not in filename:
                files.append(filepath)
    return files


def _get_files(p, fs, extensions):
    p = Path(p)
    return [p/f for f in fs if not f.startswith('.') and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]

def split_random(image_files, p):
  l1 = []
  l2 = []
  for i in range(0,len(image_files)):
    if np.random.uniform()<p:
      l1.append(image_files[i])     
    else:
      l2.append(image_files[i])
  return l1,l2

def compose(x, funcs, *args, **kwargs):
    if not funcs: return x
    for f in funcs: x = f(x, **kwargs)
    return x


class BrainSegmentationDataset(Dataset):

    def __init__(self, image_files):
        self.image_files = image_files
        self.mask_files = []
        for filepath in self.image_files:
                mask_path = filepath.split(".")[0] + "_mask." + filepath.split(".")[1]
                self.mask_files.append(mask_path)

    def __getitem__(self,i):
      x = PIL.Image.open(self.image_files[i])
      print(x)
      image_tensor = Tensor(imread(self.image_files[i]))
      mask_tensor = Tensor(imread(self.mask_files[i], as_gray=True))
      image_tensor = image_tensor.permute(2,0,1)*1/255.0
      #mask_tensor = mask_tensor.permute(2,0,1)*1/255.0
      tup = (image_tensor,mask_tensor)
      return tup
        
    def __len__(self):
      return len(self.image_files)

def load_data(image_path,image_size=256,bs=32):
    files = get_files(image_path,'.tif')
    train_files, valid_files = split_random(files, 0.8)

    train_transforms = [make_rgb, RandomResizedCrop(image_size, scale=(0.35,1)), PilRandomFlip(), np_to_float]
    valid_transforms = [make_rgb, CenterCrop(image_size), np_to_float]

    data = Data(train_files, valid_files, batch_size=bs, image_transforms=train_transforms, valid_image_transforms=valid_transforms,num_workers=8)
    # Do normalization
    return data

load_data('/scratch/un270/brain-mri/kaggle_3m')

MNIST_URL='http://deepdddlearning.net/data/mnist/mnist.pkl'
CIFAR10_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz'
class MNISTDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return len(self.x)

def get_data():
    path = 'mnist.pkl.gz'
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): 
    return (x-m)/s

def get_stats(x):
    mean, std = x.mean(), x.std()
    return mean, std

def get_data_bunch(batch_size):
    # MNIST
    x_train, y_train, x_valid, y_valid = get_data()
    
    train_mean, train_std = get_stats(x_train)
    x_train = normalize(x_train, train_mean, train_std)
    x_valid = normalize(x_valid, train_mean, train_std)

    train_ds = MNISTDataset(x_train[:,:], y_train)
    valid_ds = MNISTDataset(x_valid[:,:], y_valid)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, shuffle=False)

    data = DataBunch(train_dl, valid_dl)

    return data


def load_cifar_data(batch_size, image_size,size):
    if size==10:
       path = datasets.untar_data(URLs.CIFAR)
    else:
       path = datasets.untar_data(URLs.CIFAR_100)
    stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))
    
    tfms = (get_transforms(do_flip=True,flip_vert=False,max_rotate=25))
    data = ImageDataBunch.from_folder(path, valid='test', size=image_size,ds_tfms=tfms,bs = batch_size)
    data.normalize(imagenet_stats)

    print("Loaded data")
    return data
