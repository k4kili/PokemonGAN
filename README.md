# PokemonGAN
> A pytorch implementation of Pokemon creation using Generative Adversarial Networks. [[paper]](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

## Dataset
`256x256` images of pokemons collected from [here](https://www.kaggle.com/kvpratama/pokemon-images-dataset/home) with some preprocessing.

## Preprocessing
Given dataset has transparent background. Importing using pytorch made it in black background and contours were not distinguishable. The following python code snippet is used to whiten background and these were later normalized.

```python
from PIL import Image
import os

path = os.path.join(os.getcwd(), 'data/pokemon')
files = os.listdir(path)

for file in files:
    f = os.path.join(path, file)
    image = Image.open(f)
    image.convert('RGBA')
    canvas = Image.new('RGBA', image.size, (255,255,255,255))
    canvas.paste(image, mask=image)
    canvas.thumbnail([256, 256], Image.ANTIALIAS)
    canvas.save(f, format="PNG")

```

## Architecture
DCGAN was used with the following architecture for Generator and Discriminator.

```
Generator(
  (main): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace)
    (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
```

```
Discriminator(
  (main): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (12): Sigmoid()
  )
)
```

## Loss during training
Generator and Discriminator losses during training were plotted as follows.
![Loss](https://raw.githubusercontent.com/themousepotato/PokemonGAN/master/images/loss.png?token=AgXlPpKM0ft_hbT8RLKgn0E1pT20hg_4ks5cVvG7wA%3D%3D)

## Final result
The following result was generated on running for `1000` epochs with `128` minibatch size using `64x64` images.<br/><br/>
![Result](https://raw.githubusercontent.com/themousepotato/PokemonGAN/master/images/result.png?token=AgXlPkfTqHFM-KPvOz2X3UtVYf9TuiJDks5cVvHVwA%3D%3D)

## Conclusions
* As everyone says GANs are hard to train.
* Adding noise to the input images improves learning.
* We tried increasing input sizes. But, due to lack of resources we were unable to finish training.

## Contributors
* [Navaneeth Suresh](https://github.com/themousepotato)
* [Alwin Anto](https://github.com/kiliyan)
