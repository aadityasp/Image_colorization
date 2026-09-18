# Image Colorization

A PyTorch implementation of [Deep Koalarization](https://arxiv.org/abs/1712.03400) (Baldassarre, Morín, Rodés-Guirao): a network that takes the lightness channel of a grayscale photo and predicts its two color channels in LAB space. On top of the paper's setup, it tests two changes, a different loss function and stronger pretrained feature extractors, and compares the four resulting models on the same data.

Full write-up with figures: [`CV_Final_Project_report.pdf`](CV_Final_Project_report.pdf). Course project for CS 5330 (Computer Vision), Northeastern University.

## How the model works

An image is converted to LAB. The L channel (a grayscale image) goes through two paths at once:

- an **encoder** CNN, trained from scratch, that produces a mid-level feature map
- a **pretrained classifier**, used as a high-level feature extractor, whose embedding is tiled and concatenated onto the encoder output (the fusion block, 256 convolutions of 1x1 kernels)

A **decoder** upsamples the fused tensor back to the image size and outputs the a and b channels. The predicted ab is stacked with the original L to form the colorized image.

## The four experiments

Each one is a notebook in `implementations/` with a matching set of trained weights at the repo root.

| | Feature extractor | Loss | Batch | Epochs | Notebook | Weights |
|---|---|---|---|---|---|---|
| a | Inception-ResNet-v2 (as in the paper) | MSE | 100 | ~40 | `colorization_Author_MSEloss.ipynb` | `colorization_authorimpl_final.pt` |
| b | Inception-ResNet-v2 | MSLE | 250 | ~40 | `colorization_with_MSLEloss.ipynb` | `colorization_final_MSLE.pt` |
| c | SE-ResNet-152 | MSLE | 128 | ~48 | `colorization_with_se_resnet152.ipynb` | `colorization_final_seResnet.pt` |
| d | EfficientNet-B7 | MSLE | 64 | ~48 | `colorization_with_efficientnet.ipynb` | `colorization_final_efficientnet.pt` |

Each run took 20 to 25+ hours on a single NVIDIA GTX 1660 Ti.

## Data

245K images from the first 500 ImageNet classes. 200K of them were split 80/20 into training (over 160K) and validation (over 40K), with the remaining 45K held out as a test set. An early attempt on only 10K images overfit from the fifth epoch and produced almost no color, which is what pushed the data size up.

`data/` in this repo holds a small sample (ten classes, about 4,800 images), a `test_data/` set, and `test_outputs/` with example predictions from the SE-ResNet and MSLE models so the results can be seen without retraining.

## What the report found

- **MSLE beats MSE.** For the same feature extractor and the same number of epochs, the MSLE loss produced visibly better color than the paper's MSE loss.
- **SE-ResNet-152 + MSLE was the best of the four.** It converged to the right colors fastest and also trained fastest of the group, reaching a training loss of 0.001023 and a validation loss of 0.001318 by about 48 epochs.
- **EfficientNet-B7 lagged.** Its outputs carried an orange-red tinge at the same epoch count (training loss 0.004453, validation 0.004380).
- **All four fell short of the paper's results.** The likely cause is sampling: the 245K images come from the first 500 classes only, so the models do well on some subjects and poorly on others. There is also no regularization or batch normalization in the architecture, and overfitting shows up in every run.

## Running it

Dependencies: `torch`, `torchvision`, `pretrainedmodels`, `scikit-image`, `numpy`, `matplotlib`, `tqdm`. Open any notebook in `implementations/`, point the dataset path at a folder of RGB images, and run the cells top to bottom. The notebooks define the `Encoder`, `Decoder`, `Network`, `ImageDataset`, and `MSLELoss` classes and save weights with `torch.save(model.state_dict(), ...)`.

To colorize with the trained weights, build the same `Network` from the matching notebook and load the state dict:

```python
model = Network()  # from the notebook that matches the weights file
model.load_state_dict(torch.load("colorization_final_seResnet.pt", map_location="cpu"))
model.eval()
```

## Author

Aditya Appana
