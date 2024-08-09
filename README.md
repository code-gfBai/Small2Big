# Small2Big

Small2Big, a novel data expansion strategy for plant disease classification that uses textual features of disease symptoms to generate synthetic leaf images.

## Installation

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.3 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Small2Big requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name small2big -y python=3.7
conda activate small2big
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA 11.3, and other dependencies
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Training

### Training Small2Big:
You will need to prepare the data in the following format: instance_img for diseased images, ideally with 2-5 images, and class_img for healthy images of the species, ideally with no more than 200 images.

```
data
├── Small2Big
│   ├── instance_img
│   │   ├── disease_0.jpg
│   │   ├── ...
│   │   ├── disease_4.jpg
│   ├── class_img
│   │   ├── class_0.jpg
│   │   ├── ...
│   │   ├── class_199.jpg
```
You can adjust the train steps according to the number of images in instance_img. The recommendation is to increase the train steps by 100 for each image, e.g., 3 diseased images correspond to 300 train steps.
run:
```bash
bash scripts/train_G1_one_GPU.sh
```

### Direct fine-tuning conditional latent diffusion model:
You will need to prepare the data in the following format. `metadata.csv` can be generated via `tools/make_csv.py`.
```
data
├── direct_fine-tuning
│   ├── train
│   │   ├── species
│   │   │   ├── class_0
│   │   │   │   ├── 0.jpg
│   │   │   │   ├── ...
│   │   │   │   ├── 19.jpg
│   │   │   ├── ...
│   │   │   ├── class_3
│   │   ├── metadata.csv
```
run:
```bash
bash scripts/train_G0.sh
```

## Generating images
You can use the Small2Big or direct fine-tuning method of image generation via `scripts/gen_G1.sh` or `scripts/gen_G0.sh` respectively.

## Disease classification

### Data preparation
You can prepare the dataset in the following format:
```
data
├── species
│   ├── train
│   │   ├── class_0
│   │   │   ├── class_0_train_0.jpg
│   │   │   ├── ...
│   │   │   ├── class_0_train_19.jpg
│   │   ├── ...
│   │   ├── class_3
│   ├── val
│   ├── test
```

### Training
run:
```bash
bash scripts/train_resnet_disease_classification.sh
```

### Model evaluation
You can use the disease classification weights obtained from training in the previous step for both the confusion matrix and the ROC plot.
run:
```bash
bash scripts/test_confusion_matrix_roc.sh
```

## Tools
The tools directory contains useful scripts for data processing:

`kmeans.py`: Applies K-Means clustering.

`make_csv.py`: Generates CSV files from datasets.

`split_data.py`: Splits datasets into train, val and test sets.

`ssim.py`: Computes the Structural Similarity Index (SSIM) between images.


## Acknowledgements
Thanks for the great implementation of [diffusers](https://github.com/huggingface/diffusers). 

## License
This project is licensed under the MIT License. See the LICENSE file for details.