# VIT-experiment

This repository contains an experimental Vision Transformer (ViT) model for image classification tasks.

## Structure of the Repo

- `vit.py`: Contains the main ViT model definition and training code.
- `output.txt`: (If exists) Contains the output results of the model training.
- `LICENSE`: License file.
- `README.md`: This file, providing an overview of the repo.

## vit.py

The `vit.py` file contains a ViT model definition and training code. The model is trained and tested using the MNIST dataset.

### ViT Model

The ViT model first splits the input image into multiple small patches, then encodes these patches into a linear embedding vector. These embedding vectors are then fed into a Transformer structure, which can learn the relationships between the patches. Finally, the model classifies through a fully connected layer.

### Training and Evaluation

The model is trained and evaluated using the MNIST dataset. The dataset is first normalized and transformed into a shape suitable for the model. Then, the model is compiled and trained. Finally, the model is evaluated on the test dataset and outputs the prediction results.

## Usage

1. Clone this repo.
2. Make sure all necessary dependencies, such as numpy, tensorflow, and keras, are installed in your environment.
3. Run `vit.py` to train and evaluate the model.

## LICENSE

Please refer to the `LICENSE` file for detailed information.

## Contact

If you have any questions or suggestions, please send an email to `hibana2077@gmail.com`.
