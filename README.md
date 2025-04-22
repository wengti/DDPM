# Denoising Diffusion Probabilistic Model (DDPM)

## Credits & Acknowledgements
This project is a reimplementaion of [DDPM-PyTorch] by [explainingai-code] (https://github.com/explainingai-code/DDPM-Pytorch)
The code has been rewritten from scratch while maintaining the core concepts and functionalities of the original implementation.

## Features
- Build a DDPM that can be mofified using a config file with a compatible format.
- The model features an architecture of a modifiable base UNet that and a linear noise scheduler.
- Sinusoidal Embedding is implemented in this code.

## Description of Files:
- **extract_mnist.py** - Extracts MNIST data from CSV files.
- **custom_data.py** - Create a custom dataset.
- **model.py** - Compatible with .yaml config files to create various UNet models, featuring a sinusoidal embedding for the time step.
- **engine.py** - Defines the train step (for 1 epoch).
- **main.py** - Trains a DDPM model.
- **infer.py**
  1. Perform reverse diffusion that denoise the image step by step.
  2. Every denoised result at each step will be saved as an image.
