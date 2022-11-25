import os
import sys
import torch
import numpy as np
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

torch.manual_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)



def load_pretrained_gan_model(repository, gan_architecture, model_name, use_gpu):
    """Load pretrained GAN models from Facebook Research GAN Zoo repository"""
    if gan_architecture == 'DCGAN':
        pretrained_gan_model = torch.hub.load(repository,
                                              gan_architecture,
                                              pretrained=True,
                                              useGPU=use_gpu)
    elif gan_architecture == 'PGAN':
        pretrained_gan_model = torch.hub.load(repository,
                                              gan_architecture,
                                              model_name=model_name,
                                              pretrained=True,
                                              useGPU=use_gpu)
    else:
        print(f"GAN Model: {gan_architecture} does not exist1")
        exit()

    return pretrained_gan_model


def generate_samples(pretrained_gan_model, number_of_samples):
    """Generate samples from the pretrained GAN model"""

    noise, _ = pretrained_gan_model.buildNoiseData(number_of_samples)

    with torch.no_grad():
        generated_samples = pretrained_gan_model.test(noise)

    return generated_samples


def visualize_samples(gan_architecture, generated_samples, title):
    """Display the GAN generated samples"""

    if gan_architecture == 'DCGAN':
        grid = torchvision.utils.make_grid(generated_samples)
    elif gan_architecture == 'PGAN':
        grid = torchvision.utils.make_grid(generated_samples.clamp(min=-1, max=1), scale_each=True, normalize=True)
    else:
        print(f"GAN Model: {gan_architecture} does not exist!")
        exit()

    plt.figure(figsize = (20,20))
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())


REPOSITORY = "facebookresearch/pytorch_GAN_zoo:hub"
GAN_ARCHITECTURE = "PGAN" # options = ['DCGAN', 'PGAN']
MODEL_NAME = "celebAHQ-256" # options = ['celebAHQ-256', 'celebAHQ-512', 'DTD', 'celeba']
USE_GPU = False if device.type == 'cpu' else True
NUMBER_OF_SAMPLES = 16


gan_model = load_pretrained_gan_model(repository=REPOSITORY,
                                      gan_architecture='DCGAN',
                                      model_name='',
                                      use_gpu=USE_GPU)


synthesized_samples = generate_samples(pretrained_gan_model=gan_model,
                                       number_of_samples=NUMBER_OF_SAMPLES)

print(synthesized_samples)

type(synthesized_samples)

print(synthesized_samples.shape)


visualize_samples(gan_architecture='DCGAN',
                  generated_samples=synthesized_samples,
                  title='DCGAN generated Fashion')

# gan_model = load_pretrained_gan_model(repository=REPOSITORY,
#                                       gan_architecture=GAN_ARCHITECTURE,
#                                       model_name='celebAHQ-512',
#                                       use_gpu=USE_GPU)

gan_model = load_pretrained_gan_model(repository=REPOSITORY,
                                      gan_architecture=GAN_ARCHITECTURE,
                                      model_name=MODEL_NAME,
                                      use_gpu=USE_GPU)

synthesized_samples = generate_samples(pretrained_gan_model=gan_model,
                                       number_of_samples=NUMBER_OF_SAMPLES)

visualize_samples(gan_architecture=GAN_ARCHITECTURE,
                  generated_samples=synthesized_samples,
                  title="PGAN generated Celebrities (256x256)")

plt.show()