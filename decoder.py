import json
import torch
from RES_VAE import VAE
import os

import torchvision.utils as vutils


use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

# Load the JSON file into a Python dictionary
with open('/home/qw3971/cv-2023/output.json', 'r') as json_file:
    data_shifted = json.load(json_file)

# Access the first element of the dictionary
values_shifted_tensor = torch.tensor(data_shifted)

checkpoint = torch.load("/home/qw3971/cnn-vae-retrain/retina_result/Models/test_run_64.pt", map_location = 'cpu')
vae_net = VAE(channel_in = 3, ch=64, latent_channels = 256).to(device)

vae_net.load_state_dict(checkpoint['model_state_dict'])
decoder = vae_net.decoder

output_folder = '/home/qw3971/cnn-vae-retrain/pred_images'

vector_pred = values_shifted_tensor.to(device)
img = decoder(vector_pred)

for j, generated_image in enumerate(img):

        # Save the concatenated image using vutils.save_image
        vutils.save_image(
            generated_image,
            os.path.join(output_folder, f"{j}.png"),
            normalize=True
        )

print('hi')