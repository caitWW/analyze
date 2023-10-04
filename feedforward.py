import json
import torch 
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the JSON file into a Python dictionary
with open('/home/qw3971/cnn-vae-retrain/latent_vec.json', 'r') as json_file:
    data = json.load(json_file)

# Access the first element of the dictionary
key = list(data.keys())

values = list(data.values())
values_tensor = torch.tensor(values) 

def find_values_in_json_target(json_file, keys_to_find):
    with open(json_file, 'r') as file:
        data1 = json.load(file)
        targets = []
        orig = []
        key_updated = []

        for key in keys_to_find:
            new_key = "shifted_"+key
            if new_key in data1:
                targets.append(data1[new_key])
                key_updated.append(key)
                orig.append(data[key])
            else:
                pass

        return orig, targets, key_updated
    
json_file_path = "/home/qw3971/cnn-vae-retrain/latent_vec_shifted.json"
orig, targets, key_updated = find_values_in_json_target(json_file_path, key)
targets_tensor = torch.tensor(targets)
orig_tensor = torch.tensor(orig)

def find_values_in_json_result(json_file, keys_to_find):
    with open(json_file, 'r') as file:
        data = json.load(file)
        results = []

        for key in keys_to_find:
            if key in data:
                results.append(data[key])
            else:
                results[key] = "Key not found in JSON"

        return results



if __name__ == "__main__":
    json_file_path = "/home/qw3971/cnn-vae-old/run2_saccade.json"  # Replace with the path to your JSON file  # Replace with the list of keys you want to find

    results = find_values_in_json_result(json_file_path, key_updated)
    results_tensor = torch.tensor(results)

    #with open('/Users/cw/Desktop/test_saccade.json', 'w') as file:
        #json.dump(results, file)

    #print(results)

import torch
import torch.nn as nn

print(results_tensor.shape)
print(orig_tensor.shape)
print(targets_tensor.shape)

dataset = TensorDataset(orig_tensor, results_tensor, targets_tensor)
batch_size = 32
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)


class SaccadeShiftNN(nn.Module):
    def __init__(self):
        super(SaccadeShiftNN, self).__init__()
        self.latent_fc = nn.Linear(256*12*17, 512)
        self.saccade_fc = nn.Linear(2, 512)
        self.combine_fc = nn.Linear(512+512, 256)
        self.output_fc = nn.Linear(256, 256*12*17)

    def forward(self, latent_input, saccade_input):
        latent_flat = latent_input.view(latent_input.size(0), -1)
        saccade_flat = saccade_input.view(saccade_input.size(0), -1)
        latent_out = self.latent_fc(latent_flat)
        saccade_out = self.saccade_fc(saccade_flat)
        combined = torch.cat((latent_out, saccade_out), dim=1)
        combined_out = self.combine_fc(combined)
        output = self.output_fc(combined_out)
        output = output.view(-1, 256, 12, 17)

        return output

model = SaccadeShiftNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 100

loss = []

for i in range(epochs):
    for batch_orig, batch_results, batch_targets in dataloader:
        batch_orig = batch_orig.to(device)
        batch_results = batch_results.to(batch_orig.dtype)
        batch_results = batch_results.to(device)
        batch_targets = batch_targets.to(device)

        saccade_shifted_latent = model(batch_orig, batch_results)
        loss = criterion(saccade_shifted_latent, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   
    loss.append(loss.item())
    
    print(f'Epoch [{i+1}/{epochs}], Loss: {loss.item()}')

saccade_shifted_latent = saccade_shifted_latent.cpu().detach().numpy()
saccade_shifted_latent = saccade_shifted_latent.tolist()

with open('/home/qw3971/cv-2023/output.json', 'w') as json_file:
    json.dump(saccade_shifted_latent, json_file)

print('done') 
print(loss)