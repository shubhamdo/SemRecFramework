# https://mccormickml.com/2019/07/22/BERT-fine-tuning/

import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# with open('../../data/text/meditations/clean.txt', 'r') as fp:
#     text = fp.read().split('\n')

df = pd.read_csv("H:/Thesis/Output_Data/glassdoor_clean.csv", delimiter=";")
# df = pd.read_csv("H:/Thesis/Output_Data/glassdoor_clean.csv")

df = df[pd.notnull(df['job.description'])]
text = df['job.description'].tolist()
text[:5]

# for tt in text:
#     if type(tt) is not str:
#         print(tt)
#         print(type(tt))

inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs

inputs['labels'] = inputs.input_ids.detach().clone()
inputs.keys()

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

mask_arr

selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

selection[:5]

for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

inputs.input_ids


class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


dataset = MeditationsDataset(inputs)
del mask_arr
del rand

loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

from transformers import AdamW

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=5e-5)

from tqdm import tqdm  # for our progress bar

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and data loader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

"""
class Model:

    def bertMlmTrain(self):
        pass


if __name__ == "__main__":
    df = pd.read_csv("H:/Thesis/Output_Data/glassdoor_clean.csv")
"""
output_dir = "H:/Thesis/Output_Data/model/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
