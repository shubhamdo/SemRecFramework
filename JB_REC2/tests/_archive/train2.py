from transformers import BertTokenizer, BertForMaskedLM
import torch
import pandas as pd

print('Loading BERT tokenizer...')

# Load the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load Data and Filter Null Rows
df = pd.read_csv("H:/Thesis/Output_Data/glassdoor_clean.csv", delimiter=";")
df = df[pd.notnull(df['job.description'])]
text = df['job.description'].tolist()

# Tokenize the text and add padding
inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

# Get Labels
inputs['labels'] = inputs.input_ids.detach().clone()

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

selection = []
for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

""" 
ALTERNATIVE
# For every sentence...
for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)"""


# If there's a GPU available...
def testIfGPU():
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))
        return device
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        return device

# testIfGPU()
