import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import os
import pdb

os.environ["WANDB_DISABLED"] = "true"

df = pd.read_csv('/h/brandon/internship/Uncover_implicit_bias/gpt2_training/data/ROCStories_winter2017.csv')
df = df.head(20)

# Define the function to concatenate the story title with the sentences
def concatenate_story_elements(row):
    title = row['storytitle']
    sentences = ' '.join(row[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']].tolist())
    return f'{title} [SEP] {sentences}'

# Apply the function to each row
df['story'] = df.apply(concatenate_story_elements, axis=1)

stories_data = list(df['story'])


# Convert list of strings to a dictionary with a key
stories_dict = {'story': stories_data}

# Convert the dictionary to a Hugging Face Dataset
dataset = Dataset.from_dict(stories_dict)


# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


tokenizer.add_special_tokens({'sep_token': '[SEP]'})
token_id = tokenizer.convert_tokens_to_ids('[SEP]')  # Replace <TITLE> with your token
print(f"The id for [SEP] is: {token_id}")


# Convert id to token
token_id = 50257
token = tokenizer.convert_ids_to_tokens(token_id)
print(f"The token for id {token_id} is: {token}")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer))



# Define a function to apply to each example
def preprocess_function(example):
    # Modify examples here (e.g., tokenize text)
    return tokenizer(example["story"]) 

# Apply the function to the dataset
processed_dataset = dataset.map(preprocess_function)

def check_token_ids(dataset, tokenizer):
    # print('Checking token ids')
    for sample in processed_dataset:
        print(sample)
        input_ids = sample["input_ids"]
        if max(input_ids) >= len(tokenizer):
            print(f"Invalid token ID found: {max(input_ids)}")



# Function to decode the token IDs
def decode_sample(token_ids):
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)

# Visualize the first few samples from train_dataset
print("Some samples from the training set:\n")
for i in range(5):  # let's print out the first 5 samples
    token_ids = processed_dataset[i]['input_ids']  # get the token IDs for the i-th sample
    text = decode_sample(token_ids)  # decode the token IDs to text
    print(f"Sample {i + 1}:\n{text}\n")


## Train

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False 
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',         # Output directory
    overwrite_output_dir=True,      # Overwrite the content of the output directory
    num_train_epochs=1,             # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    per_device_eval_batch_size=1,   # Batch size for evaluation
    eval_steps=400,                 # Evaluation step
    save_steps=800,                 # After # steps model is saved
    warmup_steps=500,               # Warmup steps
    logging_dir='./logs'
)

#Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=processed_dataset,
)


# Train the model
trainer.train()

