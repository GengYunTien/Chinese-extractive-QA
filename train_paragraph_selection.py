import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from tqdm.auto import tqdm

import transformers
from transformers import AdamW
from transformers import AutoModelForMultipleChoice, AutoTokenizer, default_data_collator

from modules.utils import log, set_seed, load_data, preprocess_function
from modules.datasets import ParagraphSelectionDataset

parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
parser.add_argument("--context_file", type=str, default=None, help="A csv or a json file containing the context data.")
parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
parser.add_argument("--per_device_valid_batch_size", type=int, default=8, help="Batch size (per device) for the validation dataloader.")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Initial learning rate (after the potential warmup period) to use.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--logging_step", type=int, default=100)
parser.add_argument("--validation", type=bool, default=True)
parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
args = parser.parse_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set seed for reproducibility
set_seed(args.seed)

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path).to(device)

# load data
contextFile = load_data(args.context_file)
trainFile = load_data(args.train_file)
validFile = load_data(args.validation_file)

# process file
train_data = preprocess_function(contextFile, trainFile, tokenizer, args.max_seq_length)
valid_data = preprocess_function(contextFile, validFile, tokenizer, args.max_seq_length)

# dataset & dataloader
train_dataset = ParagraphSelectionDataset(train_data, mode='train')
valid_dataset = ParagraphSelectionDataset(valid_data, mode='val')

train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
valid_dataloader = DataLoader(valid_dataset, collate_fn=default_data_collator, batch_size=args.per_device_valid_batch_size)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# learning decay
def lr_lambda(current_step, total_steps):
    return max(0.0, 1 - current_step / total_steps)
total_steps = args.num_train_epochs * len(train_dataloader)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, total_steps))

# accelerator
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="fp16")
model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader) 

log('Strat training')
for epoch in range(args.num_train_epochs):
    step = 1
    train_loss, train_acc = 0, 0
    model.train()
    for batch in tqdm(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            acc = (predictions == batch['labels']).float().mean().item()
            train_acc += acc
            train_loss += loss.item()
        
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

        if step % args.logging_step == 0:
            log(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss / args.logging_step:.3f}, acc = {train_acc / args.logging_step:.3f}")
            train_loss, train_acc = 0, 0

    if args.validation:
        log("Start validating")
        model.eval()
        with torch.no_grad():
            valid_acc = 0
            for batch in tqdm(valid_dataloader):
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                acc = (predictions == batch['labels']).float().mean().item()
                valid_acc += acc
            log(f"Validation | Epoch {epoch + 1} | acc = {valid_acc / len(valid_dataloader):.3f}")

log("Save model")
model.save_pretrained(args.output_dir)