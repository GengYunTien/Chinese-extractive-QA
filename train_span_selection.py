import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from tqdm.auto import tqdm

import transformers
from transformers import AdamW
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from modules.utils import log, set_seed, read_data, evaluate
from modules.datasets import SpanSelectionDataset

parser = argparse.ArgumentParser(description="Finetune a transformers model on a question answering task")
parser.add_argument("--context_file", type=str, default=None, help="A csv or a json file containing the context data.")
parser.add_argument("--train_file", type=str, default=None, help="A csv or a json file containing the training data.")
parser.add_argument("--validation_file", type=str, default=None, help="A csv or a json file containing the validation data.")
parser.add_argument("--doc_stride", type=int, default=350)
parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
parser.add_argument("--per_device_valid_batch_size", type=int, default=1, help="Batch size (per device) for the validation dataloader.")
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
model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path).to(device)

# load & process data
paragraphs, train_questions, valid_questions = read_data(args.context_file, args.train_file, args.validation_file)

# tokenize data
train_questions_tokenized = tokenizer([q["question"] for q in train_questions], add_special_tokens=False)
valid_questions_tokenized = tokenizer([q["question"] for q in valid_questions], add_special_tokens=False)
paragraphs_tokenized = tokenizer(paragraphs, add_special_tokens=False)

# dataset & dataloader
train_dataset = SpanSelectionDataset("train", train_questions, train_questions_tokenized, paragraphs_tokenized, args.doc_stride)
valid_dataset = SpanSelectionDataset("valid", valid_questions, valid_questions_tokenized, paragraphs_tokenized, args.doc_stride)

train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_valid_batch_size, shuffle=False, pin_memory=True)

# Optimizer
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
            output = model(input_ids=batch[0], token_type_ids=batch[1], attention_mask=batch[2], start_positions=batch[3], end_positions=batch[4])
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            acc = ((start_index == batch[3]) & (end_index == batch[4])).float().mean()
            loss = output.loss

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
            for i, batch in enumerate(tqdm(valid_dataloader)):
                output = model(input_ids=batch[0].squeeze(dim=0).to(device), token_type_ids=batch[1].squeeze(dim=0).to(device), attention_mask=batch[2].squeeze(dim=0).to(device))
                paragraph_id = valid_questions[i]["relevant"]
                valid_acc += evaluate(batch, output, paragraphs_tokenized[paragraph_id], paragraphs[paragraph_id], args.doc_stride) == valid_questions[i]["answer"]["text"]
            log(f"Validation | Epoch {epoch + 1} | acc = {valid_acc / len(valid_dataloader):.3f}")

log("Save model")
model.save_pretrained(args.output_dir)