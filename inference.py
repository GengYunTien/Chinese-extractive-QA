import torch
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

import transformers
from transformers import AdamW
from transformers import AutoModelForMultipleChoice, AutoModelForQuestionAnswering, AutoTokenizer, default_data_collator

from modules.utils import log, set_seed, load_data, preprocess_function, evaluate
from modules.datasets import ParagraphSelectionDataset, SpanSelectionDataset

parser = argparse.ArgumentParser()
parser.add_argument("--context_file", type=str, default=None, help="A csv or a json file containing the context data.")
parser.add_argument("--test_file", type=str, default=None, help="A csv or a json file containing the training data.")
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--doc_stride", type=int, default=350)
parser.add_argument("--PS_model_path", type=str, help="Path to pretrained model")
parser.add_argument("--QA_model1_path", type=str, help="Path to pretrained model")
parser.add_argument("--QA_model2_path", type=str, help="Path to pretrained model")
parser.add_argument("--per_device_test_batch_size", type=int, default=1, help="Batch size (per device) for the validation dataloader.")
parser.add_argument("--output_file", type=str, default=None, help="Where to store the final result.")
parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
args = parser.parse_args()

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Seed for reproducibility
set_seed(args.seed)

# load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)
psModel = AutoModelForMultipleChoice.from_pretrained(args.PS_model_path).to(device)
qaModel_paths = [args.QA_model1_path, args.QA_model2_path]
qaModels = [AutoModelForQuestionAnswering.from_pretrained(path).to(device) for path in qaModel_paths]

# paragraph selection

# load data
contextFile = load_data(args.context_file)
testFile = load_data(args.test_file)

# process file
test_data = preprocess_function(contextFile, testFile, tokenizer, args.max_seq_length)

# dataset & dataloader (paragraph selection)
test_dataset = ParagraphSelectionDataset(test_data, mode='test')
test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.per_device_test_batch_size, shuffle=False)

log('Start paragraph selection testing')
psModel.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_dataloader)):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = psModel(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu()
        testFile[i]["relevant"] = testFile[i]["paragraphs"][int(predictions)]  
log("finish paragraph selection testing")

# span selection

# Tokenize data
test_questions_tokenized = tokenizer([q["question"] for q in testFile], add_special_tokens=False)
paragraphs_tokenized = tokenizer(contextFile, add_special_tokens=False)

# dataset & dataloader (span selection)
test_dataset = SpanSelectionDataset('test', testFile, test_questions_tokenized, paragraphs_tokenized, args.doc_stride)
test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_test_batch_size, shuffle=False, pin_memory=True)

for model in qaModels:
    model.eval()

log('Start span selection testing')
result = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_dataloader)):
        all_start_logits = []
        all_end_logits = []

        for model in qaModels:
            output = model(input_ids=batch[0].squeeze(dim=0).to(device), token_type_ids=batch[1].squeeze(dim=0).to(device), attention_mask=batch[2].squeeze(dim=0).to(device))
            all_start_logits.append(output.start_logits)
            all_end_logits.append(output.end_logits)

        avg_start_logits = torch.mean(torch.stack(all_start_logits), dim=0)
        avg_end_logits = torch.mean(torch.stack(all_end_logits), dim=0)

        paragraph_id = testFile[i]["relevant"]
        result.append(evaluate(batch, avg_start_logits, avg_end_logits, paragraphs_tokenized[paragraph_id], contextFile[paragraph_id], args.doc_stride))
log('Finish span selection testing')

result_file = args.output_file
with open(result_file, 'w') as f:	
    f.write("id,answer\n")
    for i, test_question in enumerate(testFile):
    # Replace commas in answers with empty strings (since csv is separated by comma)
    # Answers in kaggle are processed in the same way
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")