import random
import torch
import json
import numpy as np

from itertools import chain

# define simple logging functionality
log_fw = open("log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

# Set Seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# load data (paragraph selection)
def load_data(file):
    with open(file, 'r', encoding = 'utf-8') as readFile:
        data = json.load(readFile)
    return data

# process file (paragraph selection)
def preprocess_function(contextFile, File, tokenizer, max_seq_length):
    questions = [i["question"] for i in File]
    first_sentences = [[q] * 4 for q in questions]
    paragraphs_ids = [i["paragraphs"] for i in File]
    second_sentences = [[contextFile[p_id] for p_id in i] for i in paragraphs_ids]
    labels = [i["paragraphs"].index(i["relevant"]) for i in File]

    # Flatten out
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )
    # Un-flatten
    tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_inputs["labels"] = labels

    return tokenized_inputs

# load & process data (span selection)
def process_json(json):
    for i in json:
        i.pop("paragraphs", None)
        i["answer"]["end"] = i["answer"]["start"] + len(i["answer"]["text"]) - 1
    return json

def read_data(contextfile, trainfile, validfile):
    with open(contextfile, 'r', encoding = 'utf-8') as contextF, open(trainfile, 'r', encoding = 'utf-8') as trainF, open(validfile, 'r', encoding = 'utf-8') as validF:
        context = json.load(contextF)
        train = json.load(trainF)
        valid = json.load(validF)
        train = process_json(train)
        valid = process_json(valid)
    return context, train, valid

# Evaluation function (span selection)
def evaluate(data, output, tokenized_paragraph, paragraph, doc_stride):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    max_answer_length = 35

    token_offset = data[1][0][0].tolist().index(1)
    
    for k in range(num_of_windows):
        start_probs, start_indexs = torch.topk(output.start_logits[k], k=1, dim=0)
        for start_prob, start_index in zip(start_probs, start_indexs):
            length_prob, length = torch.max(output.end_logits[k][start_index : start_index + max_answer_length], dim=0)
            prob = start_prob + length_prob

            if prob > max_prob:
                max_prob = prob
                start_token_index = start_index - token_offset + k * doc_stride
                end_token_index = start_index + length - token_offset + k * doc_stride
                try:
                    start_char_index = tokenized_paragraph.token_to_chars(start_token_index)[0]
                    end_char_index = tokenized_paragraph.token_to_chars(end_token_index)[1]
                    answer = paragraph[start_char_index : end_char_index]
                except:
                    pass

    if '「' in answer and '」' not in answer:
        answer += '」'
    elif '「' not in answer and '」' in answer:
        answer = '「' + answer
    if '《' in answer and '》' not in answer:
        answer += '》'
    elif '《' not in answer and '》' in answer:
        answer = '《' + answer

    return answer.replace(',','')