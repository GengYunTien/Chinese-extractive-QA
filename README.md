# Chinese-extractive-QA
### Task Description - Chinese Extractive Question Answering
The model training process is divided into two stages. In the first stage, the model selects the most relevant paragraph from four given paragraphs based on the question. In the second stage, the model predicts the answer to the question from the selected paragraph.
## Download model, data and tokenizers
```
cd r11945043
bash download.sh
```
## Script
```
./r11945043
├── download
│   ├── paragraphSelection
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer.json
│   ├── questionAnswering
│   │   ├── model1
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   └── tokenizer.json
│   │   ├── model2
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   └── tokenizer.json
│   ├── context.json
│   ├── train.json
│   ├── valid.json
│   └── test.json
├── README.md
├── run.sh
├── download.sh
├── report.pdf
├── paragraphSelection.py
├── questionAnswering_1.py
├── questionAnswering_2.py
└── inference.py
```

## Stage 1 Training: Paragraph selection
* Create output_dir
```
mkdir paragraphSelection
```
* Pre-trained model: hfl/chinese-bert-wwm-ext
```
python paragraphSelection.py \
--context_file './download/context.json' \
--train_file './download/train.json' \
--validation_file './download/valid.json' \
--model_name_or_path './huggingface/hfl--chinese-bert-wwm-ext' \
--output_dir './paragraphSelection' \
--seed 42 \
```
## Stage 2 Training: Question answering
For improve performance, two pre-trained model were used to implement ensemble method. plot figure function was included in the `questionAnswering_1.py` and `questionAnswering_2.py`
* Create output_dir
```
mkdir -p questionAnswering/{model1,model2}
```
* Pre-trained model1: hfl/chinese-lert-large
```
python questionAnswering_1.py \
--context_file './download/context.json' \
--train_file './download/train.json' \
--validation_file './download/valid.json' \
--model_name_or_path './huggingface/hfl--chinese-lert-large' \
--output_dir './questionAnswering/model1' \
--seed 42 \
```
* Pre-trained model2: hfl/chinese-macbert-large
```
python model2.py \
--context_file './download/context.json' \
--train_file './download/train.json' \
--validation_file './download/valid.json' \
--model_name_or_path './huggingface/hfl--chinese-macbert-large' \
--output_dir './questionAnswering/model2' \
--seed 42 \
```
## Testing
```
python inference.py \
--context_file './download/context.json' \
--test_file './download/test.json' \
--PS_model_path './paragraphSelection' \
--QA_model1_path './questionAnswering/model1' \
--QA_model2_path '/questionAnswering/model2' \
--output_file './result.csv' \
--seed 42 \
```

# Run run.sh 
```
bash run.sh ./download/context.json ./download/test.json ./result.csv
```