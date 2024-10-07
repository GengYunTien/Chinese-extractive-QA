#!/bin/bash   

# Parse arguments
CONTEXT_PATH=$1
TEST_PATH=$2
OUTPUT_PATH=$3

PS_MODEL_DIR="./download/paragraphSelection"
QA_MODEL1_DIR="./download/questionAnswering/model1"
QA_MODEL2_DIR="./download/questionAnswering/model2"

PYTHON_SCRIPT="inference.py"

python $PYTHON_SCRIPT \
--context_file $CONTEXT_PATH \
--test_file $TEST_PATH \
--PS_model_path $PS_MODEL_DIR \
--QA_model1_path $QA_MODEL1_DIR \
--QA_model2_path $QA_MODEL2_DIR \
--output_file $OUTPUT_PATH \