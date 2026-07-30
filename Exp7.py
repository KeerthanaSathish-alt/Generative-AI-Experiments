!pip install transformers datasets evaluate accelerate torch -q
from datasets import Dataset
from transformers import (
AutoTokenizer,
AutoModelForSequenceClassification,
TrainingArguments,
Trainer,
pipeline
)
# 1. Domain

-Specific Dataset

data = {
"text": [
"The transformer model achieved excellent accuracy.",
"Large Language Models are revolutionizing AI.",
"The football team won the championship.",
"The cricket match was exciting.",
"Neural networks are widely used in deep learning.",
"The player scored a brilliant goal.",
"Machine learning improves decision making.",
"The tennis tournament starts tomorrow."
],
"label": [
1, 1, 0, 0,
1, 0, 1, 0
]
}
dataset = Dataset.from_dict(data)
# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
"bert
-base
-uncased"

)
# 3. Tokenization
def tokenize(example):
return tokenizer(
example["text"],
truncation=True,
padding="max_length",
max_length=128
)
dataset = dataset.map(tokenize)
dataset.set_format(
type="torch",
columns=[
"input_ids",
"attention_mask",
"label"
]
)
