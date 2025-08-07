from transformers import RobertaTokenizer

RobertaTokenizer.from_pretrained("roberta-large").save_pretrained("./models/roberta-large")
