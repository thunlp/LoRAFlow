import os
import json
from typing import *


import torch
from torch.utils.data import IterableDataset, Dataset
from tqdm import tqdm

from transformers.tokenization_utils import PreTrainedTokenizer
import copy
import random
import bmtrain as bmt



def load_train_dataset(data_path):
    new_data = []
    idx=0
    if "math" in data_path:
        input_field = "query"
        output_filed = "response"
    else:
        input_field = "question"
        output_filed = "answer"
    data_name = os.path.basename(data_path).split(".")[0]
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            idx+=1
            temp_id = f"{data_name}_{idx}"
            if "math" in data_path:
                temp_input = data.get(input_field).strip()
                temp_output = data.get(output_filed).strip()
            elif "code" in data_path:
                temp_input = data.get("data")[0].strip()
                temp_output = data.get("data")[1].strip()
            else:
                raise ValueError("Unexpected data_path")
            if temp_input == '' and temp_output == '':
                continue
            temp_data = {'id': temp_id, 'data': [temp_input, temp_output]}
            new_data.append(temp_data)
    return new_data


    
IGNORE_INDEX=-100


def collator(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)
    ids = [instance["id"] for instance in instances]
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        ids = ids,
    )


class PromptIterableDataset(IterableDataset):
    def __init__(self,
                 raw_dataset: Union[Dataset, List],
                 sep: List = ["EOS", "\n"],
                 tokenizer: PreTrainedTokenizer = None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = True,
                 truncate_method: Optional[str] = "tail",
                 system_prompt: Optional[str] = None,
                ):
        assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.sep = sep
        self._end_token = None
        self.start_token = self.sep[-1]
        self.teacher_forcing = teacher_forcing
        assert self.teacher_forcing, bmt.print_rank("must use teacher forcing")

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        assert self.truncate_method == "tail", bmt.print_rank("only tail truncate support")
    

    
    @property
    def end_token(self):
        return self.tokenizer.eos_token

    def tokenize_example(self, example):
        if self.system_prompt is not None and ("math" in example["id"] or "code" in example["id"]):
            system = "<s>[INST] <<SYS>>\n"+ self.system_prompt +"\n<</SYS>>\n\n"
        else:
            system = "<s>[INST] "

        labels = []
        tokenized_ids = []
        for i, c in enumerate(example["data"]):
            if i == 0:
                # system and 1st user message
                c_input = system + c + " [/INST]"
                c_input += "让我们一步一步地思考。"
                tmp_tokenized_ids = self.tokenizer(c_input, add_special_tokens=False)["input_ids"]
                tokenized_ids += tmp_tokenized_ids
                labels += [IGNORE_INDEX] * len(tmp_tokenized_ids)
            elif i % 2 == 1:
                # model
                c_input = c + " </s>"
                tmp_tokenized_ids = self.tokenizer(c_input, add_special_tokens=False)["input_ids"]
                tokenized_ids += tmp_tokenized_ids
                labels += tmp_tokenized_ids
            else:
                # user
                c_input = "<s>[INST] " + c + " [/INST]"
                tmp_tokenized_ids = self.tokenizer(c_input, add_special_tokens=False)["input_ids"]
                tokenized_ids += tmp_tokenized_ids
                labels += [IGNORE_INDEX] * len(tmp_tokenized_ids)

        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels), "id": example["id"]}

    def pad_truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        tokenized_example["attention_mask"] = torch.LongTensor([1]*len(tokenized_example["input_ids"]))
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                if k == "id":
                    continue
                tokenized_example[k] = tokenized_example[k][:-(old_len - self.max_seq_length)]
        elif old_len < self.max_seq_length:
            tokenized_example["input_ids"] = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]*(self.max_seq_length - old_len)), tokenized_example["input_ids"]])
            tokenized_example["labels"] = torch.cat([torch.LongTensor([IGNORE_INDEX]*(self.max_seq_length - old_len)), tokenized_example["labels"]])
            tokenized_example["attention_mask"] = torch.LongTensor([0]*(self.max_seq_length - old_len) + [1]*old_len)
        assert len(tokenized_example["input_ids"]) == len(tokenized_example["labels"]) == len(tokenized_example["attention_mask"]) == self.max_seq_length
        return tokenized_example


    def __iter__(self):
        for example in self.raw_dataset:
            tokenized_example = self.tokenize_example(example)
            tokenized_example = self.pad_truncate(tokenized_example)
            yield tokenized_example

    def __len__(self):
        return len(self.raw_dataset)


if __name__ == "__main__":
    pass