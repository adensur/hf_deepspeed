import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser
from typing import Dict, List, Any, Tuple, Optional
import random
from model import MyModel
from dataclasses import dataclass, field
import torch.distributed as dist
import sys
import os
@dataclass
class Arguments(TrainingArguments):
    train_path: str = field(default="data/train.jsonl", metadata={"help": "Path to training data"})
    val_path: str = field(default="data/validation.jsonl", metadata={"help": "Path to validation data"})
    model_name_or_path: str = field(default="intfloat/simlm-base-msmarco", metadata={"help": "Model name or path"})
    q_max_len: int = field(default=32, metadata={"help": "Max query length"})
    p_max_len: int = field(default=144, metadata={"help": "Max passage length"})
    temperature: float = field(default=0.02, metadata={"help": "Temperature for contrastive loss"})
    train_n_passages: int = field(default=16, metadata={"help": "Number of passages per query for training"})

def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors

class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data = []
        with open(path, "r") as f:
            for line in tqdm(f, desc="Loading data"):
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(tokenizer, args, examples: List[Dict[str, Any]]):
    pad_to_multiple_of=8 if args.fp16 else None
    # Extract queries and all docs (positive first, then negatives)
    queries = [ex["query"] for ex in examples]
    
    # Combine positive and negative docs for each example
    all_docs = []
    for ex in examples:
        pos_doc = ex["pos_doc"] # Take random positive doc
        neg_docs = random.sample(ex["neg_doc"], args.train_n_passages - 1)  # Take n-1 random negative docs
        docs = [pos_doc] + neg_docs  # Positive doc first, then negatives
        all_docs.extend(docs)
    
    # Tokenize all queries in one batch
    query_encodings = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=args.q_max_len,
        return_tensors="pt",
        pad_to_multiple_of=pad_to_multiple_of,
    )

    # Tokenize all docs in one batch
    doc_encodings = tokenizer(
        all_docs,
        padding=True,
        truncation=True,
        max_length=args.p_max_len,
        return_tensors="pt",
        pad_to_multiple_of=pad_to_multiple_of,
    )

    return {
        "query": query_encodings,
        "docs": doc_encodings
    }


class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_embeds = model.encode(inputs["query"])
        doc_embeds = model.encode(inputs["docs"])

        all_query_embeds = dist_gather_tensor(query_embeds)
        all_doc_embeds = dist_gather_tensor(doc_embeds)

        # Compute scores and labels
        scores, labels = self.full_contrastive_scores_and_labels(
            query=all_query_embeds,
            key=all_doc_embeds,
            train_n_passages=self.args.train_n_passages
        )
        
        # Scale scores by temperature
        scores = scores / self.args.temperature
        
        # Compute loss
        loss = F.cross_entropy(scores, labels)
        
        return (loss, scores) if return_outputs else loss

    # we have to override this, because "inputs" format is custom and cannot be fed directly into the model
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = False
        
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            
        return (loss, logits, None)

    def full_contrastive_scores_and_labels(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            train_n_passages: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

        labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
        labels = labels * train_n_passages

        # batch_size x (batch_size x n_psg)
        qk = torch.mm(query, key.t())

        # batch_size x dim
        sliced_key = key.index_select(dim=0, index=labels)
        assert query.shape[0] == sliced_key.shape[0]

        # batch_size x batch_size
        kq = torch.mm(sliced_key, query.t())
        kq.fill_diagonal_(float('-inf'))

        qq = torch.mm(query, query.t())
        qq.fill_diagonal_(float('-inf'))

        kk = torch.mm(sliced_key, sliced_key.t())
        kk.fill_diagonal_(float('-inf'))

        scores = torch.cat([qk, kq, qq, kk], dim=-1)

        return scores, labels

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("Saving model checkpoint to {}".format(output_dir))
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)


def main():
    print("argv: ", " ".join(sys.argv))
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = MyModel(args.model_name_or_path)
    
    train_dataset = JsonlDataset(args.train_path)
    val_dataset = JsonlDataset(args.val_path)

    trainer = ContrastiveTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda x: collate_fn(tokenizer, args, x),
        tokenizer=tokenizer,
    )

    trainer.train()
    
    trainer.save_model()
    #tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
