from transformers import AutoModel
import torch.nn.functional as F
import torch

class MyModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, batch_dict):
        preds = self.model(**batch_dict)
        hidden_state = preds.last_hidden_state
        embeds = hidden_state[:, 0]
        embeds = F.normalize(embeds, dim=-1)
        embeds = embeds.contiguous()
        return embeds

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
