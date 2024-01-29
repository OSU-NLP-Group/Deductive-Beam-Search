import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class Verifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.verifier_model_path)
        self.model = AutoModel.from_pretrained(config.verifier_model_path)
        
        self.decision_layers = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1), # 0 for neutral, 1 for entail
            # nn.Softmax(),
            nn.Sigmoid(),
        )

    def forward(self, contexts, rationales):
        inputs = [f"[CLS]{c}[SEP]{r}[SEP]" for c, r in zip(contexts, rationales)]
        tokenized = self.tokenizer(
            inputs,
            add_special_tokens=False,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.config.device)
        attention_mask = tokenized.attention_mask.to(self.config.device)
        encoded = self.model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        # print(encoded)
        logits = self.decision_layers(encoded)
        return logits
    
    def train(self, contexts, rationales, labels):
        logits = self.forward(contexts, rationales).squeeze()
        # print(logits)
        # print(labels)
        # input()
        return logits, torch.tensor(labels).float().to(logits.device)
    
    def from_pretrained(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        pretrained_dict = checkpoint["model"]
        model_dict = self.model.state_dict()
        pretrained_dict = {k.replace("model.", ""): v for k, v in pretrained_dict.items() if k.replace("model.", "") in model_dict}
        self.model.load_state_dict(pretrained_dict)
        
        pretrained_dict = checkpoint["model"]
        model_dict = self.decision_layers.state_dict()
        pretrained_dict = {k.replace("decision_layers.", ""): v for k, v in pretrained_dict.items() if k.replace("decision_layers.", "") in model_dict}
        self.decision_layers.load_state_dict(pretrained_dict)
        return self
    
class Ranker(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.verifier_model_path)
        self.model = AutoModel.from_pretrained(config.verifier_model_path)
        
        self.decision_layers = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1), # 0 for neutral, 1 for entail
            nn.Sigmoid(),
        )

    def forward(self, contexts, rationales):
        inputs = [f"[CLS]{c}[SEP]{r}[SEP]" for c, r in zip(contexts, rationales)]
        tokenized = self.tokenizer(
            inputs,
            add_special_tokens=False,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.config.device)
        attention_mask = tokenized.attention_mask.to(self.config.device)
        encoded = self.model(input_ids, attention_mask).last_hidden_state[:, 0, :]
        # print(encoded)
        logits = self.decision_layers(encoded)
        return logits
    
    def forward_train(self, contexts, rationales, labels):
        logits = self.forward(contexts, rationales).squeeze()
        # print(logits)
        # print(labels)
        # input()
        return logits, torch.tensor(labels).to(logits.device)
    
    def from_pretrained(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        pretrained_dict = checkpoint["model"]
        model_dict = self.model.state_dict()
        pretrained_dict = {k.replace("model.", ""): v for k, v in pretrained_dict.items() if k.replace("model.", "") in model_dict}
        self.model.load_state_dict(pretrained_dict)
        
        pretrained_dict = checkpoint["model"]
        model_dict = self.decision_layers.state_dict()
        pretrained_dict = {k.replace("decision_layers.", ""): v for k, v in pretrained_dict.items() if k.replace("decision_layers.", "") in model_dict}
        self.decision_layers.load_state_dict(pretrained_dict)
        return self