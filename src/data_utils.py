import json
from torch.utils.data import Dataset

def collate_fn(batch):
    return tuple(zip(*batch))

class GSM8KDataset(Dataset):
    def __init__(self, datapath, mode="train") -> None:
        if mode == "train":
            with open(datapath, "r") as fin:
                self.datas = json.load(fin)
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        """
        {
            "context": ,
            "answer": ,
            "label": 1 for entail; 0 for neutral
        }
        """
        data = self.datas[index]
        return data["context"], data["answer"], data["label"]
    
class GSM8KRankingDataset(Dataset):
    def __init__(self, datapath, mode="train") -> None:
        if mode == "train":
            with open(datapath, "r") as fin:
                self.datas = json.load(fin)
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        """
        {
            "context": ,
            "answer": ,
            "false_answer": 1 for entail; 0 for neutral
        }
        """
        data = self.datas[index]
        return data["context"], data["answer"], data["false_answer"]
    
class GSM8KRankingMultiNegativeDataset(Dataset):
    def __init__(self, datapath, mode="train") -> None:
        if mode == "train":
            with open(datapath, "r") as fin:
                self.datas = json.load(fin)
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        """
        {
            "context": ,
            "answers": , "0", "1", "2", "3"
        }
        """
        data = self.datas[index]
        return data["context"], data["answers"]