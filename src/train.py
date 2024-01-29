import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

from src.parser_utils import get_parser
from src.data_utils import GSM8KRankingMultiNegativeDataset
from src.trainer import RankingMultipleNegativeTrainer
from src.model import Ranker


def ranker_multi_negavie_main():
    parser = get_parser()
    config = parser.parse_args()
    
    if config.train_datapath is not None:
        train_dataset = GSM8KRankingMultiNegativeDataset(config.train_datapath)
    else:
        train_dataset = None
    if config.valid_datapath is not None:
        valid_dataset = GSM8KRankingMultiNegativeDataset(config.valid_datapath)
    else:
        valid_dataset = None
    if config.test_datapath is not None:
        test_dataset = GSM8KRankingMultiNegativeDataset(config.test_datapath)
    else:
        test_dataset = None
    
    model = Ranker(config)
    
    trainer = RankingMultipleNegativeTrainer(
        config,
        model,
        train_dataset,
        valid_dataset,
        test_dataset
    )
    
    trainer.train()

if __name__ == "__main__":
    ranker_multi_negavie_main()