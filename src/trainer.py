import os
import math
import json
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.data_utils import collate_fn

class RankingTrainer():
    def __init__(self,
                 config,
                 model,
                 train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,):
        self.config = config
        
        # device config
        if self.config.use_gpu:
            self.config.device = torch.device(config.device)
        else:
            self.config.device = torch.device("cpu")
        
        # logger config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"logs/{self.config.experiment_name}.log", mode="w"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # prepare data loader and target documents        
        if train_dataset is None and test_dataset is None:
            self.logger.error("At least one dataset should be passed")
            raise FileNotFoundError
        if train_dataset is not None and valid_dataset is not None:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Training set length: {len(train_dataset)}")
            self.val_dataloader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Validation set length: {len(valid_dataset)}")
        elif train_dataset is not None and valid_dataset is None:
            datasets = random_split(train_dataset, [int(0.9 * len(train_dataset)), len(train_dataset) - int(0.9 * len(train_dataset))])
            # datasets = random_split(train_dataset, [int(0.001 * len(train_dataset)), int(0.001 * len(train_dataset)), len(train_dataset) - 2 * int(0.001 * len(train_dataset))])
            train_dataset = datasets[0]
            valid_dataset = datasets[1]
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Training set length: {len(train_dataset)}")
            self.val_dataloader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Validation set length: {len(valid_dataset)}")
        if test_dataset is not None:
            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Test set length: {len(test_dataset)}")
        self.logger.info("Data init done.")
        
        # prepare model, preprocessor, loss, optimizer and scheduler
        self.model = model
        # self.margin = self.config.min_margin
        # self.loss = nn.MarginRankingLoss(margin=self.margin)
        self.loss = nn.MarginRankingLoss(margin=0.2)
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )
        
        if config.resume_path is None:
            self.logger.warning("No checkpoint given!")
            self.best_val_loss = 10000
            self.best_accu = 0
            self.last_epoch = -1
            num_training_steps = self.config.num_epochs * len(self.train_dataloader)
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=self.last_epoch
            )
        else:
            self.logger.info(f"Loading model from checkpoint: {self.config.resume_path}.")
            checkpoint = torch.load(self.config.resume_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            self.best_val_loss = checkpoint["best_val_loss"]
            self.best_accu = checkpoint["best_accu"]
            self.last_epoch = -1
            num_training_steps = self.config.num_epochs * len(self.train_dataloader)
            self.optimizer.load_state_dict(checkpoint["optimizier"])
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=self.last_epoch
            )
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.model.to(self.config.device)
        
        self.logger.info("Trainer init done.")

    def train_one_epoch(self):
        total_loss = 0
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        progress_bar = tqdm(range(num_update_steps_per_epoch))
        for step, batch in enumerate(self.train_dataloader):
            contexts, answers, false_answers = batch
            logits = self.model(contexts, answers).squeeze()
            false_logits = self.model(contexts, false_answers).squeeze()
            labels = torch.tensor([1] * len(contexts)).float().to(self.config.device)
            loss = self.loss(logits, false_logits, labels)
            
            total_loss += loss.detach().cpu().item()
            loss.backward()
            if (step != 0 and step % self.config.gradient_accumulation_steps == 0) or step == len(self.train_dataloader) - 1:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.set_postfix(
                    {
                        'loss': total_loss / (step + 1),
                    }
                )
                progress_bar.update(1)
            if self.config.margin_increase_step > 0 and step != 0 and step % self.config.margin_increase_step == 0 and self.margin < self.config.max_margin:
                self.margin += 0.1
                self.loss = nn.MarginRankingLoss(self.margin)
        return total_loss / (step + 1)
          
    def train(self):
        for epoch in range(self.config.num_epochs):
            self.logger.info("========================")
            self.logger.info("Training...")
            epoch_loss = self.train_one_epoch()
            self.logger.info(f"Epoch {epoch} training loss: {epoch_loss}")
            self.logger.info("Validating...")
            val_loss, val_accu = self.validate()
            self.logger.info(f"Epoch {epoch} validation loss: {val_loss}")
            self.logger.info(f"Epoch {epoch} validation accuracy: {val_accu}")
            # if val_loss < self.best_val_loss:
            if val_accu > self.best_accu:
                self.best_val_loss = val_loss
                self.best_accu = val_accu
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizier": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "last_epoch": self.last_epoch + epoch + 1,
                    "best_val_loss": self.best_val_loss,
                    "best_accu": self.best_accu,
                    "config": self.config,
                }
                save_path = os.path.join(self.config.save_dir, f"{self.config.experiment_name}.pt")
                self.logger.info(f"Saving best checkpoint to {save_path}")
                torch.save(checkpoint, save_path)
    
    @torch.no_grad()
    def validate(self):
        total_loss = 0
        cnt_true, cnt = 0, 0
        for step, batch in tqdm(enumerate(self.val_dataloader)):
            contexts, answers, false_answers = batch
            logits = self.model(contexts, answers).squeeze()
            false_logits = self.model(contexts, false_answers).squeeze()
            labels = torch.tensor([1] * len(contexts)).float().to(self.config.device)
            loss = self.loss(logits, false_logits, labels)
            total_loss += loss.detach().cpu().item()
            
            b_cnt_true, b_cnt = self.evaluate(logits.detach().cpu(), false_logits.detach().cpu())
            cnt_true += b_cnt_true
            cnt += b_cnt        
                
        return total_loss / (step + 1), cnt_true / cnt
    
    @torch.no_grad()
    def evaluate(self, logits, false_logits):
        cnt_true, cnt = 0, 0
        for pred, truth in zip(logits, false_logits):
            if pred > truth:
                cnt_true += 1
            cnt += 1
        return cnt_true, cnt
    
    @torch.no_grad()
    def metric(self, tp, tn, fp, fn):
        try:
            accu = (tp + tn) / (tp + tn + fp + fn)
        except:
            accu = 0
        try:
            prec = tp / (tp + fp)
        except:
            prec = 0
        try:
            reca = tp / (tp + fn)
        except:
            reca = 0
        try:
            f1 = 2 * prec * reca / (prec + reca)
        except:
            f1 = 0
        print(f"Accuracy: {accu}")
        print(f"Preision: {prec}")
        print(f"Recall: {reca}")
        print(f"F1: {f1}")
        return accu, prec, reca, f1
    
class RankingMultipleNegativeTrainer():
    def __init__(self,
                 config,
                 model,
                 train_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,):
        self.config = config
        
        # device config
        if self.config.use_gpu:
            self.config.device = torch.device(config.device)
        else:
            self.config.device = torch.device("cpu")
        
        # logger config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"logs/{self.config.experiment_name}.log", mode="w"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(self.config)
        
        # prepare data loader and target documents        
        if train_dataset is None and test_dataset is None:
            self.logger.error("At least one dataset should be passed")
            raise FileNotFoundError
        if train_dataset is not None and valid_dataset is not None:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Training set length: {len(train_dataset)}")
            self.val_dataloader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Validation set length: {len(valid_dataset)}")
        elif train_dataset is not None and valid_dataset is None:
            datasets = random_split(train_dataset, [int(0.9 * len(train_dataset)), len(train_dataset) - int(0.9 * len(train_dataset))])
            # datasets = random_split(train_dataset, [int(0.001 * len(train_dataset)), int(0.001 * len(train_dataset)), len(train_dataset) - 2 * int(0.001 * len(train_dataset))])
            train_dataset = datasets[0]
            valid_dataset = datasets[1]
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Training set length: {len(train_dataset)}")
            self.val_dataloader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Validation set length: {len(valid_dataset)}")
        if test_dataset is not None:
            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
            self.logger.info(f"Test set length: {len(test_dataset)}")
        self.logger.info("Data init done.")
        
        # prepare model, preprocessor, loss, optimizer and scheduler
        self.model = model
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )
        
        if config.resume_path is None:
            self.logger.warning("No checkpoint given!")
            self.best_val_loss = 10000
            # self.best_accu = 0
            self.last_epoch = -1
            num_training_steps = self.config.num_epochs * len(self.train_dataloader)
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=self.last_epoch
            )
        else:
            self.logger.info(f"Loading model from checkpoint: {self.config.resume_path}.")
            checkpoint = torch.load(self.config.resume_path, map_location=self.config.device) 
            self.model.to(self.config.device)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            # self.best_val_loss = checkpoint["best_val_loss"]
            self.best_val_loss = 10000
            # self.best_accu = checkpoint["best_accu"]
            self.last_epoch = checkpoint["last_epoch"]
            num_training_steps = self.config.num_epochs * len(self.train_dataloader)
            # self.optimizer.load_state_dict(checkpoint["optimizier"])
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=-1
            )
            # self.scheduler.load_state_dict(checkpoint["scheduler"])
            del(checkpoint)
        self.model.to(self.config.device)
        
        self.logger.info("Trainer init done.")

    def train_one_epoch(self):
        total_loss = 0
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        progress_bar = tqdm(range(num_update_steps_per_epoch))
        for step, batch in enumerate(self.train_dataloader):
            contexts, answers = batch
            answers_0 = [answer["0"] for answer in answers] # positive
            answers_1 = [answer["2"] for answer in answers] # negative 1, set margin to 0.3
            answers_2 = [answer["1"] for answer in answers] # negative 2, set margin to 0.6
            answers_3 = [answer["3"] for answer in answers] # negative 3, set margin to 0.6
            
            logits_0 = self.model(contexts, answers_0)
            logits_1 = self.model(contexts, answers_1)
            logits_2 = self.model(contexts, answers_2)
            logits_3 = self.model(contexts, answers_3)
            logits = torch.cat([logits_0, logits_1, logits_2, logits_3], dim=1)
            
            labels = torch.tensor([1] * len(answers_0)).float().to(self.config.device)
            labels = labels.unsqueeze(1)
            loss = nn.MarginRankingLoss(0.3)(logits_0, logits_1, labels) + nn.MarginRankingLoss(0.6)(logits_0, logits_2, labels) + nn.MarginRankingLoss(0.9)(logits_0, logits_3, labels)
            
            total_loss += loss.detach().cpu().item()
            loss.backward()
            if (step != 0 and step % self.config.gradient_accumulation_steps == 0) or step == len(self.train_dataloader) - 1:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.set_postfix(
                    {
                        'loss': total_loss / (step + 1),
                    }
                )
                progress_bar.update(1)
            # if self.config.margin_increase_step > 0 and step != 0 and step % self.config.margin_increase_step == 0 and self.margin < self.config.max_margin:
            #     self.margin += 0.1
            #     self.loss = nn.MarginRankingLoss(self.margin)
        return total_loss / (step + 1)
          
    def train(self):
        for epoch in range(self.config.num_epochs):
            self.logger.info("========================")
            self.logger.info("Training...")
            epoch_loss = self.train_one_epoch()
            self.logger.info(f"Epoch {epoch} training loss: {epoch_loss}")
            self.logger.info("Validating...")
            val_loss = self.validate() # , val_accu
            self.logger.info(f"Epoch {epoch} validation loss: {val_loss}")
            # self.logger.info(f"Epoch {epoch} validation accuracy: {val_accu}")
            if val_loss < self.best_val_loss:
            # if val_accu > self.best_accu:
                self.best_val_loss = val_loss
                # self.best_accu = val_accu
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizier": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "last_epoch": self.last_epoch + epoch + 1,
                    "best_val_loss": self.best_val_loss,
                    # "best_accu": self.best_accu,
                    "config": self.config,
                }
                save_path = os.path.join(self.config.save_dir, f"{self.config.experiment_name}.pt")
                self.logger.info(f"Saving best checkpoint to {save_path}")
                torch.save(checkpoint, save_path)
    
    @torch.no_grad()
    def validate(self):
        total_loss = 0
        # cnt_true, cnt = 0, 0
        for step, batch in tqdm(enumerate(self.val_dataloader)):
            contexts, answers = batch
            answers_0 = [answer["0"] for answer in answers] # positive
            answers_1 = [answer["2"] for answer in answers] # negative 1, set margin to 0.3
            answers_2 = [answer["1"] for answer in answers] # negative 2, set margin to 0.6
            answers_3 = [answer["3"] for answer in answers] # negative 3, set margin to 0.6
            
            logits_0 = self.model(contexts, answers_0)
            logits_1 = self.model(contexts, answers_1)
            logits_2 = self.model(contexts, answers_2)
            logits_3 = self.model(contexts, answers_3)
            
            labels = torch.tensor([1] * len(answers_0)).float().to(self.config.device)
            labels = labels.unsqueeze(1)
            loss = nn.MarginRankingLoss(0.3)(logits_0, logits_1, labels) + nn.MarginRankingLoss(0.6)(logits_0, logits_2, labels) + nn.MarginRankingLoss(0.9)(logits_0, logits_3, labels)
            
            total_loss += loss.detach().cpu().item()
                
        return total_loss / (step + 1) # , cnt_true / cnt
    
    @torch.no_grad()
    def evaluate(self, logits, false_logits):
        cnt_true, cnt = 0, 0
        for pred, truth in zip(logits, false_logits):
            if pred > truth:
                cnt_true += 1
            cnt += 1
        return cnt_true, cnt
    
    @torch.no_grad()
    def metric(self, tp, tn, fp, fn):
        try:
            accu = (tp + tn) / (tp + tn + fp + fn)
        except:
            accu = 0
        try:
            prec = tp / (tp + fp)
        except:
            prec = 0
        try:
            reca = tp / (tp + fn)
        except:
            reca = 0
        try:
            f1 = 2 * prec * reca / (prec + reca)
        except:
            f1 = 0
        print(f"Accuracy: {accu}")
        print(f"Preision: {prec}")
        print(f"Recall: {reca}")
        print(f"F1: {f1}")
        return accu, prec, reca, f1