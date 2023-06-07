import torch
import numpy as np
from transformers import AdamW, get_scheduler
from transformers import GitConfig, GitVisionConfig
from dataloader import SNV3Dataset, collate_fn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


lr = 3e-5
batch_size = 2
num_clips = 2



class GitLightning(pl.LightningModule):
    def __init__(self, model, processor, train_ds):
        super().__init__()
        self.model = model
        self.processor = processor
        self.train_ds = train_ds
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=lr)
    
    def forward(self, x):
        out = self.model(**x)
        return out
    
    def training_step(self, batch, batch_idx): 
        batch_data = {}
        batch_data['pixel_values'] = batch['clip'].permute(0, 1, 4, 2, 3).float().to(self.device)
        input_text = [self.train_ds.text_processor.detokenize(x.tolist()) for x in batch['caption']]
        batch_data['input_ids'] = torch.tensor(self.processor(text=input_text, add_special_tokens=False, padding='max_length').input_ids).to(self.device)
        batch_data['labels'] = batch_data['input_ids'].to(self.device)
        outputs = self(batch_data)
        loss = outputs.loss
        self.log("train_loss", loss, on_epoch=True)
        self.training_step_outputs.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_data = {}
        batch_data['pixel_values'] = batch['clip'].permute(0, 1, 4, 2, 3).float().to(self.device)
        input_text = [self.train_ds.text_processor.detokenize(x.tolist()) for x in batch['caption']]
        batch_data['input_ids'] = torch.tensor(self.processor(text=input_text, add_special_tokens=False, padding='max_length').input_ids).to(self.device)
        batch_data['labels'] = batch_data['input_ids'].to(self.device)
        outputs = self(batch_data)
        loss = outputs.loss  
        self.log("val_loss", loss, on_epoch=True)
        self.validation_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        print('training epoch')
        loss = self.training_step_outputs
        epoch_loss = torch.stack(loss).mean()   # Combine losses
        print("Epoch Training loss : ", epoch_loss)
        self.training_step_outputs.clear()
        
    def on_validation_epoch_end(self):
        print('validation epoch')
        loss = self.validation_step_outputs
        epoch_loss = torch.stack(loss).mean()   # Combine losses
        print("Epoch Validation loss : ", epoch_loss)
        self.validation_step_outputs.clear()



if __name__ == "__main__":
    
    
    train_ds = SNV3Dataset(split='train',
                  bucket='soccernet-230',
                  vocab_path='vocab_files/train_vocab.pyi',
                  num_clips=num_clips, window_size=3,
                  include_vid=True,
                     local_video_path="/raid/videos")
    train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=batch_size, num_workers=10)
    
    val_ds = SNV3Dataset(split='valid',
                  bucket='soccernet-230',
                  vocab_path='vocab_files/train_vocab.pyi',
                  num_clips=num_clips, window_size=3,
                  include_vid=True,
                     local_video_path="/raid/videos")
    val_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=batch_size, num_workers=10)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    print("Making Trainer")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="{epoch:02d}-{val_loss:2f}",
        save_top_k=2,
    )

    trainer = pl.Trainer(
        # strategy=None,
        accelerator="gpu",
        devices=4,
        benchmark=True,
        deterministic=False,
        num_sanity_val_steps=0,
        max_epochs=500,
        log_every_n_steps=1,
        callbacks=checkpoint_callback,
    )
    
    print("Creating Model")
    # video captioning
    processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")

    model = GitLightning(model, processor, train_ds)
    model.to(device)

    print("Training Model")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)