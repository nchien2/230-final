import torch
import numpy as np
from transformers import AdamW, get_scheduler
from transformers import GitConfig, GitVisionConfig
from modified_model_git import GitForCausalLM
from dataloader import SNV3Dataset, collate_fn
import pytorch_lightning as pl
from torch.utils.data import DataLoader



#from model_image import GitForCausalLM

num_epochs = 1
batch_size = 32
num_frames = 15
visual_embed_size = 8576
sequence_length = 400
embed_hidden_size = 768
lr = 3e-5

class plModel(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        # # update the vision encoder embedding dim
        vision_config = {"hidden_size": visual_embed_size}

        config = GitConfig(vision_config)
        config.vocab_size = vocab_size
        config.num_hidden_layers = 3
        print(config)
        self.model = GitForCausalLM(config)
        self.save_hyperparameters()

    def configure_optimizers(self):
      return AdamW(self.parameters(), lr = lr)

    def forward(self, x):
#         print('in forward')
#         print(x)
        out = self.model(**x)
        return out

    def training_step(self, batch, batch_idx):
#         position_ids = torch.tensor(range(0, batch[0]['labels'].shape[1] + 30), dtype=torch.long)
        inputs = {#"visual_embeds": torch.normal(0, 1, size=(25, 3, 768)),
                 "visual_embeds": None,
                 "inputs_embeds": batch[0]['embed'].to(self.device),
                 "labels": batch[0]['caption'].to(self.device),
                 "position_ids": None#position_ids.to(self.device)
                 }
#         print(batch[0]['embed'].shape)
#         inputs = {k:v.to(self.device) for k, v in inputs.items()}
        out = self(inputs)
        y_hat = out['logits']
        loss = out['loss']
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {#"visual_embeds": torch.normal(0, 1, size=(25, 3, 768)),
                 "visual_embeds": None,
                 "inputs_embeds": batch[0]['embed'].to(self.device),
                 "labels": batch[0]['caption'].to(self.device),
                 "position_ids": None#position_ids.to(self.device)
                 }
#         print(batch[0]['embed'].shape)
#         inputs = {k:v.to(self.device) for k, v in inputs.items()}
        out = self(inputs)
        y_hat = out['logits']
        loss = out['loss']
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss


if __name__ == "__main__":
    train_ds = SNV3Dataset(split='train',
                           vocab_path='vocab_files/train_vocab.pyi',
                           batch_size=batch_size)
    valid_ds = SNV3Dataset(split='valid',
                           vocab_path='vocab_files/valid_vocab.pyi',
                           batch_size=batch_size)
    # test_ds = SNV3Dataset(split='test',
    #                        vocab_path='vocab_files/test_vocab.pyi')

    print('Making DataLoader')
    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn, num_workers=8)
    valid_dl = DataLoader(valid_ds, batch_size=1, collate_fn=collate_fn, num_workers=8)
    # test_dl = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)

    # optimizer = AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    print('Making Trainer')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', 
                                          dirpath='checkpoints/', 
                                          filename='{epoch:02d}-{val_loss:2f}',
                                          save_top_k=2)
    
    trainer = pl.Trainer(
        # strategy=None,
        accelerator='gpu',
        devices=1,
        benchmark=True,
        deterministic=False,
        num_sanity_val_steps=0,
        max_epochs=500,
        log_every_n_steps=1,
        callbacks=checkpoint_callback
    )
    print('Creating Model')
    print(train_ds.vocab_size)

    # model = plModel.load_from_checkpoint("/lightning_logs/version_0/checkpoints/epoch=24-step=6725.ckpt")
    model = plModel(vocab_size=train_ds.vocab_size)
#     print(model.num_parameters())
    model.to(device)

    print('Training Model')
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
#     trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl, ckpt_path="lightning_logs/version_0/checkpoints/epoch=25-step=6994.ckpt")
