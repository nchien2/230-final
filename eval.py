import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM

from torchmetrics import BLEUScore
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
# from nltk.translate import cider_score
import pytorch_lightning as pl
import config

import evaluate


from dataloader import SNV3Dataset, collate_fn
# from git_trainer import GitLightning


def calculate_meteor_score(true, pred):
    return meteor_score.meteor_score([true], pred)


def calculate_bleu_score(true, pred):
    return sentence_bleu([true], pred)


def calculate_rouge_score(true, pred):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(true, pred)
    rouge_l_score = scores['rougeL'].fmeasure
    return rouge_l_score


def calculate_cider_score(true, pred):

    cider = cider_score.Cider()
    cider_score_val = cider.compute_score([true], pred)
    return cider_score_val[0]


def make_pred(batch, model, device):
    
    pixel_values = batch['clip'].permute(0, 1, 4, 2, 3).float().to(device)
    input_text = batch['caption']
    input_text = [model.train_ds.detokenize(list(input_text[i])) for i in range(len(input_text))]
    

#     sos_inputs = torch.tensor([config.SOS_TOKEN] * len(batch['caption'])).unsqueeze(1).to(device)
#     generated_ids = model.model.generate(pixel_values=pixel_values, 
#                                          max_length=50,
#                                          top_p=0.92,
#                                          do_sample=True,
#                                          top_k=20)
    generated_ids = model.model.generate(pixel_values=pixel_values, 
                                         max_length=50)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)


    return input_text, generated_text

    
class GitLightning(pl.LightningModule):
    def __init__(self, processor, train_ds): # model 1st arg
        super().__init__()
#         self.model = model
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")
        self.processor = processor
        self.train_ds = train_ds
        self.training_step_outputs = []
        self.validation_step_outputs = []
#         self.save_hyperparameters()
        
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
    
    test_ds = SNV3Dataset(split='test',
                  bucket='soccernet-230',
                  vocab_path='vocab_files/train_vocab.pyi',
                  num_clips=4, window_size=3,
                  include_vid=True,
                     local_video_path="/data/videos",
                  no_sample=False)
    
#     test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=batch_size, num_workers=10)
    
    # video captioning
#     processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
#     auto_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")
    
#     model = GitLightning(processor, test_ds)
    
    model = GitLightning.load_from_checkpoint("./checkpoints/epoch=285-val_loss=0.017928.ckpt", 
                                              processor=processor, train_ds=test_ds )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()


    
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    eval_lists = {
        'meteor': [],
        'bleu': [],
        'rouge': []
    }
    
    all_pred_captions = []
    all_true_captions = []
    with torch.no_grad():
        for batch in test_ds:
            input_text, generated_text = make_pred(batch, model, device)
            all_pred_captions += generated_text
            all_true_captions += input_text


                
    # average over all examples:
    eval_lists['meteor'] = meteor.compute(predictions=all_pred_captions, references=all_true_captions)
    eval_lists['bleu'] = bleu.compute(predictions=all_pred_captions, references=all_true_captions)
    eval_lists['rouge'] = rouge.compute(predictions=all_pred_captions, references=all_true_captions)
    print(eval_lists)
