import os
import torch
import evaluate
from argparse import ArgumentParser
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiWOZDataset, MultiWOZBatchGenerator


class NotAScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def set_args():
    parser = ArgumentParser()
    args = parser.parse_args()


    # set args as an easydict for now
    from easydict import EasyDict as edict
    args = edict({
        'lr': 0.001,
        'batch_size': 4,
        'num_accum': 1,
        'num_epoch': 5,
        'metric': 'bleu',    # only supports bleu for now
        'device': 'cpu',
        'optimizer': 'Adam',
        'fp16': False,
        'ckpt_dir': 'ckpt/',
        'ckpt_keep': 1
    })

    return args



def train_epoch(
        model, 
        optimizer, 
        loader, 
        scaler,
        progbar, 
        logger,
    ):
    
    model.train()

    for batch_idx, batch in enumerate(loader):
        src_ids = batch['src_ids'].to(args.device)
        src_mask = batch['src_mask'].to(args.device)
        tgt_ids = batch['tgt_ids'].to(args.device)

        output = model(input_ids=src_ids, attention_mask=src_mask, labels=tgt_ids)

        loss = output.loss
        scaler.scale(loss).backward()

        # add gradient accumulation / fp16 if needed
        if ((batch_idx + 1) % args.num_accum == 0) or (batch_idx + 1 == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # update progress
        loss_item = loss.detach().cpu().item()
        progbar.set_description(f"Loss: {round(loss_item, 3)}")
        progbar.update()

    # log information
    

def evaluate_epoch(model, tokenizer, loader, grader, logger):
    model.eval()

    accum_score = 0
    # dev loader
    for batch in loader:
        src_ids = batch['src_ids'].to(args.device)
        tgt = batch['tgt']

        with torch.no_grad():
            pred_ids = model.generate(src_ids).detach()
        
        pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # TODO: expand metrics e.g. Rouge, WER
        accum_score += grader.compute(predictions=pred, references=tgt)['bleu'] * len(pred)
    
    avg_score = accum_score / len(loader)
    return avg_score



def main():
    # init model and tokenizer from t5-base
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # init optimizer
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # init dataloader
    batch_generator = MultiWOZBatchGenerator(tokenizer)
    train_dataset = MultiWOZDataset('data/train_processed.json')
    train_dataloader = DataLoader(train_dataset, collate_fn=batch_generator, batch_size=args.batch_size)
    dev_dataset = MultiWOZDataset('data/dev_processed.json')
    dev_dataloader = DataLoader(dev_dataset, collate_fn=batch_generator, batch_size=args.batch_size)

    # init evaluation metric
    grader = evaluate.load(args.metric)
    
    # init scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else NotAScaler()


    # start training
    best_score = float('-inf')
    progbar = tqdm(range(args.num_epoch * len(train_dataloader)))
    for epoch in range(args.num_epoch):
        train_epoch(model, optimizer, train_dataloader, scaler, progbar, None)
        score = evaluate_epoch(model, tokenizer, dev_dataloader, grader, None)

        ckpt_path = os.path.join(args.ckpt_dir, str(epoch))
        model.save_pretrained(ckpt_path)

        
        # rm_ckpt_path = os.path.join(args.ckpt_dir, str(epoch - args.ckpt_keep))
        # if rm_ckpt_path in os.listdir(args.ckpt_dir):
        #     os.remove(rm_ckpt_path)
        

        if score > best_score:
            best_ckpt_path = os.path.join(args.ckpt_dir, 'best')
            model.save_pretrained(best_ckpt_path)



if __name__ == '__main__':
    global args
    args = set_args()

    main()
