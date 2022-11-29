import os
import torch
import wandb
import evaluate as huggingface_evaluate

from argparse import ArgumentParser, BooleanOptionalAction
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MultiWOZDataset, MultiWOZBatchGenerator


class NotAScaler:
    """Scaler if fp16 is NOT used.
    """
    def scale(self, loss):
        """No scaling needed.
        @return loss    loss output of a model
        """
        return loss

    def step(self, optimizer):
        """Update parameter.
        @param
        """
        optimizer.step()

    def update(self):
        """No update needed.
        """
        pass


def set_args():
    """Parse arguments from command line.
    @return args    argument of the script
    """
    parser = ArgumentParser()

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4
    )
    parser.add_argument(
        '--num_accum',
        type=int,
        default=1
    )
    parser.add_argument(
        '--fp16',
        type=bool,
        default=False,
        action=BooleanOptionalAction
    )
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=5
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='bleu'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam'
    )
    parser.add_argument(
        '--evaluate',
        type=bool,
        default=True,
        action=BooleanOptionalAction
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='ckpt/'
    )
    parser.add_argument(
        '--ckpt_keep',
        type=int,
        default=1
    )
    parser.add_argument(
        '--wandb',
        type=bool,
        default=True,
        action=BooleanOptionalAction
    )
    parser.add_argument(
        '--wandb_exp_name',
        type=str,
        default='tmp-experiment',
    )
    parser.add_argument(
        '--log_freq',
        type=int,
        default=10,
    )
    args = parser.parse_args()

    return args



def evaluate(model, tokenizer, loader, grader, progbar):
    """Evaluate model.
    @param  model       model to evaluate
    @param  tokenizer   tokenizer of the model
    @param  loader      dev or test dataloader
    @param  grader      computes metric score e.g. bleu, rouge

    @return avg_score   average metric score
    """
    model.eval()
    progbar.set_description("Evaluating...")

    accum_score = 0
    # dev loader
    if args.wandb:
        sample_table = wandb.Table(columns=['act', 'pred_utt', 'gt_utt'])

    for batch in loader:
        src_ids = batch['src_ids'].to(args.device)
        src = batch['src']
        tgt = batch['tgt']

        with torch.no_grad():
            pred_ids = model.generate(
                    src_ids,
                    max_new_tokens=256,
                    num_beams=5, 
                    no_repeat_ngram_size=2, 
                    early_stopping=True
                )
        
        pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # TODO: expand metrics e.g. Rouge, WER
        accum_score += grader.compute(predictions=pred, references=tgt)['bleu'] * len(pred)
    
        if args.wandb:
            for s, p, t in zip(src, pred, tgt):
                sample_table.add_data(s, p, t)

    if args.wandb:
        wandb.log({f'samples_{str(progbar.n)}': sample_table})

    avg_score = accum_score / len(loader)
    return avg_score


def train(
        model, optimizer, loader, scaler,
        eval_tokenizer=None, eval_loader=None, eval_grader=None
    ):
    """Train model.
    @param  model       model to train
    @param  optimizer   optimize strategy e.g. AdamW
    @param  loader      train dataloader
    @param  scaler      scaler for updating gradient
    @param  eval_xxx    parameters for evaluation

    """
    model.train()

    best_score = float('-inf')
    progbar = tqdm(range(args.num_epoch * len(loader)))

    for epoch in range(args.num_epoch):
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

            # log loss
            if args.wandb:
                if progbar.n % args.log_freq == 0:
                    wandb.log({'loss': loss_item})

            # evaluate model
            if args.evaluate:
                if progbar.n % args.evaluate_every == 0:

                    score = evaluate(model, eval_tokenizer, eval_loader, eval_grader, progbar)
                    model.train()

                    if args.wandb:
                        wandb.log({f'{args.metric}': score})

                    # save best checkpoint
                    if score > best_score:
                        best_ckpt_path = os.path.join(args.ckpt_dir, 'best_model')
                        model.save_pretrained(best_ckpt_path)
                        best_score = score


def main():
    # init model and tokenizer from t5-base
    model = T5ForConditionalGeneration.from_pretrained('t5-base').to(args.device)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # init optimizer
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # init dataloader
    batch_generator = MultiWOZBatchGenerator(tokenizer)
    train_dataset = MultiWOZDataset('data/train_processed.json')
    train_dataloader = DataLoader(train_dataset, collate_fn=batch_generator, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # init for evaluation
    if args.evaluate:
        # init dataloader
        dev_dataset = MultiWOZDataset('data/dev_processed.json')
        dev_dataloader = DataLoader(dev_dataset, collate_fn=batch_generator, batch_size=args.batch_size, num_workers=args.num_workers)

        # init evaluation metric
        grader = huggingface_evaluate.load(args.metric)
    
    # init scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else NotAScaler()


    # init logger
    if args.wandb:
        import wandb
    
        # parameter validation
        assert 'tmp' not in args.wandb_exp_name, "Set wandb_exp_name"
        # login
        wandb.login()

        # init_wandb
        wandb.init(
            project="data-to-text-baseline", 
            name=args.wandb_exp_name, 
            config=args,
        )


    # start training
    if not args.evaluate:
        train(model, optimizer, train_dataloader, scaler)
    else:
        train(model, optimizer, train_dataloader, scaler, tokenizer, dev_dataloader, grader)


    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    global args
    args = set_args()

    main()
