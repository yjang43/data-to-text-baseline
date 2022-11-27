import json
from torch.utils.data import Dataset



class MultiWOZDataset(Dataset):
    def __init__(self, data_fp, task_prefix='data to text: '):
        """Initialize MultiWOZ dataset
        @param  data_fp     file of act and utterance pairs e.g. data/xxx_precessed.json
        """
        # load dataset
        with open(data_fp, 'r') as f:
            self.data = json.load(f)

        self.task_prefix = task_prefix


    def __len__(self):
        """Get the number of act and utterance pairs
        @return number of act and utterance pairs
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get an act and utterance pair
        @param  idx     index of the pair

        @return item    a pair of prefixed linearized act and utterance
        """
        item = {
            'act': self.task_prefix + self.data[idx]['act'],
            'utt': self.data[idx]['utterance']
        }

        return item



class MultiWOZBatchGenerator:

    def __init__(self, tokenizer):
        """Initialize collate function for MultiWOZ dataset
        @param  tokenizer   tokenizer to tokenize a string
        """
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        """Generate batch
        @param  batch   batch of act and utterance pair in string
        @return batch   batch of src_ids, src_mask, tgt_ids, tgt in tensor
        """
        acts = [item['act'] for item in batch]
        utts = [item['utt'] for item in batch]

        src = self.tokenizer(
            acts,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tgt = self.tokenizer(
            utts,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        # ignore padding from the loss function
        tgt['input_ids'][tgt['input_ids'] == self.tokenizer.pad_token_id] = -100
        
        batch = {
            'src_ids': src['input_ids'],
            'src_mask': src['attention_mask'],
            'tgt_ids': tgt['input_ids'],
            'tgt': utts
        }
        
        return batch



if __name__ == '__main__':
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader
    from pprint import pprint

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = MultiWOZDataset('data/processed.json')
    batch_generator = MultiWOZBatchGenerator(tokenizer)
    dataloader = DataLoader(dataset, collate_fn=batch_generator, batch_size=4)

    iter_dataloader = iter(dataloader)
    pprint(next(iter_dataloader))
