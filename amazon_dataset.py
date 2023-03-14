import datasets
import numpy as np
import torch
from sklearn.utils import class_weight


class AmazonBookReviewDataset:
    def __init__(self, tokenizer, batch_size=32, max_length=256, device=None) -> None:
        self.dataset = datasets.load_dataset('amazon_us_reviews', 'Books_v1_02')['train']
        self.dataset = self.dataset.remove_columns([col for col in self.dataset.column_names if col != 'review_body' and col != 'star_rating'])

        self.class_weight = torch.tensor(class_weight.compute_class_weight('balanced', classes=np.unique(self.dataset['star_rating']), y=self.dataset['star_rating']), device=device).type(torch.float)

        self.num_train_samples = int(0.8 * len(self.dataset))
        self.num_val_samples = int(0.1 * len(self.dataset))
        self.num_test_samples = len(self.dataset) - self.num_train_samples - self.num_val_samples

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
    
    def _tokenize(self, examples):
        return self.tokenizer(examples, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)

    def _get_random_batch(self, offset, length):
        choices = np.random.choice(length, self.batch_size, replace=False) + offset
        x = self._tokenize([self.dataset[int(i)]['review_body'] for i in choices]).to(self.device)
        y = torch.tensor([self.dataset[int(i)]['star_rating']-1 for i in choices], device=self.device)
        return x, y

    def get_random_train_batch(self):
        return self._get_random_batch(0, self.num_train_samples)

    def get_random_val_batch(self):
        return self._get_random_batch(self.num_train_samples, self.num_val_samples)

    def get_random_test_batch(self):
        return self._get_random_batch(self.num_train_samples + self.num_val_samples, self.num_test_samples)
