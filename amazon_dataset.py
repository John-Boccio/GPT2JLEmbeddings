import datasets
import numpy as np
import torch
from sklearn.utils import class_weight


class AmazonBookReviewDataset:
    def __init__(self, tokenizer, batch_size=32, max_length=256, device=None) -> None:
        dataset = datasets.load_dataset('amazon_us_reviews', 'Books_v1_02')['train']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'review_body' and col != 'star_rating'])

        # The smallest star rating has 166,384 examples
        train_count = 50000
        val_count = 10000
        test_count = 10000
        min_count = train_count + val_count + test_count

        # Balance the dataset
        train_reviews = [None] * (train_count * 5)
        val_reviews = [None] * (val_count * 5)
        test_reviews = [None] * (test_count * 5)

        count = np.zeros(5)
        train_idx = 0
        val_idx = 0
        test_idx = 0
        for item in dataset:
            if np.all(count >= min_count):
                break

            review, rating = item['review_body'], item['star_rating']
            if count[rating-1] >= min_count:
                continue

            if count[rating-1] < train_count:
                train_reviews[train_idx] = {
                    'review_body': review,
                    'star_rating': rating - 1
                }
                train_idx += 1
            elif count[rating-1] < train_count + val_count:
                val_reviews[val_idx] = {
                    'review_body': review,
                    'star_rating': rating - 1
                }
                val_idx += 1
            else:
                test_reviews[test_idx] = {
                    'review_body': review,
                    'star_rating': rating - 1
                }
                test_idx += 1
            count[rating-1] += 1

        self.train_dataset = train_reviews
        self.val_dataset = val_reviews
        self.test_dataset = test_reviews

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
    
    def _tokenize(self, examples):
        return self.tokenizer(examples, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)

    def _get_random_batch(self, dataset):
        choices = np.random.choice(len(dataset), self.batch_size, replace=False)
        x = self._tokenize([dataset[int(i)]['review_body'] for i in choices]).to(self.device)
        y = torch.tensor([dataset[int(i)]['star_rating'] for i in choices], device=self.device)
        return x, y

    def get_random_train_batch(self):
        return self._get_random_batch(self.train_dataset)

    def get_random_val_batch(self):
        return self._get_random_batch(self.val_dataset)

    def get_random_test_batch(self):
        return self._get_random_batch(self.test_dataset)
