from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.utils.data.sampler import Sampler
import numpy as np


class GeneExpressionDataset(Dataset):
	def __init__(self, data, genes, train = False):
		self.data = data
		self.genes = genes
		self.X = self.data[self.genes].values.astype('float32')
		self.y = self.data.label.values.astype('float32')
	def __getitem__(self, index):
		sample = self.X[index], self.y[index]
		return sample
	def __len__(self):
		return len(self.data)
        
		
class plTrainerDataset(Dataset):
	def __init__(self, data, genes, train = False):
		self.data = data
		self.genes = genes
		self.X = self.data[self.genes].values.astype('float32')
		self.y = self.data.label.values.astype('int32')
	def __getitem__(self, index):
		sample = self.X[index], self.y[index]
		return sample
	def __len__(self):
		return len(self.data)
		

class CustomSampler(Sampler):
	def __init__(self, df):
		self.df = df
		self.n = len(df)
	def __iter__(self):
		l1_idxs = np.where(self.df.label == 1)[0]
		l0_idxs = np.where(self.df.label == 0)[0]
		if len(l1_idxs) > len(l0_idxs):
			n_diff = len(l1_idxs) - len(l0_idxs)
			l1 = np.random.choice(l1_idxs, len(l1_idxs), replace = False)
			l0 = np.append(np.random.choice(l0_idxs, len(l0_idxs), replace = False), np.random.choice(l0_idxs, n_diff, replace = False))
			idxs = [sub[item] for item in range(len(l0)) for sub in [l1, l0]]
		else:
			n_diff = len(l0_idxs) - len(l1_idxs)
			l0 = np.random.choice(l0_idxs, len(l0_idxs), replace = False)
			l1 = np.append(np.random.choice(l1_idxs, len(l1_idxs), replace = False), np.random.choice(l1_idxs, n_diff, replace = False))
			idxs = [sub[item] for item in range(len(l1)) for sub in [l0, l1]]
		return iter(idxs)
	def __len__(self):
		return self.n 


class DataModule(pl.LightningDataModule):
	def __init__(self, train, val, test, genes):
		super(DataModule, self).__init__()
		self.genes = genes
		self.train = train
		self.sampler = CustomSampler(self.train)
		self.train_dataset = plTrainerDataset(data = self.train, genes = self.genes, train=True)
		self.val = val
		self.val_dataset = plTrainerDataset(data = self.val, genes = self.genes, train=True)
		self.test = test
		self.test_dataset = plTrainerDataset(data = self.test, genes = self.genes, train = False)
	def setup(self, stage):
		train = self.train
		val = self.val
	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=2, shuffle = False, sampler = self.sampler, num_workers = 16)
	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=2, shuffle = False, num_workers = 16)
	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=len(self.test), shuffle = False, num_workers = 16)
