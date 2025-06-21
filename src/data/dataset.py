import torch, logging
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import pandas as pd

class FraudDataset(Dataset):
    def __init__(self, config):
        # load data
        X_imputed_path = config.X_imputed_path
        mask_path = config.mask_path
        y_path = config.y_path

        X_imputed = pd.read_csv(X_imputed_path)
        X_imputed_tensor = torch.tensor(X_imputed.values, dtype=torch.float32)
        mask = pd.read_csv(mask_path)
        mask_tensor = torch.tensor(mask.values.astype(float), dtype=torch.float32)  # or int if needed
        self.dim = len(X_imputed_path[0])
        if y_path is not None:
            y = pd.read_csv(y_path)
            y_tensor = torch.tensor(y.values, dtype=torch.float32) 
        else:
            y_tensor = None
        self.X_imputed = X_imputed_tensor
        self.mask = mask_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.X_imputed)

    def __getitem__(self, idx):
        x_imp = self.X_imputed[idx]
        m = self.mask[idx]
        if self.y is not None:
            target = self.y[idx]
            return x_imp, m, target
        else:
            return x_imp, m

class FraudDataloader():
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("FraudDataLoader")
        self.logger.info("Loading DATA...")
        self.dataset = FraudDataset(config)
        self.sampler, self.class_weights = makeSampler(config)
        self.train_loader = DataLoader(self.dataset, self.config.batch_size, sampler=self.sampler)


def makeSampler(config):
    y_path = config.y_path
    y = pd.read_csv(y_path)
    y = y.iloc[:,0]

    counts = y.value_counts()
    zero_count = counts[0]
    one_count = counts[1]

    class_weights = {0: float(1/zero_count), 1: float(1/one_count)}
    sample_weights = y.map(class_weights).values

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # or more if you want more total samples
        replacement=True
    )

    return sampler, class_weights
