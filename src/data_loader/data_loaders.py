from base import BaseDataLoader
from data_loader.shape3d_dataset import Shape3DDataset


class Shape3DLoader(BaseDataLoader):
    """
    data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, task, dataset, PolyPool, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.ToTensor(),
        #     transforms.ToTensor(),
        #     transforms.ToTensor(),
        #     transforms.ToTensor(),
        # ])
        self.data_dir = data_dir

        self.dataset = Shape3DDataset(self.data_dir, task=task, dataset=dataset, PolyPool=PolyPool, transform=None)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
