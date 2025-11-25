from ...core import register
from torch.utils.data import ConcatDataset
from .coco_dataset import CocoDetection

@register()
class ConcatCocoDataset(CocoDetection):
    __inject__ = ['transforms']

    def __init__(self, datasets, transforms, return_masks=False, remap_mscoco_category=False):
        self.datasets = datasets
        datasets_list = [CocoDetection(**dataset_cfg, 
                                          transforms=None, 
                                          return_masks=return_masks, 
                                          remap_mscoco_category=remap_mscoco_category) 
                        for dataset_cfg in datasets]
        self.dataset = ConcatDataset(datasets_list)
        self._transforms = transforms
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def load_item(self, idx):
        return self.dataset.__getitem__(idx)

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def __len__(self):
        return len(self.dataset)
    
    def extra_repr(self) -> str:
        s = f' num_datasets: {len(self.datasets)}\n'
        s += f' datasets_cfgs: {self.datasets}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s
    