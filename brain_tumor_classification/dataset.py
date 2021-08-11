import numpy as np
import torch

from brain_tumor_classification.utils import load_dicom_images_3d


class BrainTumorClassificationDataset(torch.utils.Dataset):
    def __init__(
        self,
        paths,
        targets=None,
        mri_type=None,
        label_smoothing=0.01,
        split="train",
        augment=False,
    ):
        self.paths = paths
        self.targets = targets
        self.mri_type = mri_type
        self.label_smoothing = label_smoothing
        self.split = split
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        scan_id = self.paths[index]
        if self.targets is None:
            data = load_dicom_images_3d(
                str(scan_id).zfill(5), mri_type=self.mri_type[index], split=self.split
            )
        else:
            rotation = np.random.randint(0, 4) if self.augment else 0
            data = load_dicom_images_3d(
                str(scan_id).zfill(5),
                mri_type=self.mri_type[index],
                split="train",
                rotate=rotation,
            )

        if self.targets is None:
            return {"X": torch.tensor(data).float(), "id": scan_id}

        return {
            "X": torch.tensor(data).float(),
            "y": torch.tensor(
                abs(self.targets[index] - self.label_smoothing), dtype=torch.float
            ),
        }
