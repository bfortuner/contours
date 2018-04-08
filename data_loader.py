import numpy as np
import torch.utils.data


class SliceDataset(torch.utils.data.Dataset):
    """PyTorch class for loading images and masks by index and applying transforms.

    The DataLoader calls this class with randomly selected indices.

    Args:
        slices: List of dictionaries
                { 'patient_id': patient id
                  'slice_id': slice id
                  'img_fpath': numpy img
                  'mask_fpath': numpy img
                  'dicom_fpath': path to dicom file}
        img_transform: Torchvision compatible list of data augmentations
        mask_transform: Torchvision compatible list of data augmentations
        joint_transform: Transforms applied equally to both img and mask (e.g. rotation)
    """
    def __init__(self, slices,
                 img_transform=None,
                 mask_transform=None,
                 joint_transform=None):
        self.slices = slices
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform

    def _get_target(self, index):
        fpath = self.slices[index]['mask_fpath']
        mask = np.load(fpath)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return mask

    def _get_input(self, index):
        fpath = self.slices[index]['img_fpath']
        img = np.load(fpath)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img

    def _get_info(self, index):
        patient_id = self.slices[index]['patient_id']
        slice_id = self.slices[index]['slice_id']
        return '{:s}_{:d}'.format(patient_id, slice_id)

    def __getitem__(self, index):
        img = self._get_input(index)
        mask = self._get_target(index)
        info = self._get_info(index)
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        return img, mask, info

    def __len__(self):
        return len(self.slices)


def get_dataloader(dataset, batch_size, shuffle=False, n_workers=2):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=n_workers)
