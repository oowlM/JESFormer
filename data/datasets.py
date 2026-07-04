from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.data_utils import dual_paired_paths_from_folder
from data.transforms import dual_paired_random_crop, dual_paired_resize, random_augmentation
from utils import FileClient, dual_padding, img2tensor, imfrombytes


class Dataset_DualPairedImage(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.denoise_gt_folder = opt['dataroot_denoise_gt']
        self.segment_gt_folder = opt['dataroot_segment_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.paths = dual_paired_paths_from_folder(
            [self.lq_folder, self.denoise_gt_folder, self.segment_gt_folder],
            ['lq', 'denoise_gt', 'segment_gt'],
            self.filename_tmpl,
        )
        if self.opt['phase'] == 'train':
            self.gt_size = self.opt['gt_size']
            self.scale = self.opt['scale']
            self.geometric_augs = opt['geometric_augs']
            self.transform = {
                'resize': dual_paired_resize,
                'random_crop': dual_paired_random_crop,
            }.get(opt['transform'])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        keys = ['lq', 'denoise_gt', 'segment_gt']
        flags = ['color', 'color', 'binary']
        images, image_paths = [], []
        for key, flag in zip(keys, flags):
            path = self.paths[index][f'{key}_path']
            img_bytes = self.file_client.get(path, key)
            images.append(imfrombytes(img_bytes, flag=flag, float32=True))
            image_paths.append(path)
        img_lq, img_denoise_gt, img_segment_gt = images
        lq_path, denoise_gt_path, segment_gt_path = image_paths

        if self.opt['phase'] == 'train':
            img_denoise_gt, img_segment_gt, img_lq = dual_padding(img_denoise_gt, img_segment_gt, img_lq, self.gt_size)
            img_denoise_gt, img_segment_gt, img_lq = self.transform(
                img_denoise_gt,
                img_segment_gt,
                img_lq,
                self.gt_size,
                self.scale,
                denoise_gt_path,
                segment_gt_path,
            )
            if self.geometric_augs:
                img_denoise_gt, img_segment_gt, img_lq = random_augmentation(img_denoise_gt, img_segment_gt, img_lq)

        img_denoise_gt, img_lq = img2tensor([img_denoise_gt, img_lq], bgr2rgb=True, dtype='float32')
        img_segment_gt = img2tensor(img_segment_gt, bgr2rgb=False, dtype='float32')
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_denoise_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'denoise_gt': img_denoise_gt,
            'segment_gt': img_segment_gt,
            'lq_path': lq_path,
            'denoise_gt_path': denoise_gt_path,
            'segment_gt_path': segment_gt_path,
        }

    def __len__(self):
        return len(self.paths)

