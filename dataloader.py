
class MyDataset(Dataset):
    def __init__(self, data_root, image_size=1024, data_aug=True, random_seed=42):
        self.data_root = data_root
        self.gt_path = join(data_root, 'labels')
        self.img_path = join(data_root, 'images')
        self.map_path = join(data_root, 'attribution_map')
        self.gt_path_files = sorted(glob(join(self.gt_path, '*.png'), recursive=True))
        self.gt_path_files = [
            file for file in self.gt_path_files
            if isfile(join(self.img_path, basename(file)))
        ]
        self.image_size = image_size
        self.target_length = image_size
        self.data_aug = data_aug
        self.random_seed = random_seed

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        # image
        img_3c = cv2.imread(join(self.img_path, img_name))  # (H, W, 3)
        H, W, _ = img_3c.shape
        # Resizing
        img_resize = self.resize_longest_side(img_3c)
        img_padded = self.pad_image(img_resize)  # (image_size, image_size, 3)
        # convert the shape to (3, image_size, image_size)
        img_padded_trans = np.transpose(img_padded, (2, 0, 1))  # (3, image_size, image_size)
        # label
        gt = cv2.imread(self.gt_path_files[index], 0).astype(np.uint8)  # (H,W)
        gt = cv2.resize(gt, (img_resize.shape[1], img_resize.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)  # (image_size, image_size)
        gt = self.pad_image(gt)
        non_zero_values = gt[gt != 0]
        max_label_value = int(non_zero_values.max())
        gt2D = np.uint8(gt == max_label_value)  # (image_size, image_size)
        gt2D = np.uint8(gt2D > 0)
      
        map = cv2.imread(join(self.map_path, img_name), 0).astype(np.uint8)
        map = cv2.resize(vmap, (img_resize.shape[1], img_resize.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        map = self.pad_image(vmap)

        # add data augmentation: random fliplr and random flipud
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if self.data_aug:
            if random.random() > 0.5:
                img_padded_trans = np.ascontiguousarray(np.flip(img_padded_trans, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                map = np.ascontiguousarray(np.flip(map, axis=-1))
            if random.random() > 0.5:
                img_padded_trans = np.ascontiguousarray(np.flip(img_padded_trans, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                map = np.ascontiguousarray(np.flip(map, axis=-2))

        return {
            "image": torch.tensor(img_padded_trans),  # [3,image_size,image_size]
            "gt2D": torch.tensor(gt2D[None, :, :]),  # [1,image_size,image_size]
            "attribution_map": torch.tensor(map[None, :, :]),  # [1,image_size,image_size]
            "image_name": img_name,
            "orig_h": np.array(H),
            "orig_w": np.array(W),
        }

    def resize_longest_side(self, image):

        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (newh, neww)
        return np.array(resize(to_pil_image(image), target_size))

    def resize_longest_side_clip(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 or float32 format.
        """
        long_side_length = self.target_length

        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)  # cv2.resize 使用 (width, height)
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return resized

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3:  ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else:  ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))
        return image_padded
