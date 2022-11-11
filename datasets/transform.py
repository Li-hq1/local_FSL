import PIL
import torch 

from torchvision import transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class PretrainTransform(object):
    def __init__(self, args):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ])

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


def build_finetune_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# class PretrainTransform(object):
#     def __init__(self, args):
#         mean = IMAGENET_DEFAULT_MEAN
#         std = IMAGENET_DEFAULT_STD

#         self.transform = transforms.Compose([
#             transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
#             ])

#         self.masked_position_generator = RandomMaskingGenerator(
#             args.window_size, args.mask_ratio
#         )

#     def __call__(self, image):
#         return self.transform(image), self.masked_position_generator()

#     def __repr__(self):
#         repr = "(DataAugmentationForBEiT,\n"
#         repr += "  transform = %s,\n" % str(self.transform)
#         repr += "  Masked position generator = %s,\n" % str(
#             self.masked_position_generator
#         )
#         repr += ")"
#         return repr