import torch
from data_utils.ShapeNetDataLoader import PartNormalDataset
from config import *
from models import pointnet2_part_seg_ssg as pointnet2_seg
from models import pointnet2_cls_ssg as pointnet2_cls

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def get_discriminator_network():
    discriminator = pointnet2_cls.get_model(num_class= 2, normal_channel=normal).cuda()
    criterion = pointnet2_cls.get_loss().cuda()

    try:
        checkpoint = torch.load(discriminator_checkpoint_path)
        start_epoch = checkpoint['epoch']
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        print('Use discriminator pretrain model')
    except:
        print('No existing discriminator model, starting training from scratch...')
        start_epoch = 0
    return discriminator,criterion,start_epoch

def get_generator_network():
    segmentation_network = pointnet2_seg.get_model(num_part, normal_channel=normal).cuda()
    ce_criterion,adv_criterion = pointnet2_seg.get_ce_loss().cuda(),pointnet2_seg.get_adv_loss().cuda()

    try:
        checkpoint = torch.load(generator_checkpoint_path)
        start_epoch = checkpoint['epoch']
        segmentation_network.load_state_dict(checkpoint['model_state_dict'])
        print('Use Generator pretrain model')
    except:
        print('No existing generator model, starting training from scratch...')
        start_epoch = 0
        segmentation_network = segmentation_network.apply(weights_init)
    return segmentation_network,ce_criterion,adv_criterion,start_epoch

def get_seg_classes():
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                   'Mug': [36, 37],
                   'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                   'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15],
                   'Knife': [22, 23]}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    return seg_classes,seg_label_to_cat


def get_dataset():
    TRAIN_DATASET = PartNormalDataset(root=data_path, npoints=npoint, split='trainval',
                                      normal_channel=normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True,
                                                  num_workers=4)
    TEST_DATASET = PartNormalDataset(root=data_path, npoints=npoint, split='test', normal_channel=normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainDataLoader, testDataLoader

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y
