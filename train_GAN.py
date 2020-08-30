from config import *
import utils
from tqdm import tqdm
import provider


def start():
    trainDataLoader, testDataLoader = utils.get_dataset()
    seg_classes, seg_label_to_cat = utils.get_seg_classes()
    segmentation_network, generator_criterion, generator_start_epoch = utils.get_segmentation_network()
    discriminator, discriminator_criterion, discriminator_start_epoch = utils.get_discriminator_network()

    generator_optimizer = torch.optim.Adam(
        segmentation_network.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=decay_rate
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=decay_rate
    )

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(generator_start_epoch, final_epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(learning_rate * (lr_decay ** (epoch // step_size)), LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in generator_optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        segmentation_network = segmentation_network.apply(lambda x: utils.bn_momentum_adjust(x, momentum))

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, label, target = data
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            discriminator_optimizer.zero_grad()

            ###################################################
            ######### Training The Discriminator ###############
            ###################################################
            # Training with real batch
            gt_features = torch.eye(50)[target].transpose(1, 2).cuda()
            point_with_features = torch.cat([gt_features, points], dim=1)
            discriminator = discriminator.train()
            pred, _ = discriminator(point_with_features)
            discriminator_loss = discriminator_criterion(pred, torch.ones(target.shape[0]).cuda().to(torch.long))
            discriminator_loss.backward()
            # Training with fake batch
            with torch.no_grad():
                pred, _ = segmentation_network(points, utils.to_categorical(label, num_classes))
            point_with_features = torch.cat([pred.transpose(1,2),points],dim=1)
            pred, _ = discriminator(point_with_features)
            discriminator_loss = discriminator_criterion(pred, torch.zeros(target.shape[0]).cuda().to(torch.long))
            discriminator_loss.backward()
            discriminator_optimizer.step()

            ###################################################
            ######### Training The Generator ##################
            ###################################################
            segmentation_network = segmentation_network.train()
            seg_pred, _ = segmentation_network(points, utils.to_categorical(label, num_classes))
            point_with_features = torch.cat([seg_pred.transpose(1,2),points],dim=1)
            with torch.no_grad():
                discriminator_pred, _ = discriminator(point_with_features)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            generator_loss = generator_criterion(seg_pred, target,discriminator_pred)

            generator_loss.backward()
            generator_optimizer.step()

if __name__ == '__main__':
    start()
