from config import *
import utils
from tqdm import tqdm
import provider
import time


def start():
    trainDataLoader, testDataLoader = utils.get_dataset()
    seg_classes, seg_label_to_cat = utils.get_seg_classes()

    generator, generator_ce_criterion, generator_adv_criterion, generator_start_epoch = utils.get_generator_network()
    discriminator, discriminator_criterion, discriminator_start_epoch = utils.get_discriminator_network()

    print("Generator_start_epoch: ", generator_start_epoch)
    print("discriminator_start_epoch: ", discriminator_start_epoch)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=generator_learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=decay_rate
    )

    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=discriminator_learning_rate, momentum=0.9)

    if generator_start_epoch > 0:
        checkpoint = torch.load(generator_checkpoint_path)
        generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if discriminator_start_epoch > 0:
        checkpoint = torch.load(discriminator_checkpoint_path)
        discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    best_acc = 0
    global_epoch = generator_start_epoch
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    r = 50

    g_step = 0

    for epoch in range(global_epoch, final_epoch):
        r += 1
        if r < 50:
            g_step = 0
        else:
            g_step = 1

        print('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, final_epoch))
        '''Adjust learning rate and BN momentum'''
        #lr = max(generator_learning_rate * (lr_decay ** (epoch // step_size)), LEARNING_RATE_CLIP)
        lr = generator_learning_rate
        print('Learning rate:%f' % lr)
        for param_group in generator_optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // step_size))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        generator = generator.apply(lambda x: utils.bn_momentum_adjust(x, momentum))

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, label, target = data
            points = points.data.numpy()

            #points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            #points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            discriminator_optimizer.zero_grad()

            ###################################################
            ######### Training The Discriminator ###############
            ###################################################
            # Training with real batch

            with torch.no_grad():
                pred, _ = generator(points, utils.to_categorical(label, num_classes))
            pred = pred.transpose(1, 2)
            exp_pred = torch.exp(pred)
            gt_pred = torch.gather(exp_pred, 1, target[:, None, :])
            high_value = torch.clamp(tau + torch.normal(0, 1, (1,))[0] * 0.1, min=0.65, max=0.95)

            normalized_gt = torch.max(gt_pred, torch.zeros(points.shape[0], 1, npoint).cuda() + high_value)
            gt_features = (1 - normalized_gt) / ((1 - gt_pred) + 1e-10) * exp_pred
            gt_features[gt_features == 0] = 1e-10
            gt_features.scatter_(1, target[:, None, :], normalized_gt)
            #gt_features = torch.log(gt_features)

            point_with_features = torch.cat([points, gt_features], dim=1)
            discriminator = discriminator.train()
            D_score, _ = discriminator(point_with_features)
            discriminator_real_loss = discriminator_criterion(D_score,
                                                              torch.ones(target.shape[0]).cuda().to(torch.long))

            prefix_str = "epoch " + str(epoch) + " step" + str(i) + " "
            log_file.write(prefix_str + "discriminator real loss: " + str(discriminator_real_loss) + "\n")

            # Training with fake batch
            point_with_features = torch.cat([points, torch.exp(pred)], dim=1)
            D_score, _ = discriminator(point_with_features)

            discriminator_fake_loss = discriminator_criterion(D_score,
                                                              torch.zeros(target.shape[0]).cuda().to(torch.long))
            log_file.write(prefix_str + "discriminator fake loss: " + str(discriminator_fake_loss) + "\n")
            discriminator_loss = discriminator_real_loss + discriminator_fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()
            ###################################################
            ######### Training The Generator ##################
            ###################################################
            for xx in range(g_step):
                generator_optimizer.zero_grad()
                seg_pred, _ = generator(points, utils.to_categorical(label, num_classes))
                point_with_features = torch.cat([points, torch.exp(seg_pred.transpose(1, 2))], dim=1)
                discriminator_pred, _ = discriminator(point_with_features)
                generator_loss = landa * generator_adv_criterion(discriminator_pred)
                log_file.write(prefix_str + "generator adv loss " + str(generator_loss / landa) + "\n")
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1].detach().cpu()
                correct = pred_choice.eq(target.detach().cpu().data).detach().cpu().sum()
                mean_correct.append(correct.item() / (batch_size * npoint))
                generator_ce_loss = generator_ce_criterion(seg_pred, target)
                log_file.write(prefix_str + "generator ce loss: " + str(generator_ce_loss) + "\n")
                generator_loss = generator_loss + generator_ce_loss
                log_file.write(prefix_str + "total loss: " + str(generator_loss) + "\n")
                generator_loss.backward()
                generator_optimizer.step()
            log_file.write("__________________________________________________________________________\n")
            log_file.flush()
        train_instance_acc = np.mean(mean_correct)
        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                          smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                generator = generator.eval()
                seg_pred, _ = generator(points, utils.to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()
                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        print('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= 0):
            print('Saving Generator model...')
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': generator_optimizer.state_dict(),
            }
            torch.save(state, generator_checkpoint_path)

            print('Saving discriminator model...')
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': discriminator_optimizer.state_dict(),
            }
            torch.save(state, discriminator_checkpoint_path)

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        print('Best accuracy is: %.5f' % best_acc)
        print('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        print('Best instance avg mIOU is: %.5f' % best_inctance_avg_iou)


if __name__ == '__main__':
    start()