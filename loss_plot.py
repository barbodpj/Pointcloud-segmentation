from torch.utils.tensorboard import SummaryWriter
import re
log_file =  open("trained_models/log.txt","r")
writer = SummaryWriter("my_experiment")

def extract_information(s):
    loss = dict()
    loss['d_real_loss'] = float(re.search(".*tensor\((\d*\.\d*)", s[0])[1])
    loss['d_fake_loss'] = float(re.search(".*tensor\((\d*\.\d*)", s[1])[1])
    loss['generator_adv_loss1'] = float(re.search(".*tensor\((\d*\.\d*)", s[2])[1])
    loss['generator_ce_loss1'] = float(re.search(".*tensor\((\d*\.\d*)", s[3])[1])
    loss['generator_total_loss1'] = float(re.search(".*tensor\((\d*\.\d*)", s[4])[1])
    # loss['generator_adv_loss2'] = float(re.search(".*tensor\((\d*\.\d*)", s[5])[1])
    # loss['generator_ce_loss2'] = float(re.search(".*tensor\((\d*\.\d*)", s[6])[1])
    # loss['generator_total_loss2'] = float(re.search(".*tensor\((\d*\.\d*)", s[7])[1])

    return loss

for epoch in range(12):
    for j in range(875):
        x = []
        for i in range(5):
            x.append(log_file.readline()[:-1])
        log_file.readline()
        if j%20 !=0:
            continue
        loss = extract_information(x)
        for key in loss.keys():
            writer.add_scalar('Loss/' + key, loss[key], j+ 875*epoch)

# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)