import torch
import os
import copy
import numpy as np
import Data_Generator
import torch.optim as optim
from PtrNet import NeuralCombOptRL
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# parameters
batch_size = 128
train_size = 10000
val_size = 1000
seq_len = 21   # servive_num and depot num
input_dim = 2
embedding_dim = 128
hidden_dim = 128
vehicle_init_capacity = 30
p_dim = 128  # as same to embedding_dim
R = 4
n_process_blocks = 3
n_glimpses = 1
use_tanh = True
C = 10  # tanh exploration
n_epochs = 100
use_cuda = False
random_seed = 111
is_train = True
critic_beta = 0.9
beam_size = 1  # if set B=1 then the technique is same as greedy search
actor_net_lr = 1e-4
critic_net_lr = 1e-4
actor_lr_decay_step = 5000
actor_lr_decay_rate = 0.96
critic_lr_decay_step = 5000
critic_lr_decay_rate = 0.96
disable_tensorboard = False
load_path = ''
output_dir = 'tsp_model/A2C'
log_dir = 'runs/A2C'
data_dir = 'data/tsp/A2C'

torch.manual_seed(random_seed)

if not disable_tensorboard:
    writer = SummaryWriter(os.path.join(log_dir, 'tsp'))

training_dataset = Data_Generator.VRPDataset(node_num=seq_len, num_samples=train_size)
# val_dataset = Data_Generator.VRPDataset(filename='./data/tsp/tsp20_test.txt')   # if specify filename, other arguments not required
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
# validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

# instantiate the Neural Combinatorial Opt with RL module
model = NeuralCombOptRL(embedding_dim,
                        hidden_dim,
                        seq_len,
                        n_glimpses,
                        n_process_blocks,
                        C,
                        use_tanh,
                        beam_size,
                        is_train,
                        use_cuda,
                        vehicle_init_capacity,
                        p_dim,
                        R)

# Load the model parameters from a saved state
if load_path != '':
    print('[*] Loading model from {}'.format(load_path))
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), load_path)))  # load parameters
    model.actor_net.decoder.seq_len = seq_len
    model.is_train = is_train

save_dir = os.path.join(os.getcwd(), output_dir)

try:
    os.makedirs(save_dir)
except:
    pass

critic_mse = torch.nn.MSELoss()
critic_optim = optim.Adam(model.critic_net.parameters(), lr=critic_net_lr)
actor_optim = optim.Adam(model.actor_net.parameters(), lr=actor_net_lr)

actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
                                           list(range(actor_lr_decay_step,
                                                      actor_lr_decay_step * 1000,
                                                      actor_lr_decay_step)),
                                           gamma=actor_lr_decay_rate)

critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
                                            list(range(critic_lr_decay_step,
                                                       critic_lr_decay_step * 1000,
                                                       critic_lr_decay_step)),
                                            gamma=critic_lr_decay_rate)

if use_cuda:
    model = model.cuda()
    critic_mse = critic_mse.cuda()

step = 0
val_step = 0
log_step = 1
epoch = 100


def train_one_epoch(i):
    global step
    # put in train mode!
    model.train()

    # sample_batch is [batch_size x sourceL x input_dim]
    for batch_id, sample_batch in enumerate(tqdm(training_dataloader, disable=False)):
        if use_cuda:
            sample_batch = sample_batch.cuda()

        R, b, probs, actions_idxs = model(copy.deepcopy(sample_batch))
        advantage = R - b  # means L(π|s) - b(s)

        # compute the sum of the log probs for each tour in the batch
        # probs: [2(seq_len-1)+1 x batch_size], logprobs: [batch_size]
        logprobs = sum([torch.log(prob) for prob in probs])
        # clamp any -inf's to 0 to throw away this tour
        logprobs[(logprobs < -1000).detach()] = 0.  # means log p_(\theta)(π|s)

        # multiply each time step by the advanrate
        reinforce = advantage * logprobs
        actor_loss = reinforce.mean()

        # actor net processing
        actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        # clip gradient norms
        torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(),  max_norm=2.0, norm_type=2)
        actor_optim.step()
        actor_scheduler.step()

        # critic net processing
        R = R.detach()
        critic_loss = critic_mse(b, R)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(), max_norm=2.0, norm_type=2)
        critic_optim.step()
        critic_scheduler.step()

        step += 1

        if not disable_tensorboard:
            writer.add_scalar('avg_reward', R.mean().item(), step)
            writer.add_scalar('actor_loss', actor_loss.item(), step)
            writer.add_scalar('critic_loss', critic_loss.item(), step)

        if step % log_step == 0:
            print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(i, batch_id, R.mean().item()))


# def validation():
#     global val_step
#     model.actor_net.decoder.decode_type = 'beam_search'
#     print('\n~Validating~\n')
#
#     example_input = []
#     example_output = []
#     avg_reward = []
#
#     # put in test mode!
#     model.eval()
#
#     for batch_id, val_batch in enumerate(tqdm(validation_dataloader, disable=False)):
#         if use_cuda:
#             val_batch = val_batch.cuda()
#
#         R, v, probs, actions, action_idxs = model(val_batch)
#
#         avg_reward.append(R[0].item())
#         val_step += 1
#
#         if not disable_tensorboard:
#             writer.add_scalar('val_avg_reward', R.item(), int(val_step))
#
#         if val_step % log_step == 0:
#             print('val_avg_reward:', R.item())
#
#             if plot_att:
#                 probs = torch.cat(probs, 0)
#                 plot_attention(example_input, example_output, probs.cpu().numpy())
#     print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
#     print('Validation overall reward var: {}'.format(np.var(avg_reward)))


def train_model():
    for i in range(epoch):
        if is_train:
            train_one_epoch(i)
        # Use beam search decoding for validation
        # validation()

        if is_train:
            model.actor_net.decoder.decode_type = 'stochastic'
            print('Saving model...epoch-{}.pt'.format(i))
            torch.save(model.state_dict(), os.path.join(save_dir, 'epoch-{}.pt'.format(i)))


if __name__ == '__main__':
    train_model()
