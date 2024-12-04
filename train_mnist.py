import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils as tvutils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import logging
import time
from io import BytesIO

from score import ScoreNet
from networks import UNet, SimpleEncoder, SimpleDecoder 
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'Learning Rate')
flags.DEFINE_float('step_lr', 2e-3, 'Step LR for sampling')
flags.DEFINE_integer('num_epochs', 3, 'Number of Epochs')
flags.DEFINE_integer('seed', 2, 'Random seed')
flags.DEFINE_string('output_dir', 'runs/mnist-fc/', 'Output Directory')
flags.DEFINE_string('model_type', 'simple_fc', 'Network to use')
flags.DEFINE_float('sigma_begin', 2, 'Largest sigma value')
flags.DEFINE_float('sigma_end', 0.1, 'Smallest sigma value')
flags.DEFINE_integer('noise_level', 20, 'Number of noise levels')
flags.DEFINE_integer('log_every', 200, 'Frequency of logging the loss')
flags.DEFINE_integer('sample_every', 200, 'Frequency for saving generated samples')
flags.DEFINE_integer('batch_size', 128, 'Batch Size for Training')
flags.DEFINE_string('sigma_type', 'geometric', 'The type of sigma distribution, geometric or linear')
flags.DEFINE_string('mnist_data_dir', './data', 'Where to download MNIST dataset')

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if not len(logging.getLogger().handlers): 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>8s} [{global_step:07d}]: {value:5f}')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_scorenet(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(FLAGS.seed)

    writer = SummaryWriter(FLAGS.output_dir, max_queue=1000, flush_secs=120)

    if FLAGS.model_type == "unet":
        net = UNet()
    elif FLAGS.model_type == "simple_fc":
        net = torch.nn.Sequential(
          SimpleEncoder(input_size=1024, hidden_size=128, latent_size=16),
          SimpleDecoder(latent_size=16, hidden_size=128, output_size=1024))
    
    scorenet = ScoreNet(net, FLAGS.sigma_begin, FLAGS.sigma_end,
                        FLAGS.noise_level, FLAGS.sigma_type)
    logging.info(f'Number of parameters in ScoreNet: {count_parameters(scorenet)}')
    scorenet.train()
    
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    dataset = datasets.MNIST(FLAGS.mnist_data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scorenet.to(device)
    optimizer = optim.Adam(scorenet.parameters(), lr=FLAGS.lr)
    iterations = 0

    train_loss = []
    for epoch in range(1, FLAGS.num_epochs + 1):
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.reshape(data.shape[0], -1)
            data = data.to(device)
            optimizer.zero_grad()
            loss = scorenet.get_loss(data)
            loss.backward()
            optimizer.step()
            
            train_loss += [loss.item()]
            iterations += 1

            if iterations % FLAGS.log_every == 0:
                writer.add_scalar('loss', np.mean(train_loss), iterations)
                logger('loss', np.mean(train_loss), iterations)
                train_loss = []
            
            if iterations % FLAGS.sample_every == 0:
                scorenet.eval()
                with torch.no_grad():
                    X_gen = scorenet.sample(64, 1024, step_lr=FLAGS.step_lr)[-1, -1].view(-1, 1, 32, 32)
                    
                    samples_image = BytesIO()
                    tvutils.save_image(X_gen, samples_image, 'png')
                    samples_image = Image.open(samples_image)
                    file_name = f'{FLAGS.output_dir}/samples_{iterations:08d}.png'
                    samples_image.save(file_name)
                    writer.add_image('samples', np.transpose(np.array(samples_image), [2,0,1]), iterations)

                    X_gt = data.view(-1,1,32,32)[:64]
                    gt_image = BytesIO()
                    tvutils.save_image(X_gt, gt_image, 'png')
                    gt_image = Image.open(gt_image)
                    writer.add_image('gt', np.transpose(np.array(gt_image), [2,0,1]), iterations)
                scorenet.train()

if __name__ == "__main__":
    app.run(train_scorenet)
