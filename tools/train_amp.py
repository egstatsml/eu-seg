#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys

sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler, WarmupOnlyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from tools.format_state import format_state_params, add_mean_var_to_state

torch.autograd.set_detect_anomaly(True)
## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
  parse = argparse.ArgumentParser()
  parse.add_argument(
      '--config',
      dest='config',
      type=str,
      default='configs/bisenetv2.py',
  )
  parse.add_argument(
      '--finetune-from',
      type=str,
      default=None,
  )
  parse.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
  parse.add_argument('-g',
                     '--gpus',
                     default=1,
                     type=int,
                     help='number of gpus per node')
  parse.add_argument('-nr',
                     '--nr',
                     default=0,
                     type=int,
                     help='ranking within the nodes')
  args = parse.parse_args()
  args.world_size = args.gpus * args.nodes
  return args


args = parse_args()
cfg = set_cfg_from_file(args.config)


def make_one_hot(labels, classes):
  one_hot = torch.FloatTensor(labels.size()[0], classes,
                              labels.size()[2],
                              labels.size()[3]).zero_().to(labels.device)
  target = one_hot.scatter_(1, labels.data, 1)
  return target


def get_weights(target):
  t_np = target.view(-1).data.cpu().numpy()

  classes, counts = np.unique(t_np, return_counts=True)
  cls_w = np.median(counts) / counts
  #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

  weights = np.ones(7)
  weights[classes] = cls_w
  return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):

  def __init__(self, weight=None, ignore_index=255, reduction='mean'):
    super(CrossEntropyLoss2d, self).__init__()
    self.CE = nn.CrossEntropyLoss(weight=weight,
                                  ignore_index=ignore_index,
                                  reduction=reduction)

  def forward(self, output, target):
    loss = self.CE(output, target)
    return loss


def set_model(lb_ignore=255):
  logger = logging.getLogger()
  net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
  if not args.finetune_from is None:
    logger.info(f'load pretrained weights from {args.finetune_from}')
    loaded_state = torch.load(args.finetune_from, map_location='cpu')
    # for key in loaded_state.keys():
    #   print(key)
    loaded_state = format_state_params(loaded_state, cfg.model_type)
    net.load_state_dict(loaded_state, strict=False)
  if cfg.use_sync_bn:
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
  net.cuda()
  net.train()
  criteria_pre = OhemCELoss(0.7, lb_ignore)
  # criteria_pre = CrossEntropyLoss2d(ignore_index=lb_ignore, reduction='mean')
  criteria_aux = [OhemCELoss(0.7, lb_ignore) for _ in range(cfg.num_aux_heads)]
  net.aux_mode = 'eval'
  return net, criteria_pre, criteria_aux


def set_optimizer(model):
  if 'bayes' in cfg.model_type:
    # only training the final layer
    # params_list = model.module.get_final_params()
    params_list = []
    for param in model.parameters():
      if param.requires_grad:
        params_list.append(param)

    print(len(params_list))
  elif hasattr(model, 'get_params'):
    wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params(
    )
    #  wd_val = cfg.weight_decay
    no_wd_val = 0.0
    params_list = [
        {
            'params': wd_params,
            # 'weight_decay': wd_val
        },
        {
            'params': nowd_params,
            'weight_decay': no_wd_val
        },
        {
            'params': lr_mul_wd_params,
            # 'weight_decay': wd_val,
            'lr': cfg.lr_start * 10
        },
        {
            'params': lr_mul_nowd_params,
            'weight_decay': no_wd_val,
            'lr': cfg.lr_start * 10
        },
    ]
  else:
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
      if param.dim() == 1:
        non_wd_params.append(param)
      elif param.dim() == 2 or param.dim() == 4:
        wd_params.append(param)
    params_list = [
        {
            'params': wd_params,
        },
        {
            'params': non_wd_params,
            'weight_decay': 0
        },
    ]

  optim = torch.optim.SGD(
      params_list,
      lr=cfg.lr_start,
      momentum=cfg.momentum,
      weight_decay=cfg.weight_decay,
  )

  return optim


def set_model_dist(net):
  local_rank = int(os.environ['LOCAL_RANK'])
  net = nn.parallel.DistributedDataParallel(net,
                                            device_ids=[
                                                local_rank,
                                            ],
                                            find_unused_parameters=True,
                                            output_device=local_rank)
  return net


def set_meters():
  time_meter = TimeMeter(cfg.max_iter)
  loss_meter = AvgMeter('loss')
  kl_meter = AvgMeter('kl')
  loss_pre_meter = AvgMeter('loss_prem')
  loss_aux_meters = [
      AvgMeter('loss_aux{}'.format(i)) for i in range(cfg.num_aux_heads)
  ]
  return time_meter, loss_meter, loss_pre_meter, kl_meter, loss_aux_meters


def nan_hook(self, inp, output):
  if not isinstance(output, tuple):
    outputs = [output]
  else:
    outputs = output

  for i, out in enumerate(outputs):
    nan_mask = torch.isnan(out)
    inf_mask = torch.isinf(out)
    if nan_mask.any():
      print("In", self.__class__.__name__)
      raise RuntimeError(f"Found NAN in output {i} at indices: ",
                         nan_mask.nonzero(), "where:",
                         out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
    if nan_mask.any():
      print("In", self.__class__.__name__)
      raise RuntimeError(f"Found inf in output {i} at indices: ",
                         inf_mask.nonzero(), "where:",
                         out[inf_mask.nonzero()[:, 0].unique(sorted=True)])

def initialise_var_params(net):
  if 'bisenetv2' in cfg.model_type:
    weight_squared_sum = torch.zeros(
        net.module.head.conv_out.weight.shape).cuda()
    bias_squared_sum = torch.zeros(net.module.head.conv_out.bias.shape).cuda()
  elif 'bisenetv1' in cfg.model_type:
    weight_squared_sum = torch.zeros(
        net.module.conv_out.conv_out.weight.shape).cuda()
    bias_squared_sum = torch.zeros(net.module.conv_out.conv_out.bias.shape).cuda()
  elif 'pidnet' in cfg.model_type:
    weight_squared_sum = torch.zeros(
        net.module.final_layer.conv2.weight.shape).cuda()
    bias_squared_sum = torch.zeros(
        net.module.final_layer.conv2.bias.shape).cuda()
  else:
    # is enet
    weight_squared_sum = torch.zeros(net.module.fullconv.weight.shape).cuda()
    bias_squared_sum = None
  step_count = 0
  return weight_squared_sum, bias_squared_sum, step_count


def update_var_params(net, weight_mean, bias_mean, weight_squared_sum,
                      bias_squared_sum, step_count):
  if 'bisenetv2' in cfg.model_type:
    # now update our sum parameters for the  weight and bias terms
    weight_squared_sum += torch.square(net.module.head.conv_out.weight - weight_mean)
    bias_squared_sum += torch.square(net.module.head.conv_out.bias - bias_mean)
  elif 'bisenetv1' in cfg.model_type:
    # now update our sum parameters for the  weight and bias terms
    weight_squared_sum += torch.square(net.module.conv_out.conv_out.weight - weight_mean)
    bias_squared_sum += torch.square(net.module.conv_out.conv_out.bias - bias_mean)
  elif 'pidnet' in cfg.model_type:
    # weight_sum += net.module.final_layer.conv2.weight
    # bias_sum += net.module.final_layer.conv2.bias
    weight_squared_sum += torch.square(net.module.final_layer.conv2.weight - weight_mean)
    bias_squared_sum += torch.square(net.module.final_layer.conv2.bias - bias_mean)
  else:
    raise NotImplementedError()
  # increment the step_count
  step_count += 1
  return weight_squared_sum, bias_squared_sum, step_count


def set_trainable_params(net):
  if 'bayes' in cfg.model_config:
    # only want the final layer to be trainable
    # first set all of them to be non trainable
    # then activate just the ones we need
    grad_list =  [
          'module.final_layer.conv2.weight',  # pidnet
          'module.final_layer.conv2.bias',
          'module.head.conv_out.weight',  # bisenetv2
          'module.head.conv_out.bias',
          'module.conv_out.conv_out.weight',  # bisenetv1
          'module.conv_out.conv_out.bias'
      ]
    for name, param in net.named_parameters():
      if not any(name == x for x in grad_list):
        print(name)
        param.requires_grad = False
    print('these params didnt require a grad')
    # time.sleep(100)


def get_lr_scheduler(optim):
  """Get lr scheduler depending on training type

    For the bayesian method we only want to use a LR warmup
    and then a constant LR, but for training we want a warmup and
    then a decay.
    """
  if 'bayes' in cfg.model_config:
    lr_schdr = WarmupOnlyLrScheduler(
        optim,
        power=0.9,
        max_iter=cfg.max_iter,
        warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1,
        warmup='exp',
        last_epoch=-1,
    )
  else:
    lr_schdr = WarmupPolyLrScheduler(
        optim,
        power=0.9,
        max_iter=cfg.max_iter,
        warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1,
        warmup='exp',
        last_epoch=-1,
    )
  return optim, lr_schdr


def init_mean(net, cfg):
  if cfg.model_type == 'bisenetv2':
    w_mean = net.module.head.conv_out.weight.data.detach().clone()
    b_mean = net.module.head.conv_out.bias.data.detach().clone()
  elif cfg.model_type == 'bisenetv1':
    w_mean = net.module.conv_out.conv_out.weight.data.detach().clone()
    b_mean = net.module.conv_out.conv_out.bias.data.detach().clone()
  elif cfg.model_type == 'pidnet':
    w_mean = net.module.final_layer.conv2.weight.data.detach().clone()
    b_mean = net.module.final_layer.conv2.bias.data.detach().clone()
  else:
    raise NotImplementedError(
      f'Extracting mean not specified for your model {cfg.model_type}')
  return w_mean, b_mean



def train():
  logger = logging.getLogger()
  ## dataset
  dl = get_data_loader(cfg, mode='train')
  ## model
  net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)
  ## ddp training
  print(net)
  net = net.cuda()
  net = set_model_dist(net)
  set_trainable_params(net)
  ## optimizer
  optim = set_optimizer(net)
  ## mixed precision training
  scaler = amp.GradScaler()
  # print(net)
  ## meters
  (time_meter, loss_meter, kl_meter, loss_pre_meter,
   loss_aux_meters) = set_meters()
  optim, lr_schdr = get_lr_scheduler(optim)
  # initialising the variance tracking parameters
  weight_squared_sum, bias_squared_sum, step_count = initialise_var_params(net)
  # initialise the mean from pretrained network
  w_mean, b_mean = init_mean(net, cfg)

  for param in net.parameters():
    if torch.isnan(param).any():
      print('found nan in param', param)
  for name, param in net.named_parameters():
    print(name)
  # for submodule in net.modules():
  #     submodule.register_forward_hook(nan_hook)
  # submodule.register_full_backward_hook(nan_hook)
  ## train loop
  for it, (im, lb) in enumerate(dl):
    im = im.cuda()
    lb = lb.cuda()
    lb = torch.squeeze(lb, 1)
    optim.zero_grad()
    # train with mixed precision if needed
    with amp.autocast(enabled=cfg.use_fp16, dtype=torch.float16):
      if ('pidnet' in cfg.model_type) or (net.module.aux_mode == 'eval'):
        logits = net(im)
        loss_pre = criteria_pre(logits, lb)
        # loss aux is set just as a dummy varible
        # it isn't actually tracking anything important
        # or tracking anything really
        loss_aux = None#[crit(0, 0) for crit in criteria_aux]
        loss = loss_pre
      elif 'enet' == cfg.model_type:
        logits = net(im)
        loss_pre = criteria_pre(logits, lb)
        scaler.scale(loss_pre).backward()
        loss = loss_pre
        # loss aux is set just as a dummy varible
        # it isn't actually tracking anything important
        # or tracking anything really
        loss_aux = [crit(0, 0) for crit in criteria_aux]
      # is normal BiSeNet
      else:
        logits, *logits_aux = net(im)
        loss_pre = criteria_pre(logits, lb)
        loss_aux = [
            crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)
        ]
        loss = loss_pre + sum(loss_aux)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    torch.cuda.synchronize()
    if (it >= cfg.warmup_iters) and ('bayes' in cfg.model_config) and (
        (it + 1) % cfg.var_step == 0):
      # now update our sum parameters for the  weight and bias terms
      weight_squared_sum, bias_squared_sum, step_count = update_var_params(
        net, w_mean, b_mean,
        weight_squared_sum, bias_squared_sum,
        step_count)

    time_meter.update()
    loss_meter.update(loss.item())
    loss_pre_meter.update(loss_pre.item())
    # If we are normally training the BiSeNet models, will want to log
    # the auxilliary losses, but if is just fine tuning for the bayesin model than
    # we don't.
    if loss_aux is not None:#('bisenet' in cfg.model_type):  # and not ('bayes' in cfg.model_type):
      _ = [
          mter.update(lss.item())
          for mter, lss in zip(loss_aux_meters, loss_aux)
      ]
      kl_meter = None
    else:
      kl_meter = None
      loss_aux_meters = None

    ## print training log message
    if (it + 1) % 20 == 0:
      lr = lr_schdr.get_lr()
      lr = sum(lr) / len(lr)
      print_log_msg(it, cfg.max_iter, lr, time_meter, loss_meter, kl_meter,
                    loss_pre_meter, loss_aux_meters)
    lr_schdr.step()

  if 'bayes' in cfg.model_config:
    # divide the sum terms by the number of iterations to get the value for the SWAG params
    weight_var =  weight_squared_sum / (step_count - 1)
    bias_var =  bias_squared_sum / (step_count - 1)

    torch.save(weight_var,
               osp.join(cfg.respth, f'{cfg.model_type}_weight_var.pt'))
    torch.save(bias_var, osp.join(cfg.respth, f'{cfg.model_type}_bias_var.pt'))
    print('weight_var = ', weight_var)
    print('bias_var = ', bias_var)
    bayes_state = add_mean_var_to_state(cfg.model_type, net.module.state_dict(),
                                        w_mean, b_mean, weight_var,
                                        bias_var)
    # now save the bayes state
    bayes_save_pth = osp.join(cfg.respth,
                              f'{cfg.model_type}_{cfg.dataset}_bayes_model_final.pth')
    if dist.get_rank() == 0:
      torch.save(bayes_state, bayes_save_pth)

  ## dump the final model and evaluate the result
  # save_pth = osp.join(cfg.respth, f'{cfg.model_type}_{cfg.dataset}_model_final.pth')
  # logger.info('\nsave models to {}'.format(save_pth))
  # state = net.module.state_dict()
  # if dist.get_rank() == 0:
  #   torch.save(state, save_pth)

  # logger.info('\nevaluating the final model')
  # torch.cuda.empty_cache()
  # iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
  # logger.info('\neval results of f1 score metric:')
  # logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
  # logger.info('\neval results of miou metric:')
  # logger.info('\n' +
  #             tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))

  # return


def main():
  # os.environ['MASTER_ADDR'] = '192.168.1.3'
  # os.environ['MASTER_PORT'] = '8888'
  # os.environ['LOCAL_RANK'] = 1
  local_rank = int(os.environ['LOCAL_RANK'])
  print(f'local rank {local_rank}')
  torch.cuda.set_device(local_rank)
  dist.init_process_group(backend='nccl')
  # dist.init_process_group(
  #     backend='nccl',
  #     init_method='env://',
  #     world_size=args.world_size,
  #     rank=rank
  # )
  if not osp.exists(cfg.respth):
    os.makedirs(cfg.respth)
  setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
  train()


if __name__ == "__main__":
  main()
