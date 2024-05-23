import argparse
import os
import time
import socket
# import logging
import random
import numpy as np

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist
torch.cuda.empty_cache()

import models
from models.losses import CrossEntropyLossSoft, KLDivLossSoft, Custom_CrossEntropy_PSKD
from datasets.data import get_dataset, get_transform, gen_paths
from optimizer import get_optimizer_config, get_lr_scheduler
from utils import setup_logging, setup_gpus, get_world_size, get_rank, save_checkpoint
from utils import reduce_tensor, to_python_float, log_train_results, log_evaluate_results
from utils import AverageMeter, accuracy, RASampler

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    #Save model log
    parser.add_argument('--results_dir', type=str, default='code/Super_net/results', help='result dir')
    parser.add_argument('--save_name', type=str, default='resnet18q_8642mixbits', help='result dir')
    parser.add_argument('--multi_log', action='store_true', help='print multi_log acording to the number of gpus') 
    parser.add_argument('--evaluate', default=False, action='store_true', help='only evaluate model')

    # Model architecture
    parser.add_argument('--model', default='resnet18q', choices=['resnet20q','resnet18q','resnet50q','mobilenetv2q',], help='model architecture')
    parser.add_argument('--sync_bn', default=False, help='enabling DDP sync BN')

    # data processing
    parser.add_argument('--dataset', default='imagnet', choices=['cifar10', 'cifar100', 'imagnet'], help='dataset name or folder')
    parser.add_argument('--repeated_aug', action='store_true')

    # Training parameter settings
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=90, type=int, help='number of epochs')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--print-freq', '-p', default=100, type=int, help='print frequency')
    parser.add_argument('--pretrain', default=True, action='store_true', help='path to pretrained full-precision checkpoint')
    parser.add_argument('--resume', default='', help='path to latest checkpoint')
    parser.add_argument('--seed', default=0, choices=[0, 42, 3407], type=int, help='seed for initializing training')

    # Optimizer parameter settings
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'], help='optimizer function used')
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', choices=['multi_step', 'MultiStepLR', 'StepLR', 
                                                                                'CosineAnnealingLR', 'ReduceLROnPlateau'], help='lr scheduler')
    parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument('--warm_up', action='store_true', help="enable warm up learning rate")
    parser.add_argument('--warmup_epoch', default=10, help="epoch number of warm up learning rate")
    parser.add_argument('--lr_min', default=1e-10, type=float, help="warm up's initial learning rate")
    parser.add_argument('--lr_decay', default='20,40,60', help='lr decay steps')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--sgd_momentum', default=0.9, type=float, help='momentum factor')
    parser.add_argument('--sgd_dampening', default=0, type=float, help='dampening for momentum factor')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='coefficients used for computing running averages of gradient')
    parser.add_argument('--adam_beta2', default=0.999, type=float, help='coefficients used for computing running averages of gradient_square')
    parser.add_argument('--double_optimizer', action='store_true', help='two optimizer function used')
    parser.add_argument('--step_inside', action='store_true', help='the position of optimizer.step()')
    parser.add_argument('--scale_no_wd', default=False, action='store_true', help='set weight_decay=0 of updating LSQ scales')

    # Quantification configuration file
    parser.add_argument('--bit_width_list', default='2,4,6,8', help='bit width list') # default='2,4,6,8','2,3,4','8,6,4'
    parser.add_argument('--type_w', default='lsq', choices=['dorefa', 'lsq', 'lsq_plus'], help='weight quant_type')
    parser.add_argument('--type_a', default='lsq', choices=['dorefa', 'lsq', 'lsq_plus'], help='activation quant_type')

    # ALRS
    parser.add_argument('--multi_lr', action='store_true', help='multiple lr according to diff bit_width')
    parser.add_argument('--multi_lr_alpha', default=0.1, type=float, help='multi_lr deacy factor')
    parser.add_argument('--multi_lr_stop_bit', default=0, type=float, help='stop less than certain bit update')

    # Knowledge Distillation
    parser.add_argument('--PSKD', action='store_true', help='PSKD')
    parser.add_argument('--alpha_T',default=0.7, type=float, help='alpha_T')

    # Mixed bits are randomly switched
    parser.add_argument('--mix_warmup', default=10, choices=[0, 10, 20, 30], help='the number of mixed_bits training warm_up epochs')
    parser.add_argument('--sigma', default=1/2, choices=[1/5, 1/4, 1/3, 1/2, 2/3], help='probability threshold of bit_switching')
    parser.add_argument('--Trans_BN', default=True, action='store_true', help='enbale strategy of Transitional Batch-Norm')
    parser.add_argument('--second_stage', default=True, action='store_true', help='which stage enable strategy of Transitional Batch-Norm')
    parser.add_argument('--roulette', default=True, action='store_true', help='enable Roulette bit-switching of HASB stochastic process')
    parser.add_argument('--HMT', default=[0.1655166950175371, 0.031237508303352764, 0.005463641462108445, 0.04536974264515771, 0.008738782020315292, 0.06347734138133033, 0.013690473541380867, 0.04473570840699332, 0.0009587626490328047, 0.015475474771053072, 0.004567343712089554, 0.012828331558950364, 0.007550859143809667, 0.01605478354862758, 0.0022888322848649252, 0.006276593913161566, 0.0035778317630054454, 0.004113129827947844, -0.0004940806252379266, -0.0024140890352018587], help='Hessian matrix trace (HMT) of different layers of FP32 mode')

    # Distributed training DDP distributed
    parser.add_argument('--manu_gpu', default=0, type=int,help='manual specified node rank for distributed training')
    parser.add_argument('--rank', default=0, type=int,help='node rank for distributed training')
    parser.add_argument('--node', default=1, type=int,help='number of nodes for distributed training')
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--dist_backend', type=str, default='nccl', choices=['cpu: gloo', 'gpu: nccl'], help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:8080', type=str,help='url used to set up distributed training')
    # parser.add_argument('--local_rank', type=int, default=0, help='LOCAL_PROCESS_RANK, which will be provided by "torch.distributed.launch" module.')
    parser.add_argument('--multiprocessing_distributed', default=False, action='store_true',
                        help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or multi node data parallel training')

    #for cloud
    parser.add_argument('--data_url', type=str, default='/root/data', help='the training data')
    parser.add_argument('--train_url', type=str, default='', help='the path model saved')

    # args, unkown = parser.parse_args()
    args = parser.parse_args()
    return args


def setup_seed(seed):
	torch.manual_seed(seed) # set the seed for generating random numbers.
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) #set the seed for generating random numbers on all GPUS.
		# torch.backends.cudnn.deterministic = True
		# torch.backends.cudnn.benchmark = False


def main(gpu, args):
    #----------------------------------------------------
    #  Set logger
    #----------------------------------------------------
    if args.rank ==0 or args.multi_log:
        # hostname = socket.gethostname()
        if args.evaluate:
            logdir = f'evaluate-{args.save_name}-{args.dataset}'  # the same time as log + random id
        else:
            logdir = f'{args.save_name}-{args.dataset}-{time.strftime("%Y-%m-%d-%H-%M")}-{random.randint(100, 10000)}' 
        logging = setup_logging(os.path.join(args.results_dir, 'log_{}.txt'.format(logdir)))
        logging.info(f"use logdir: {os.path.join(args.results_dir, 'log_{}.txt'.format(logdir))}")
        for arg, val in args.__dict__.items():
            logging.info(f"{arg}: {val}")

    #----------------------------------------------------
    #  Multiprocessing & Distributed Training 
    #----------------------------------------------------
    args.gpu = gpu   #The index of the GPU used by the current process.
    if args.distributed: #Multi-gpu
        if args.multiprocessing_distributed: # rank needs to be the global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + gpu  
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        print("[!] [Rank {}] Distributed Init Setting Done.".format(args.rank))
    else: #single-gpu
        best_gpu, gpus_id = setup_gpus()
        args.gpu = best_gpu  #The index of the GPU used by the current process.
        if args.manu_gpu != best_gpu:
            args.gpu = int(args.manu_gpu) 
        args.rank = get_rank()
    print("[Info] Use GPU : {} for training".format(args.gpu))

    #---------------------------------------------------
    #  Load Dataset
    #---------------------------------------------------
    data_paths = gen_paths(data_url= args.data_url)

    train_transform = get_transform(args.dataset, 'train')
    train_data = get_dataset(args.dataset, 'train', train_transform, data_paths = data_paths, PSKD=args.PSKD)

    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'val', val_transform, data_paths = data_paths, PSKD=args.PSKD)

    #----------------------------------------------------
    #  Load CNN Classifier#
    #----------------------------------------------------
    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()  # Candidate bit-widths sorting
    model = models.__dict__[args.model](bit_width_list, train_data.num_classes, w_schema=args.type_w, a_schema=args.type_a, Trans_BN=args.Trans_BN)
    def weight_init(net):
        # Recursively obtain all sub-modules of net
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    if 'cifar' in args.dataset:
        weight_init(model) # Use Pytroch to initialize, for cifar.

    #----------------------------------------------------
    #  Load checkpoint or Pre-trained model
    #----------------------------------------------------
    if args.pretrain:
        if args.dataset == 'imagenet':
            if args.model == 'resnet18':
                if args.train_url:
                    checkpoint = torch.load('code/Multi_Precision/pre_models/imagenet/resnet18/resnet18q_epoch_90.pth.tar', map_location=torch.device('cpu'))
                else:
                    checkpoint = torch.load('', map_location=torch.device('cpu'))
            elif args.model == 'resnet18q':
                if len(bit_width_list) == 2:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif len(bit_width_list) == 3:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif len(bit_width_list) == 4:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
            elif args.model == 'resnet50':
                if args.train_url:
                    checkpoint = torch.load('', map_location=torch.device('cpu'))
                else:
                    checkpoint = torch.load('', map_location=torch.device('cpu'))
            elif args.model == 'resnet50q':
                if len(bit_width_list) == 2:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif len(bit_width_list) == 3:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif len(bit_width_list) == 4:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
            elif args.model == 'mobilenetv2q':
                if len(bit_width_list) == 2:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif len(bit_width_list) == 3:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif len(bit_width_list) == 4:
                    if args.train_url:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
                    else:
                        checkpoint = torch.load('',  map_location=torch.device('cpu'))
        elif args.dataset == 'cifar10':
            if args.model == 'resnet20':
                if args.train_url:
                    checkpoint = torch.load('', map_location=torch.device('cpu'))
                else:
                    checkpoint = torch.load('', map_location=torch.device('cpu'))
            else:
                if args.train_url:
                    checkpoint = torch.load("code/Super_net/pre_models/cifar10/resnet20q_8642/resnet20q_8642bits_11_model_best.pth.tar", map_location=torch.device('cpu'))
                else:
                    checkpoint = torch.load("/root/code/Super_net/pre_models/cifar10/resnet20q_8642/resnet20q_8642bits_11_model_best.pth.tar", map_location=torch.device('cpu'))

        if len(bit_width_list)>1 and not args.evaluate:
            old_state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k,v in old_state_dict.items():
                # new_state_dict[k[7:]] = v
                if 'bn_dict' in k:
                    if args.Trans_BN:
                        if 'layers.0.bn0' in k:
                            if args.second_stage:
                                new_state_dict[k]=v
                            else:
                                for bits in bit_width_list:
                                    new_state_dict[k.replace('32', str(bits))] = v
                        else:
                            from itertools import product
                            for bits in product(bit_width_list, repeat=2):
                                if args.second_stage:
                                    if 'layers' in k:
                                        new_state_dict[k.replace('bn_dict.'+k[21], 'bn_dict.'+str(bits))] = v
                                    else:
                                        new_state_dict[k.replace('bn_dict.'+k[11], 'bn_dict.'+str(bits))] = v
                                else:
                                    new_state_dict[k.replace('32', str(bits))] = v
                    else:
                        for bits in bit_width_list:
                            new_state_dict[k.replace('32', str(bits))] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            del old_state_dict, new_state_dict
        else:
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        logging.info("loaded pretrain Multi-precicion of '%s'", args.model)
                 

    #----------------------------------------------------
    #  DDP for model
    #----------------------------------------------------
    if args.distributed:
        if args.sync_bn: #Start DDP BN synchronization
            if args.rank ==0 or args.multi_log:
                logging.info("[!] [Rank {}] using synced BN!".format(args.rank))
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        if args.rank ==0 or args.multi_log:
            logging.info("[!] [Rank {}] Distributed DataParallel Setting Start".format(args.rank))
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        if args.rank ==0 or args.multi_log:
            logging.info("[Info] [Rank {} GPU {}] Workers: {}, Batch_size: {}".format(args.rank, args.gpu, args.workers, args.batch_size))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        if args.rank ==0 or args.multi_log:
            logging.info("[!] [Rank {}] Distributed DataParallel Setting End".format(args.rank))
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()


    #----------------------------------------------------
    #  DDP for dataset 
    #----------------------------------------------------
    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        if args.seed > 0:
            setup_seed(args.seed + global_rank)
        if args.repeated_aug:
            sampler_train = RASampler(train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.DistributedSampler(train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        multi_gpu_ddp_test = True
        if multi_gpu_ddp_test:
            sampler_val = torch.utils.data.DistributedSampler(val_data, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = None
        if args.rank ==0 or args.multi_log:
            logging.info("[!] [Rank {}] Distributed Sampler Data Loading Done".format(args.rank))
    else:
        sampler_train = None
        sampler_val = None
        logging.info("[!] [Rank {}] Data Loading Done".format(args.rank))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               pin_memory=True,
                                               batch_size=args.batch_size,
                                               sampler=sampler_train,
                                               shuffle=(sampler_train is None),
                                               num_workers=args.workers
                                               )

    val_loader = torch.utils.data.DataLoader(val_data,
                                             pin_memory=True,
                                             batch_size=args.batch_size,
                                             sampler=sampler_val,
                                             shuffle=False,
                                             num_workers=args.workers
                                             )
   
    #---------------------------------------------------
    #  Define loss function (criterion) , optimizer and lr_scheduler
    #----------------------------------------------------
    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.PSKD:
        criterion_CE_pskd = Custom_CrossEntropy_PSKD().cuda(args.gpu)
    else:
        criterion_CE_pskd = None
    if args.double_optimizer:
        optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay, args.sgd_momentum, args.sgd_dampening,
                                            args.adam_beta1, args.adam_beta2, args.scale_no_wd, args.double_optimizer)
    else:
        optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay, args.sgd_momentum, args.sgd_dampening,
                                            args.adam_beta1, args.adam_beta2, args.scale_no_wd)
        
    if args.warm_up: # warmup + lr_scheduler
        args.epochs = args.epochs - args.warmup_epoch
    lr_decay = list(map(int, args.lr_decay.split(',')))
    if args.double_optimizer:
        lr_scheduler1, lr_scheduler2 = get_lr_scheduler(args.lr_scheduler, optimizer, lr_decay, args.epochs, double_optimizer= args.double_optimizer)
    else:
        lr_scheduler = get_lr_scheduler(args.lr_scheduler, optimizer, lr_decay, args.epochs)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    if args.rank == 0 or args.multi_log:
        logging.info("number of model parameters: %d", num_parameters)

    #----------------------------------------------------
    #  Empty matrix for store predictions
    #----------------------------------------------------
    if args.PSKD:
        all_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)
        if args.rank == 0 or args.multi_log:
            logging.info("[Info] all_predictions matrix shape {}".format(all_predictions.shape))
    else:
        all_predictions = None
    
    #----------------------------------------------------
    #  load status & Resume Learning
    #----------------------------------------------------
    best_prec1 = 0
    val_best_bits = None
    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        else:
            if args.distributed:
                # Map model to be loaded to specified single gpu.
                dist.barrier()
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)

        args.start_epoch = checkpoint['epoch'] + 1
        alpha_t = checkpoint['alpha_t']
        all_predictions = checkpoint['prev_predictions'].cpu() if checkpoint['prev_predictions'] is not None and checkpoint['prev_predictions'].numel() > 0 else None
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler = get_lr_scheduler(args.lr_scheduler, optimizer, lr_decay, args.epochs, last_epoch=checkpoint['epoch'])
        if args.rank == 0 or args.multi_log:
            logging.info("[Rank '%s'] loaded resume checkpoint '%s' (epoch %s)", args.rank, args.resume, checkpoint['epoch'])
    
    #----------------------------------------------------
    #  Bit-mixing for Mixed-Precision
    #----------------------------------------------------
    # model.apply(lambda m: setattr(m, 'sigma', args.sigma))
    model.apply(lambda m: setattr(m, 'bit_list', bit_width_list))
    model.apply(lambda m: setattr(m, 'HMT', args.HMT[1:-1])) 
    model.apply(lambda m: setattr(m, 'Avg_HMT', np.mean(args.HMT[1:-1])))   


    def forward(data_loader, model, criterion_CE, epoch, training=True, optimizer=None, alpha_t=None, all_predictions=None, criterion_CE_pskd=None):
        # import pdb; pdb.set_trace();
        #-----------------------------------
        # Bit-mixing for warm up stage
        #----------------------------------- 
        sigma = args.sigma *  (epoch+1) / args.epochs           
        if epoch < args.mix_warmup:
            model.apply(lambda m: setattr(m, 'sigma', 0))      
        else:
            model.apply(lambda m: setattr(m, 'sigma', sigma))  
    
        losses = [AverageMeter() for _ in bit_width_list]
        top1 = [AverageMeter() for _ in bit_width_list]
        top5 = [AverageMeter() for _ in bit_width_list]

        if training:
            # warm up phase
            if args.warm_up and epoch < args.warmup_epoch:
                step_lr = (args.lr - args.lr_min) / (args.warmup_epoch - 1)
                if epoch ==0:
                    lr = args.lr_min
                else:
                    lr = args.lr_min + step_lr * epoch
                logging.info(f"{epoch}_epoch_warm_up_lr:{lr}")
                if args.double_optimizer:
                    optimizer[0].param_groups[0]['lr'] = lr
                    optimizer[1].param_groups[0]['lr'] = lr
                else:
                    optimizer.param_groups[0]['lr'] = lr
            else:
                args.warm_up = False
                if args.rank == 0 or args.multi_log:
                    if args.double_optimizer:
                        logging.info(f"{epoch}_epoch_not_warm_up_lr:{optimizer[0].param_groups[0]['lr']}")
                    else:
                        logging.info(f"{epoch}_epoch_not_warm_up_lr:{optimizer.param_groups[0]['lr']}")
            
            if args.double_optimizer:
                optimizer_lr = optimizer[0].param_groups[0]['lr']  
                # optimizer_lr = optimizer[1].param_groups[0]['lr']
            else:
                optimizer_lr = optimizer.param_groups[0]['lr'] 

        for i, input_target in enumerate(data_loader):
            if args.PSKD:
                input, target, input_indices = input_target
            else:
                input, target = input_target
            if not training:
                with torch.no_grad():
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    for bw, am_l, am_t1, am_t5 in zip(bit_width_list, losses, top1, top5):
                        model.apply(lambda m: setattr(m, 'wbit', bw))
                        model.apply(lambda m: setattr(m, 'abit', bw))
                        model.apply(lambda m: setattr(m, 'p_sigma', 0))
                        output = model(input)
                        loss = criterion_CE(output, target)
                        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                        # Average loss and accuracy across process for logging
                        if args.distributed:
                            reduced_loss = reduce_tensor(loss.data)
                            prec1 = reduce_tensor(prec1)
                            prec5 = reduce_tensor(prec5)
                        else:
                            reduced_loss = loss.data
                        # to_python_float incure a host<->device sync
                        am_l.update(to_python_float(reduced_loss), input.size(0))
                        am_t1.update(to_python_float(prec1), input.size(0))
                        am_t5.update(to_python_float(prec5), input.size(0))
            else:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                if args.double_optimizer:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                if args.multi_lr:
                    optimizer_lr_update = optimizer_lr
                    
                for bw, am_l, am_t1, am_t5 in zip(bit_width_list[:][::-1], losses[:][::-1], top1[:][::-1], top5[:][::-1]):
                    model.apply(lambda m: setattr(m, 'wbit', bw))
                    model.apply(lambda m: setattr(m, 'abit', bw))
                    #-----------------------------------
                    # Self-KD or none
                    #-----------------------------------      
                    if args.PSKD:
                        targets_numpy = target.cpu().detach().numpy()
                        identity_matrix = torch.eye(len(train_loader.dataset.classes)) 
                        targets_one_hot = identity_matrix[targets_numpy]
                        
                        if epoch == 0 and bw == max(bit_width_list):
                            all_predictions[input_indices] = targets_one_hot

                        # create new soft-targets
                        soft_targets = ((1 - alpha_t) * targets_one_hot) + (alpha_t * all_predictions[input_indices])
                        soft_targets = soft_targets.cuda()
                            
                        # student model
                        output = model(input)
                        softmax_output = F.softmax(output, dim=1) 
                        loss = criterion_CE_pskd(output, soft_targets)
                        
                        if args.distributed:
                            gathered_prediction = [torch.ones_like(softmax_output) for _ in range(dist.get_world_size())]
                            dist.all_gather(gathered_prediction, softmax_output)
                            gathered_prediction = torch.cat(gathered_prediction, dim=0)

                            gathered_indices = [torch.ones_like(input_indices.cuda()) for _ in range(dist.get_world_size())]
                            dist.all_gather(gathered_indices, input_indices.cuda())
                            gathered_indices = torch.cat(gathered_indices, dim=0)
                    else:
                        output = model(input)
                        loss = criterion_CE(output, target)

                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

                    # Average loss and accuracy across process for logging
                    if args.distributed:
                            reduced_loss = reduce_tensor(loss.data)
                            prec1 = reduce_tensor(prec1)
                            prec5 = reduce_tensor(prec5)
                    else:
                        reduced_loss = loss.data

                    # to_python_float incure a host<->device sync
                    am_l.update(to_python_float(reduced_loss), input.size(0))
                    am_t1.update(to_python_float(prec1), input.size(0))
                    am_t5.update(to_python_float(prec5), input.size(0))
                    
                    if args.multi_lr:
                        assert args.step_inside, 'multi_lr must be optimizer.step_inside'
                        if args.double_optimizer:
                            optimizer[0].param_groups[0]['lr'] = optimizer_lr_update
                            # optimizer[1].param_groups[0]['lr'] = optimizer_lr_update
                        else:
                            optimizer.param_groups[0]['lr'] = optimizer_lr_update
                        if bw >= args.multi_lr_stop_bit: 
                            optimizer_lr_update = args.multi_lr_alpha * optimizer_lr_update

                    # logging.info(f"{bw}_bit_loss:{loss.data}")
                    loss.backward()
                    
                    if args.step_inside: 
                        if args.double_optimizer:
                            optimizer[0].step()
                            optimizer[1].step()
                            optimizer[0].zero_grad()
                            optimizer[1].zero_grad()
                        else:
                            optimizer.step()
                            optimizer.zero_grad()

                    if args.PSKD and bw == max(bit_width_list):
                        if args.distributed:
                            for jdx in range(len(gathered_prediction)):
                                all_predictions[gathered_indices[jdx]] = gathered_prediction[jdx].detach()
                        else:
                            all_predictions[input_indices] = softmax_output.cpu().detach()

                if args.step_inside:
                    if args.double_optimizer:
                        optimizer[0].step()
                        optimizer[1].step()
                    else:
                        optimizer.step()

                if i % args.print_freq == 0 and (args.rank==0 or args.multi_log):
                    logging.info('Epoch {0}, iter {1}/{2}, Bitwidth: avg_loss {3:.2f}, avg_prec1 {4:.2f}, avg_prec5 {5:.2f}'.format(
                        epoch, i, len(data_loader),
                        np.mean([losses[i].val for i in range(len(losses))]),
                        np.mean([top1[i].val for i in range(len(top1))]),
                        np.mean([top5[i].val for i in range(len(top5))]), 
                        ))

        if training and args.multi_lr:
            if args.double_optimizer:
                optimizer[0].param_groups[0]['lr'] = optimizer_lr
                # optimizer[1].param_groups[0]['lr'] = optimizer_lr
            else:
                optimizer.param_groups[0]['lr'] = optimizer_lr

        return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5], all_predictions
    
    #-----------------------------------
    # Start training and testing
    #-----------------------------------  
    if args.evaluate:
        model.eval()
        val_loss, val_prec1, val_prec5, _ = forward(val_loader, model, criterion_CE, 0, False)
        if args.rank == 0 or args.multi_log:
            log_evaluate_results(bit_width_list, val_loss, val_prec1, val_prec5)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        if args.PSKD:
            alpha_t = args.alpha_T * ((epoch + 1) / args.epochs) #  Alpha_t update
            alpha_t = max(0, alpha_t)
        else:
            alpha_t = -1

        #  Train
        model.train()
        train_loss, _, _, all_predictions= forward(train_loader, model, criterion_CE, epoch, True, optimizer, alpha_t, all_predictions, criterion_CE_pskd)
        if args.distributed:
            dist.barrier()

        #  Validation
        model.eval()
        val_loss, val_prec1, val_prec5, _ = forward(val_loader, model, criterion_CE, epoch, False)
        if args.distributed:
            dist.barrier()

        if not args.warm_up:
            if args.double_optimizer:
                lr_scheduler1.step()
                lr_scheduler2.step()
            else:
                lr_scheduler.step()

        mean_val_prec1 = np.mean(val_prec1)
        if mean_val_prec1 > best_prec1:
            is_best = True
            best_prec1 = max(mean_val_prec1, best_prec1)
            val_best_bits = val_prec1
        else:
            is_best = False
            best_prec1 = max(mean_val_prec1, best_prec1)
        
        if args.rank == 0 or args.multi_log:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': args.model,
                    'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': (optimizer[0].state_dict(), optimizer[1].state_dict()) if args.double_optimizer else optimizer.state_dict(),
                    'alpha_t' : alpha_t,
                    'prev_predictions': all_predictions
                },
                is_best,
                path=args.results_dir + '/ckpt',
                save_name = args.save_name,
                name = args.save_name + "_epoch_" + str(epoch + 1) + "_acc1_" +str(f"{mean_val_prec1:.2f}")
                )

        if args.rank == 0 or args.multi_log:
            log_train_results(bit_width_list, epoch, train_loss, val_loss, val_prec1, val_prec5)

    if args.rank == 0 or args.multi_log:
        logging.info(f"bits:{bit_width_list}, best_prec1:{val_best_bits}, average:{best_prec1:.2f}")
    if args.distributed:
        dist.barrier() #Synchronizes all processes
        dist.destroy_process_group()
        if args.rank == 0 or args.multi_log:
            logging.info("[!] [Rank {}] Distroy Distributed process".format(args.rank))


if __name__ == '__main__':
    args = parse_args()

    if args.seed > 0:
        setup_seed(args.seed)

    if args.train_url:
        args.results_dir = args.train_url

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if torch.cuda.is_available():
        args.ngpus_per_node = torch.cuda.device_count()
        print('The number of gpus in per_node: ', args.ngpus_per_node)
        torch.backends.cudnn.benchmark = True
    else:
        args.ngpus_per_node = 0
        print("No GPUs are found!")

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.node
        mp.spawn(main, 
                 args=(args), 
                 nprocs=args.world_size, 
                 join=True,
                 )
        print("[!] Multi-GPU All multiprocessing_distributed Training Done.")
    else:
        print("[!] Multi-GPU/Single-GPU of Single-node.")
        main(args.manu_gpu, args)