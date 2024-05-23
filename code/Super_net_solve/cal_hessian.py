import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import gc
import time, random
torch.cuda.empty_cache()

from utils import *
from pythessian import hessian
from summary_model import summary
import models
from datasets.data import get_dataset, get_transform, gen_paths
from ILP import cal_ILP
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='calculate the top 1(n) eigenvalue(s) or trace of the neural network')
    # Save model log
    parser.add_argument('--results_dir', type=str, default='code/Super_net_solve/results', help='result dir')
    parser.add_argument('--save_name', type=str, default='resnet18_cal_HMT', help='result dir')
    
    # Model architecture is set with parameters
    parser.add_argument('--model', default='resnet18q', choices=['resnet20q','resnet18q','resnet50q','mobilenetv2q',], help='model architecture')
    parser.add_argument('--pretrain', default=True, help='path to pretrained full-precision checkpoint')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--full_dataset', default=True, action='store_true', help='use full_dataset to compute the hessian information')
    parser.add_argument('--bit_width_list', default='32', help='bit width list')    # default='32','2,4,6,8','2,3,4','8,6,4'
    parser.add_argument('--type_w', default='lsq', choices=['dorefa', 'lsq', 'lsq_plus'], help='weight quant_type')
    parser.add_argument('--type_a', default='lsq', choices=['dorefa', 'lsq', 'lsq_plus'], help='activation quant_type')
    
    # data processing
    parser.add_argument('--dataset', default='imagnet', choices=['cifar10', 'cifar100', 'imagnet'], help='dataset name or folder')
    parser.add_argument('--data_url', type=str, default='./data', help='the training data')
    parser.add_argument('--train_url', type=str, default='', help='the path model saved')

    # Hessian matrix trace configuration
    parser.add_argument('--cal_hessain', default=False, action='store_true', help='enable calculate parames.')
    parser.add_argument('--a_H', default=False, action='store_true', help='enable calculate activation hessian parames.')
    parser.add_argument('--w_H', default=True, action='store_true', help='enable calculate weight hessian parames.')
    parser.add_argument('--average_H', default=True, action='store_true', help='enable calculate average hessian parames.')
    parser.add_argument('--cal_topE', default=False, action='store_true', help='enable calculate hessian top eigenvalue')
    parser.add_argument('--cal_trace', default=True, help='enable calculate hessian trace.')
    parser.add_argument('--num_points', default=1024, help='Number of Data points for calculating hessian')
    parser.add_argument('--num_Hutchsteps', default=7, help='Number of Huchinson Steps.')

    # ILP Configuration
    parser.add_argument('--cal_solve', default=True, action='store_true', help='enable calculate super_net pareto optimal frontier.')
    parser.add_argument('--run_infer', default=True, action='store_true', help='enable the inference of super_net.')
    parser.add_argument('--avg_bits', default=[2,4,6,8], help='Average bit width of the model.') # default=[2,4,6,8], [2,3,4], [4,8]
    parser.add_argument('--candidate_bit_list', default='2,4,6,8', help='Candidate bit of each layer') # default='2,4,6,8','2,3,4','8,6,4'

    # inference for mixed-precision
    parser.add_argument('--sigma', default=0, choices=[0, 1/5, 1/4, 1/3, 1/2, 2/3], help='probability threshold of bit_switching')
    parser.add_argument('--Trans_BN', default=False, help='enable strategy of Transitional Batch-norm.')

    # args, unkown = parser.parse_args()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    assert args.a_H or args.w_H, "enable calculate ......" 

    logdir = f'{args.save_name}-{args.dataset}-{time.strftime("%Y-%m-%d-%H-%M")}-{random.randint(100, 10000)}'
    logging = setup_logging(os.path.join(args.results_dir, 'log_{}.txt'.format(logdir)))
    log_file = os.path.join(os.path.join(args.results_dir, 'log_{}.txt'.format(logdir)))
    for arg, val in args.__dict__.items():
        logging.info(f"{arg}: {val}")

    # set up distributed device
    best_gpu, gpus_id = setup_gpus()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True 
        torch.cuda.set_device(best_gpu)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    
    data_paths = gen_paths(data_url= args.data_url)

    # train_transform = get_transform(args.dataset, 'train')
    # train_data = get_dataset(args.dataset, args.train_split, train_transform, data_paths = data_paths)
    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'val', val_transform, data_paths = data_paths)

    # train_loader = torch.utils.data.DataLoader(train_data,sampler=sampler_train, batch_size=args.batch_size,shuffle=(sampler_train is None),num_workers=args.workers,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data,sampler=None, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    #----------------------------------------------------
    #  First:  Calculate the Hessian eigenvalue or trace
    #----------------------------------------------------
    if args.cal_hessain:
        bit_width_list = list(map(int, args.bit_width_list.split(',')))  # FP32 model ['32']
        bit_width_list.sort()  # Candidate bit-widths sorting
        args.Trans_BN = False
        model = models.__dict__[args.model](bit_width_list, val_data.num_classes, w_schema=args.type_w, a_schema=args.type_a, Trans_BN=args.Trans_BN)
        model.apply(lambda m: setattr(m, 'sigma', 0))        # Bit-switching probability during test phase
        if args.pretrain:
            if args.dataset == 'imagenet':
                if args.model == 'resnet18q':
                    checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif args.model == 'resnet50q':
                    checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif args.model == 'mobilenetv2q':
                    checkpoint = torch.load('',  map_location=torch.device('cpu'))
            elif args.dataset == 'cifar10':
                if args.model == 'resnet20q':
                    checkpoint = torch.load('code/Super_net_solve/pre_model/cifar10/resnet20/model_best_float.pth.tar',  map_location=torch.device('cpu'))
                elif args.model == 'mobilenetv2q':
                    checkpoint = torch.load('',  map_location=torch.device('cpu'))

            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain full-precicion of '%s'", args.model)

        # change the model to eval model to disable running stats update
        model.eval()
        # model.apply(lambda m: setattr(m, 'sigma', 0))
        if args.dataset == 'cifar10':
            summary(model.cuda(), (3,32,32), batch_size=args.batch_size, device="cuda", logging=logging)
        else: # 'imagenet-1K'
            summary(model.cuda(), (3,224,224), batch_size=args.batch_size, device="cuda", logging=logging)

        # Analyze the parameters of the model
        input_param = [] 
        weight_param = []
        with open(log_file) as f:
            for lines_f in f:
                if "Conv2d" in lines_f or "Linear" in lines_f:
                    lines = lines_f.split("]")[-1].split()
                    if float(lines_f.split()[-3].split(']')[0])==1 and float(lines_f.split()[-4].split(',')[0])==1 and args.model != 'mobilenetv2':
                        input_param.append(0)
                        weight_param.append(0)
                    else:
                        input_param.append(float(lines[0]))
                        weight_param.append(float(lines[1]))

        # create loss function
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # we use cuda to make the computation fast
        model = model.cuda()

        assert args.num_points >= args.batch_size, "num_points >= batch_size"
        all_steps = args.num_points / args.batch_size
        input_top_eigenvalue, input_traces = np.array(0), np.array(0)
        weight_top_eigenvalue, weight_traces = np.array(0), np.array(0)

        #  create the hessian computation modul 
        if args.full_dataset:
            hessian_comp = hessian(model, criterion, dataloader=val_loader, cuda=True, logging=logging, a_H=args.a_H, w_H=args.w_H, all_steps=all_steps)
            if args.a_H: # For calculation of activations
                if args.cal_topE:
                    I_eigenvalues, I_eigenvector = hessian_comp.eigenvalues(maxIter=args.num_Hutchsteps, tol=1e-3, top_n=1, whole_model=False, opt_type="actiavte", logging=logging)
                    input_top_eigenvalue = np.array(I_eigenvalues[0])
                    del I_eigenvalues, I_eigenvector
                if args.cal_trace:
                    input_traces = hessian_comp.trace(maxIter=args.num_Hutchsteps, tol=1e-3, whole_model=False, opt_type="activate", logging=logging)
            
            if args.w_H: # For calculation of weights
                if args.cal_topE:
                    W_eigenvalues, W_eigenvector = hessian_comp.eigenvalues(maxIter=args.num_Hutchsteps, tol=1e-3, top_n=1, whole_model=False, opt_type="weight", logging=logging)
                    weight_top_eigenvalue = np.array(W_eigenvalues[0])
                    del W_eigenvalues, W_eigenvector
                if args.cal_trace:
                    weight_traces = hessian_comp.trace(maxIter=args.num_Hutchsteps, tol=1e-3, whole_model=False, opt_type="weight", logging=logging)
            
            model.zero_grad()
            for p in model.parameters():
                p.grad = None 
            gc.collect()
            torch.cuda.empty_cache()
        else:
            i=1
            for inputs, targets in val_loader:
                if i <= all_steps:
                    break
            print(f"--------{i}/{int(all_steps)}---------")
            inputs, targets = inputs.cuda(), targets.cuda()
            hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True, logging=logging, a_H=args.a_H, w_H=args.w_H)

            if args.a_H:
                if args.cal_topE:
                    I_eigenvalues, I_eigenvector = hessian_comp.eigenvalues(maxIter=args.num_Hutchsteps, tol=1e-3, top_n=1, whole_model=False, opt_type="actiavte", logging=logging)
                    input_top_eigenvalue = np.array(I_eigenvalues[0])
                    del I_eigenvalues, I_eigenvector
                if args.cal_trace:
                    input_trace = hessian_comp.trace(maxIter=args.num_Hutchsteps, tol=1e-3, whole_model=False, opt_type="activate", logging=logging)
                    input_traces = input_traces + (input_trace-np.array(input_traces)) / (i+1)
            
            if args.w_H:
                if args.cal_topE:
                    W_eigenvalues, W_eigenvector = hessian_comp.eigenvalues(maxIter=args.num_Hutchsteps, tol=1e-3, top_n=1, whole_model=False, opt_type="weight", logging=logging)
                    weight_top_eigenvalue = np.array(W_eigenvalues[0])
                    del W_eigenvalues, W_eigenvector
                if args.cal_trace:
                    weight_trace = hessian_comp.trace(maxIter=args.num_Hutchsteps, tol=1e-3, whole_model=False, opt_type="weight", logging=logging)
                    weight_traces = weight_traces + (weight_trace-np.array(weight_traces)) / (i+1)
            
            model.zero_grad()
            for p in model.parameters():
                p.grad = None 
            gc.collect()
            torch.cuda.empty_cache()
            i += 1

        if args.model == 'resnet50q' or args.model == 'mobilenetv2':
            plt.figure(num='Hessian', figsize=(16,6), dpi=300)  
        else: # resnet18/20
            plt.figure(num='Hessian', figsize=(8,6), dpi=300) 
        ax = plt.subplot(1,1,1)

        if args.a_H:
            if args.cal_topE:
                logging.info(f"The Hessian top eigenvalue of the {args.model} activation layers is:\n {list(input_top_eigenvalue)}")
                if args.average_H:
                    input_avg_topE = []
                    for i, param in enumerate(input_param):
                        if param != 0:
                            input_avg_topE.append(input_top_eigenvalue[i]/param)
                    logging.info(f"The average Hessian top eigenvalue of the {args.model} activation layers is:\n {list(input_avg_topE)}")
                else:
                    input_avg_topE = input_top_eigenvalue
                plt.xticks(list(range(1, len(input_avg_topE)+1))) # Set the abscissa to be displayed as an integer
                plt.plot(list(range(1, len(input_avg_topE)+1)), input_avg_topE, "r", label='input_top_eigenvalue', linewidth=1, linestyle='--')

            if args.cal_trace:
                logging.info(f"The Hessian trace of the {args.model} activation layers is:\n {list(input_top_eigenvalue)}")
                if args.average_H:
                    input_avg_trace = []
                    for i, param in enumerate(input_param):
                        if param != 0:
                            input_avg_trace.append(input_traces[i]/param)
                    logging.info(f"The average Hessian trace of the {args.model} activation layers is:\n {list(input_avg_trace)}")
                else:
                    input_avg_trace = input_traces
                plt.xticks(list(range(1, len(input_avg_trace)+1)))
                plt.plot(list(range(1, len(input_avg_trace)+1)), input_avg_trace, "r", label='input_traces', linewidth=1)

        if args.w_H:
            if args.cal_topE:
                logging.info(f"The Hessian top eigenvalue of the {args.model} weight layers is:\n {list(weight_top_eigenvalue)}")
                if args.average_H:
                    weight_avg_topE = []
                    for i, param in enumerate(weight_param):
                        if param != 0:
                            weight_avg_topE.append(weight_top_eigenvalue[i]/param)
                    logging.info(f"The average Hessian top eigenvalue of the {args.model} weight layers is:\n {list(weight_avg_topE)}")
                else:
                    weight_avg_topE = weight_top_eigenvalue
                plt.xticks(list(range(1, len(weight_avg_topE)+1)))
                plt.plot(list(range(1, len(weight_avg_topE)+1)), weight_avg_topE, "r", label='weight_top_eigenvalue', linewidth=1, linestyle='--')

            if args.cal_trace:
                logging.info(f"The Hessian trace of the {args.model} weight layers is:\n {list(weight_traces)}")
                if args.average_H:
                    weight_avg_trace = []
                    for i, param in enumerate(weight_param):
                        if param != 0:
                            weight_avg_trace.append(weight_traces[i]/param)
                    logging.info(f"The average Hessian trace of the {args.model} weight layers is:\n {list(weight_avg_trace)}")
                else:
                    weight_avg_trace = weight_traces
                plt.xticks(list(range(1, len(weight_avg_trace)+1)))
                plt.plot(list(range(1, len(weight_avg_trace)+1)), weight_avg_trace, "r", label='weight_traces', linewidth=1)

        plt.xlabel('layers')
        if args.average_H:
            plt.ylabel('Average HMT')
        else:
            plt.ylabel('HMT')


        plt.title(args.save_name)
        plt.legend()
        plt.tight_layout()

        figPath = os.path.join(os.path.join(args.results_dir, 'log_{}.png'.format(logdir)))
        plt.savefig(figPath, dpi=300)
        # plt.show()
        plt.close()

        model.zero_grad()
        for p in model.parameters():
            p.grad = None
        del model
        gc.collect(generation=2)
        torch.cuda.empty_cache()

        logging.info("Finsh cal_hessain!")

    #----------------------------------------------------
    #  Second: Test the performance of the mixed-precision model and draw the Pareto optimal frontier
    #----------------------------------------------------
    if args.cal_solve:
        bit_width_list = list(map(int, args.candidate_bit_list.split(',')))  #Mixed-Precision ['8,6,4,2']
        bit_width_list.sort()  # Candidate bit-widths sorting
        args.Trans_BN = True
        model = models.__dict__[args.model](bit_width_list, val_data.num_classes, args.type_w, args.type_a, args.Trans_BN)
        if args.pretrain: #使用混合精度模型
            if args.dataset == 'imagenet':
                if args.model == 'resnet18q':
                    checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif args.model == 'resnet50q':
                    checkpoint = torch.load('',  map_location=torch.device('cpu'))
                elif args.model == 'mobilenetv2q':
                    checkpoint = torch.load('',  map_location=torch.device('cpu'))
            elif args.dataset == 'cifar10':
                if args.model == 'resnet20q':
                    checkpoint = torch.load('code/Super_net_solve/pre_model/cifar10/resnet20q_8642/resnet20q_8642mixbits_epoch.pth.tar',  map_location=torch.device('cpu'))
                elif args.model == 'mobilenetv2q':
                    checkpoint = torch.load('',  map_location=torch.device('cpu'))

            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain Mixed-precicion of '%s'", args.model)

        model = model.cuda()
        model.eval()
        # model.apply(lambda m: setattr(m, 'logging', logging))
        # if args.model == 'resnet18q':
        #     model.apply(lambda m: setattr(m, 'max_layer_num', 16))
        # elif args.model == 'resnet20q':
        #     model.apply(lambda m: setattr(m, 'max_layer_num', 18))
        # elif args.model == 'mobilenetv2':
        #     model.apply(lambda m: setattr(m, 'max_layer_num', 51))

        model.apply(lambda m: setattr(m, 'mix_infer', 1))    # Mixed precision inference in test phase
        model.apply(lambda m: setattr(m, 'sigma', 0))        # No bit-switching is performed during testing phase    
        # model.apply(lambda m: setattr(m, 'model_name', args.model))

        if args.model == 'resnet18q':
            weight_avg_trace = np.array([])
            input_avg_trace = np.array([])
        elif args.model == 'resnet20q':
            weight_avg_trace = np.array([0.1655166950175371, 0.031237508303352764, 0.005463641462108445, 0.04536974264515771, 0.008738782020315292, 0.06347734138133033, 0.013690473541380867, 0.04473570840699332, 0.0009587626490328047, 0.015475474771053072, 0.004567343712089554, 0.012828331558950364, 0.007550859143809667, 0.01605478354862758, 0.0022888322848649252, 0.006276593913161566, 0.0035778317630054454, 0.004113129827947844, -0.0004940806252379266, -0.0024140890352018587])
            input_avg_trace = np.array([])
        elif args.model == 'mobilenetv2':
            weight_avg_trace = np.array([])
            input_avg_trace = np.array([])
        if weight_avg_trace.all():
            Hutchinson_trace = weight_avg_trace
        elif input_avg_trace:
            Hutchinson_trace = input_avg_trace

        # Hutchinson_trace = Hutchinson_trace - np.min(Hutchinson_trace) 
        args.avg_bits = list(np.arange(4, 8.05, 0.05))   # Expected average bit width
        for avg_i in args.avg_bits:
            logging.info(f'avg_bit={avg_i}:')
            # Calculate the bit allocation combinations of different layers under a given average bit width
            all_candidate_bits = cal_ILP(avg_i, bit_width_list, Hutchinson_trace, logging=logging)
            if args.run_infer and all_candidate_bits:
                for candidate_bits in all_candidate_bits:
                    logging.info(f"Current candidate_bits:{candidate_bits}")
                    top1 = AverageMeter()
                    top5 = AverageMeter()
                    candidate_bits = candidate_bits[1:-1]   # The first and last layers are not quantified
                    model.apply(lambda m: setattr(m, 'cur_bits', candidate_bits))  # Hybrid bit inference in testing phase  
                    # init_mix_batch = 1

                    for input, target in val_loader:
                        with torch.no_grad():
                            input = input.to(device)
                            target = target.cuda(non_blocking=True)

                            model.apply(lambda m: setattr(m, 'wbit', candidate_bits[0]))
                            model.apply(lambda m: setattr(m, 'abit', candidate_bits[0]))
                            # Check the quantization errors of weights and activations of different layers
                            # if init_mix_batch == 1:
                            #     model.apply(lambda m: setattr(m, 'init_mix_batch', 1))
                            #     init_mix_batch = 0
                            # else:
                            #     model.apply(lambda m: setattr(m, 'init_mix_batch', 0))

                            output = model(input)
                            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                            top1.update(to_python_float(prec1), input.size(0))
                            top5.update(to_python_float(prec5), input.size(0))
                    
                    logging.info('val prec1: {:.2f}, val prec5: {:.2f}'.format(top1.avg, top5.avg))

        logging.info("Finsh cal_solv of SuperNets!")



















