import os
import torch
import torch.distributed as dist
import logging
import shutil
import gpustat
import random
import math
import numpy as np

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed, with repeated augmentation.
        It ensures that different each augmented version of a sample will be visible to a different process (GPU)
        Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class AverageMeter:
    """Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, list):
            val = np.array(val)
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1, )):
    """Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_logging(log_file):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def setup_gpus():
    """Adapted from https://github.com/bamos/setGPU/blob/master/setGPU.py
    """
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    best_gpu = min(pairs, key=lambda x: x[1])[0]
    # import pdb; pdb.set_trace()
    gpus_id = [int(gpu.entry['index']) for gpu in stats]
    return best_gpu, gpus_id


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def save_checkpoint(state, is_best, path, save_name="model_best", name='model_latest'):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = path + '/' + name + "_.pth.tar"
    torch.save(state, save_path)
    logging.info('checkpoint saved to {}'.format(save_path))
    if is_best:
        shutil.copyfile(save_path, path + '/' + save_name + '_model_best.pth.tar')


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


# item() is a recent addition, so that helps with backward compatibility
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
    

def log_train_results(bit_width_list, epoch, train_loss, val_loss, val_prec1, val_prec5):
    if len(bit_width_list) == 1:
        logging.info('epoch:{}, train loss(topk_bit)-> 1_32: {:.2f}, ****val prec: 1_32: {:.2f}, 5_32: {:.2f}****'.format(
            epoch, train_loss[0], val_prec1[0], val_prec5[0]))
    elif len(bit_width_list) == 4:
        logging.info('epoch:{}, train loss(topk_bit)-> 1_2: {:.2f} 1_4: {:.2f} 1_6: {:.2f} 1_8: {:.2f},\n \
                     ****val prec: 1_2: {:.2f}, 1_4: {:.2f}, 1_6: {:.2f}, 1_8: {:.2f}, 1_average:{:.2f}\n \
                     5_2: {:.2f}, 5_4: {:.2f}, 5_6: {:.2f}, 5_8: {:.2f}, 5_average:{:.2f}****'
                    .format(epoch, train_loss[0], train_loss[1], train_loss[2], train_loss[3],
                    val_prec1[0], val_prec1[1], val_prec1[2], val_prec1[3], np.mean(val_prec1),
                    val_prec5[0], val_prec5[1], val_prec5[2], val_prec5[3], np.mean(val_prec5)))
    elif len(bit_width_list) == 5:
        logging.info('epoch:{}, train loss(topk_bit)-> 1_2: {:.2f} 1_4: {:.2f} 1_6: {:.2f} 1_8: {:.2f} 1_32: {:.2f},\n \
                     ****val prec: 1_2: {:.2f}, 1_4: {:.2f}, 1_6: {:.2f}, 1_8: {:.2f}, 1_32: {:.2f}, 1_average:{:.2f}\n \
                     5_2: {:.2f}, 5_4: {:.2f}, 5_6: {:.2f}, 5_8: {:.2f}, 5_32: {:.2f}, 5_average:{:.2f}****'
                    .format(epoch, train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4],
                    val_prec1[0], val_prec1[1], val_prec1[2], val_prec1[3], val_prec1[4], np.mean(val_prec1),
                    val_prec5[0], val_prec5[1], val_prec5[2], val_prec5[3], val_prec5[4], np.mean(val_prec5)))
    elif len(bit_width_list) == 7:
        logging.info('Pref_val(hz):{:.2f}, val loss: {:.2f}, val prec1: {:.2f}, val prec5: {:.2f}'.format(
            val_loss[0], val_prec1[0], val_prec5[0]))
    elif len(bit_width_list) == 3:
        logging.info('epoch:{}, train loss(topk_bit)-> 1_2: {:.2f} 1_3: {:.2f} 1_4: {:.2f},\n \
                     ****val prec: 1_2: {:.2f}, 1_3: {:.2f}, 1_4: {:.2f}, 1_average:{:.2f}\n \
                     5_2: {:.2f}, 5_3: {:.2f}, 5_4: {:.2f}, 5_average:{:.2f}****'
                    .format(epoch, train_loss[0], train_loss[1], train_loss[2],
                    val_prec1[0], val_prec1[1], val_prec1[2], np.mean(val_prec1),
                    val_prec5[0], val_prec5[1], val_prec5[2], np.mean(val_prec5)))
    # elif len(bit_width_list) == 3:
    #     logging.info('epoch:{}, train loss(topk_bit)-> 1_4: {:.2f} 1_6: {:.2f} 1_8: {:.2f},\n \
    #                  ****val prec: 1_4: {:.2f}, 1_6: {:.2f}, 1_8: {:.2f}, 1_average:{:.2f}\n \
    #                  5_4: {:.2f}, 5_6: {:.2f}, 5_8: {:.2f}, 5_average:{:.2f}****'
    #                 .format(epoch, train_loss[0], train_loss[1], train_loss[2],
    #                 val_prec1[0], val_prec1[1], val_prec1[2], np.mean(val_prec1),
    #                 val_prec5[0], val_prec5[1], val_prec5[2], np.mean(val_prec5)))
    elif len(bit_width_list) == 2:
        logging.info('epoch:{}, train loss(topk_bit)-> 1_4: {:.2f} 1_8: {:.2f},\n \
                     ****val prec: 1_4: {:.2f}, 1_8: {:.2f}, 1_average:{:.2f}\n \
                     5_4: {:.2f}, 5_8: {:.2f}, 5_average:{:.2f}****'
                    .format(epoch, train_loss[0], train_loss[1], 
                    val_prec1[0], val_prec1[1], np.mean(val_prec1), 
                    val_prec5[0], val_prec5[1], np.mean(val_prec5)))
        
def log_evaluate_results(bit_width_list, val_loss, val_prec1, val_prec5):
    if len(bit_width_list) == 1:
        logging.info('Bit:{}, val loss: 1_32: {:.2f}, val prec1: {:.2f}, val prec5: {:.2f}'.format(
            str(bit_width_list), val_loss[0], val_prec1[0], val_prec5[0]))
    elif len(bit_width_list) == 4:
        logging.info('Bit:{}, val loss: 1_2: {:.2f} 1_4: {:.2f} 1_6: {:.2f} 1_8: {:.2f},\n \
                     ****val prec: 1_2: {:.2f}, 1_4: {:.2f}, 1_6: {:.2f}, 1_8: {:.2f}, 1_average:{:.2f}\n \
                                   5_2: {:.2f}, 5_4: {:.2f}, 5_6: {:.2f}, 5_8: {:.2f}, 5_average:{:.2f}****'
                     .format(str(bit_width_list), val_loss[0], val_loss[1], val_loss[2], val_loss[3], 
                        val_prec1[0], val_prec1[1], val_prec1[2], val_prec1[3], np.mean(val_prec1),
                        val_prec5[0], val_prec5[1], val_prec5[2], val_prec5[3], np.mean(val_prec5)))
    elif len(bit_width_list) == 7:
        logging.info('Pref_val(hz):{:.2f}, val loss: {:.2f}, val prec1: {:.2f}, val prec5: {:.2f}'.format(
            val_loss[0], val_prec1[0], val_prec5[0]))
    elif len(bit_width_list) == 3:
        logging.info('Bit:{}, val loss: 1_2: {:.2f} 1_3: {:.2f} 1_4: {:.2f},\n \
                     ****val prec: 1_2: {:.2f}, 1_3: {:.2f}, 1_4: {:.2f}, 1_average:{:.2f}\n \
                                   5_2: {:.2f}, 5_3: {:.2f}, 5_4: {:.2f}, 5_average:{:.2f}****'
                     .format(str(bit_width_list), val_loss[0], val_loss[1], val_loss[2],
                        val_prec1[0], val_prec1[1], val_prec1[2], np.mean(val_prec1),
                        val_prec5[0], val_prec5[1], val_prec5[2], np.mean(val_prec5)))
    # elif len(bit_width_list) == 3:
    #     logging.info('Bit:{}, val loss: 1_4: {:.2f} 1_6: {:.2f} 1_8: {:.2f},\n \
    #                  ****val prec: 1_4: {:.2f}, 1_6: {:.2f}, 1_8: {:.2f}, 1_average:{:.2f}\n \
    #                  5_4: {:.2f}, 5_6: {:.2f}, 5_8: {:.2f}, 5_average:{:.2f}****'
    #                  .format(str(bit_width_list), val_loss[0], val_loss[1], val_loss[2],
    #                     val_prec1[0], val_prec1[1], val_prec1[2], np.mean(val_prec1),
    #                     val_prec5[0], val_prec5[1], val_prec5[2], np.mean(val_prec5)))
    elif len(bit_width_list) == 2:
        logging.info('Bit:{}, val loss: 1_4: {:.2f} 1_8: {:.2f},\n \
                     ****val prec: 1_4: {:.2f}, 1_8: {:.2f}, 1_average:{:.2f}\n \
                                   5_4: {:.2f}, 5_8: {:.2f}, 5_average:{:.2f}****'
                     .format(str(bit_width_list), val_loss[0], val_loss[1], 
                        val_prec1[0], val_prec1[1], np.mean(val_prec1), 
                        val_prec5[0], val_prec5[1], np.mean(val_prec5)))
