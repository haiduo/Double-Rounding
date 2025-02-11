# Nearly Lossless Adaptive Bit Switching
This repository is the official implementation of "Nearly Lossless Adaptive Bit Switching", which includes training, evaluation, and other related scripts.  

<img width="572" alt="image" src="images\quantization_types.png"> <br>
_Figure 1: Comparison between different quantization types during quantization-aware training._

>📋 We propose a bit-switching quantization method that doesn't require storing a full-precision model and achieves nearly lossless switching from high-bits to low-bits. Specifically, we propose unified representation, normalized learning steps, and tuned probability distribution for different precisions so that an efficient and stable learning process is achieved across multiple and mixed precisions, as depicted in Figure 2.

<img width="572" alt="image" src="images\overview.png"> <br>
_Figure 2: Overview of our proposed lossless adaptive bit-switching strategy._

### The structure of the code
* code <br>
  * Multi_Precision&nbsp;&nbsp; -->For training Multi_Precision models
  * Super_net&nbsp;&nbsp; -->For training one-shot Mixed_Precision SuperNets
  * Super_net_solve&nbsp;&nbsp; -->For searching Mixed_Precision SubNets
* data 
  * cifar10
  * Imagenet-1K

 
## Requirements
We have tested the code on the following environments and settings:

* Python 3.8.19 / Pytorch (>=1.6.0) / torchvision (>=0.7.0)
* Prepare ImageNet-1k data following pytorch [example](https://github.com/pytorch/examples/tree/main/imagenet).

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Set up the environment, e.g. we use conda to build our code.

## Training

To train the model(s) in the paper, run this command:

```train
First, need to attain a Pre-trained FP32 model by:
    python code/Multi_Precision/train.py  -bit_width_list='32' ... etc.

Then, 
For multi-precision:
    python code/Multi_Precision/train.py -bit_width_list='8,6,4,2' ... etc.
For mixed-precision:
    (1) Attain the Hessian Matrix Trace:
        python code/Super_net_solve/cal_hessian.py -bit_width_list='32'  ... etc.
    (2) Train Mixed-precision models
        python code/Super_net/train.py -bit_width_list='8,6,4,2' ... etc.
For SubNets:
    python code/Super_net_solve/cal_hessian.py --cal_solve

Note that Multi-GPUs parallel training needs to be turned on: --multiprocessing_distributed
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python code/Multi_Precision/train.py --evaluate ... etc.
```


## Pre-trained Models

You can download pre-trained models here:

- coming soon. 

## Reference
For technical details and full experimental results, please check [the paper of Double rounding](https://arxiv.org/abs/2502.01199).
```
@misc{huang2025nearlylosslessadaptivebit,
      title={Nearly Lossless Adaptive Bit Switching}, 
      author={Haiduo Huang and Zhenhua Liu and Tian Xia and Wenzhe zhao and Pengju Ren},
      year={2025},
      eprint={2502.01199},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.01199}, 
}
```



 
 
