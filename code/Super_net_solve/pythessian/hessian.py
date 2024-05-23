
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np

from pythessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal


class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
    """
    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True, logging=None, a_H=False, w_H=False, all_steps=512):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """
        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion
        self.a_H = a_H
        self.w_H = w_H
        self.all_steps=all_steps

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        features_in = []
        grads_in = []
        hooks = []

        def register_hook(module):
            # forward hook function
            def hook_fn_forward(module, input, output):
                m_name = str(module.__class__).split(".")[-1].split("'")[0]
                logging.info(f'{m_name} input: {list(input[0].size())}, output: {list(output.size())}')
                features_in.append(input[0])
            
            # backward hook function
            def hook_fn_backward(module, grad_input, grad_output):
                m_name = str(module.__class__).split(".")[-1].split("'")[0]
                if 'Conv2d' in m_name:
                    # import pdb; pdb.set_trace();
                    logging.info(f'{m_name} grad_input: {list(grad_input[0].size())}, grad_output: {list(grad_output[0].size())}') 
                    grads_in.append(grad_input[0])
                elif 'Linear' in m_name:
                    # import pdb; pdb.set_trace();
                    logging.info(f'{m_name} grad_input: {list(grad_input[1].size())}, grad_output: {list(grad_output[0].size())}') 
                    grads_in.append(grad_input[1])

            if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
                temp_str = str(module.__class__).split(".")[-1].split("'")[0]
                if 'Conv2d' in temp_str or 'Linear' in temp_str:
                    hooks.append(module.register_forward_hook(hook_fn_forward))
                    hooks.append(module.register_backward_hook(hook_fn_backward))
                    # hooks.append(module.register_full_backward_hook(hook_fn_backward))
        
        # register hook
        if a_H:
            logging.info("register forwatd and backward hook of model")
            model.apply(register_hook)

        # pre-processing for single batch case to simplify the computation.

        if self.full_dataset:
            self.inputs, self.targets = next(iter(self.data))
        else:
            self.inputs, self.targets = self.data
        if self.device == 'cuda':
            self.inputs, self.targets = self.inputs.cuda().requires_grad_(), self.targets.cuda()

        # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
        outputs = self.model(self.inputs)
        loss = self.criterion(outputs, self.targets)
        if self.full_dataset:
            loss.backward(create_graph=True)
        else:
            loss.backward(create_graph=True)

        if w_H: # this step is used to extract the parameters from the model
            logging.info("extract the parameters from the model")
            params, gradsH = get_params_grad(self.model, full_dataset=self.full_dataset, logging=logging)
            self.params = params
            self.gradsH = gradsH  # gradient used for Hessian computation
        if a_H:
            self.inputs = features_in
            self.gradin = grads_in[::-1]

        # remove these hooks
        for h in hooks:
            h.remove()
            del h

    def dataloader_hv_product(self, v, whole_model=False, temp_params = None, logging=None):
        features_in = []
        grads_in = []
        hooks = []
        def register_hook(module):
            # forward hook function
            def hook_fn_forward(module, input, output):
                # m_name = str(module.__class__).split(".")[-1].split("'")[0]
                # logging.info(f'{m_name} input: {list(input[0].size())}, output: {list(output.size())}')
                features_in.append(input[0])
            
            # backward hook function
            def hook_fn_backward(module, grad_input, grad_output):
                m_name = str(module.__class__).split(".")[-1].split("'")[0]
                if 'Conv2d' in m_name:
                    #import pdb;
                    # logging.info(f'{m_name} grad_input: {list(grad_input[0].size())}, grad_output: {list(grad_output[0].size())}') 
                    grads_in.append(grad_input[0])
                elif 'Linear' in m_name:
                    #import pdb;
                    # logging.info(f'{m_name} grad_input: {list(input[1].size())}, grad_output: {list(grad_output[0].size())}') 
                    grads_in.append(grad_input[1])

            if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == self.model)):
                temp_str = str(module.__class__).split(".")[-1].split("'")[0]
                if 'Conv2d' in temp_str or 'Linear' in temp_str:
                    hooks.append(module.register_forward_hook(hook_fn_forward))
                    hooks.append(module.register_backward_hook(hook_fn_backward)) 
                    # hooks.append(module.register_full_backward_hook(hook_fn_backward))
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in self.params]  # accumulate result
        if self.a_H:
            logging.info("register forwatd and backward hook of model")
            self.model.apply(register_hook)
        if self.w_H:
            logging.info("extract the parameters from the model")
        i = 1
        for inputs, targets in self.data:
            if i > self.all_steps:
                break
            logging.info(f"------{i}/{int(self.all_steps)}--------")
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device)).requires_grad_()
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)

            if self.w_H:
                assert not self.a_H, 'not .....'
                params, gradsH = get_params_grad(self.model, logging=logging)

            if self.a_H:
                params = features_in
                gradsH = grads_in[::-1]
            
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False)
            THv = [THv1 + Hv1 * float(tmp_num_data) + 0. for THv1, Hv1 in zip(THv, Hv)]
            num_data += float(tmp_num_data)

            grads_in.clear()
            features_in.clear()
            i += 1

        for h in hooks:
            h.remove()
            del h

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v, whole_model=whole_model)
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1, whole_model=False, opt_type="weight", logging=None):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """
        assert top_n >= 1
        device = self.device
        eigenvalues = []
        eigenvectors = []
        computed_dim = 0
        if opt_type == "weight":
            temp_params = self.params
            temp_grad = self.gradsH
        elif opt_type == "activate":
            temp_params = self.inputs
            temp_grad = self.gradin

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if i == maxIter-1:
                    d_graph = True
                else:
                    d_graph = False

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v, whole_model=whole_model, temp_params=temp_params, logging=logging)
                else:
                    Hv = hessian_vector_product(temp_grad, temp_params, v, d_graph)
                    tmp_eigenvalue = group_product(Hv, v, whole_model=whole_model)
                v = normalization(Hv)
                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if (np.abs(np.subtract(eigenvalue, tmp_eigenvalue)) / np(abs(eigenvalue) + 1e-6) < tol).all() == True:
                        logging.info(f"The iterations number used to compute the Top {computed_dim+1} eigenvalue is :{i+1}")
                        if not self.full_dataset:
                            d_graph = True
                            _ = hessian_vector_product(temp_grad, temp_params, v, d_graph)
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            if i == maxIter-1:
                logging.info(f"The maximum iterations number used to compute the Top {computed_dim} eigenvalue not Convergence!")
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3, whole_model=False, opt_type="weight", logging=None):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """
        device = self.device
        trace_vhv = []
        trace = 0.
        if opt_type == "weight":
            temp_params = self.params
            temp_grad = self.gradsH
        elif opt_type == "activate":
            temp_params = self.inputs
            temp_grad = self.gradin

        for i in range(maxIter):
            self.model.zero_grad()
            v = [torch.randint_like(p, high=2, device=device) for p in self.params]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if i == maxIter-1:
                d_graph = True
            else:
                d_graph = False

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v, whole_model=whole_model, temp_params=temp_params, logging=logging)
            else:
                Hv = hessian_vector_product(temp_grad, temp_params, v, d_graph)
            trace_vhv.append(group_product(Hv, v, whole_model=whole_model))
            if (np.abs(np.subtract(np.mean(trace_vhv, axis=0), trace)) / (np.abs(trace) + 1e-6) < tol).all() == True:
                logging.info(f"The iterations number used to compute trace is :{i+1}")
                if not self.full_dataset:
                    d_graph = True
                    _ = hessian_vector_product(temp_grad, temp_params, v, d_graph)
                return np.mean(trace_vhv, axis=0)
            else:
                trace = np.mean(trace_vhv, axis=0)
        logging.info(f"The maximum iterations number used to compute trace not Convergence!")
        return np.mean(trace_vhv, axis=0)
