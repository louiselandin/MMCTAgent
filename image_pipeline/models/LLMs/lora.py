import math
from typing import Any, Mapping
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
import deepspeed


class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 num_models=1,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None,
                 lora_dict=None):
        super(LinearLayer_LoRA, self).__init__()
        self.num_models = num_models
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        for i in range(self.num_models):
            setattr(
                self, "lora_right_weight_{}".format(i),
                nn.Parameter(torch.zeros(columns, lora_dim)))
            setattr(
                self, "lora_left_weight_{}".format(i),
                nn.Parameter(torch.zeros(lora_dim, rows)))
        self.lora_scaling = lora_scaling / lora_dim

        # self.lora_right_weight = nn.Parameter(torch.zeros(
        #     columns,
        #     lora_dim))  # apply transpose so in forward we do not need to
        # self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        # self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()
        if lora_dict is not None:
            self.load_state_dict(lora_dict)
        else:
            self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)
    
    def reset_parameters(self):
        for i in range(self.num_models):
            nn.init.kaiming_uniform_(getattr(self, "lora_right_weight_{}".format(i)), a=math.sqrt(5))
            nn.init.zeros_(getattr(self, "lora_left_weight_{}".format(i)))
        # nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self, model_id=-1):
        if self.num_models >= 1 and model_id == -1:
            raise Exception("Cannot fuse multiple LoRA weights")
        
        if not self.fuse_lora:
            if model_id < self.num_models:
                if self.num_models == 1:
                    model_id = 0
                self.weight.data += self.lora_scaling * torch.matmul(
                    getattr(self, "lora_left_weight_{}".format(model_id)).t(),
                    getattr(self, "lora_right_weight_{}".format(model_id)).t())
                self.fuse_lora = True
            else:
                raise Exception("Model id out of range should be less than {}".format(self.num_models))

    def unfuse_lora_weight(self, model_id=-1):
        if self.num_models >= 1 and model_id == -1:
            raise Exception("Cannot unfuse multiple LoRA weights")
        if self.fuse_lora:
            if model_id < self.num_models:
                if self.num_models == 1:
                    model_id = 0
                self.weight.data -= self.lora_scaling * torch.matmul(
                    getattr(self, "lora_left_weight_{}".format(model_id)).t(),
                    getattr(self, "lora_right_weight_{}".format(model_id)).t())
                self.fuse_lora = False
            else:
                raise Exception("Model id out of range should be less than {}".format(self.num_models))
            

    def forward(self, input, model_id=0):
        if model_id >= self.num_models:
            raise Exception("Model id out of range should be less than {}".format(self.num_models))
        if self.fuse_lora and self.num_models == 1:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ getattr(self, "lora_right_weight_{}".format(model_id))
                              @ getattr(self, "lora_left_weight_{}".format(model_id))) * self.lora_scaling


# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 num_models=1,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_droppout=0):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(
            module.weight, num_models, lora_dim, lora_scaling, lora_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]


# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.lora_left_weight,
                module.lora_right_weight
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()
    return model

def parse_multiple_lora_models(model, model_id):
    if model.num_models == 1:
        raise Exception("Cannot parse multiple LoRA params for single LoRA model")
    lora_params = []
    if model_id >= model.num_models and model_id<0:
        raise Exception("Model id out of range should be less than {}".format(model.num_models))
    for name, _ in model.name_parameters():
        if "lora_right_weight_{}".format(model_id) in name or "lora_left_weight_{}".format(model_id) in name:
            lora_params.append(name)
    for name in lora_params:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
            module.weight, module.bias, getattr(module, "lora_left_weight_{}".format(model_id)),
            getattr(module, "lora_right_weight_{}".format(model_id))
    ]),   
                                            modifier_rank=0,
                                            enabled=zero_stage_3):
            module.fuse_lora_weight(model_id) 
    return model
    
def parse_multiple_lora_params(model):
    if model.num_models == 1:
        raise Exception("Cannot parse multiple LoRA params for single LoRA model")
    lora_params = []
    for i in range(model.num_models):
        lora_ind_model_params = []
        for name, _ in model.name_parameters():
            if "lora_right_weight_{}".format(i) in name or "lora_left_weight_{}".format(i) in name:
                lora_ind_model_params.append(name)
        lora_params.append(lora_ind_model_params)
    state_dict = {}
    for i, lora_ind_model_params in enumerate(lora_params):
        for name in lora_ind_model_params:
            module = recursive_getattr(model, name)
            zero_stage_3 = hasattr(module.weight, 'ds_id')
            with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                getattr(module, "lora_left_weight_{}".format(i)),
                getattr(module, "lora_right_weight_{}".format(i))
        ]),   
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
                
                state_dict[f"{name}.lora_left_weight_{i}"] = getattr(module, "lora_left_weight_{}".format(i))
                state_dict[f"{name}.lora_right_weight_{i}"] = getattr(module, "lora_right_weight_{}".format(i))
    return state_dict

def only_optimize_lora_parameters(model, force_optimize_params=[]):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name or name in force_optimize_params:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model
