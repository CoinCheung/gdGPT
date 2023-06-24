# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from dataclasses import dataclass
from deepspeed import comm as dist


@dataclass
class fragment_address:
    numel: int
    start: int


@dataclass
class tensor_fragment:
    lp_fragment: torch.Tensor
    lp_fragment_address: fragment_address
    hp_fragment: torch.Tensor
    hp_fragment_address: fragment_address
    optim_fragment: {}
    gradient_dict: {}
    offload_gradient_dict: {}
    use_offload: bool
    param_group_index: int

    def update_hp(self):
        self.hp_fragment.data.copy_(self.lp_fragment.data)

    def update_lp(self):
        self.lp_fragment.data.copy_(self.hp_fragment.data)

    def get_optim_state_fragment(self, key):
        if key in self.optim_fragment:
            return self.optim_fragment[key]
        else:
            raise ValueError(f'{key} not found in optimizer state fragment')

    def get_hp_fragment_address(self):
        return self.hp_fragment_address

    def get_optim_state_keys(self):
        return list(self.optim_fragment.keys())


def get_full_hp_param(self, optim_state_key=None):
    reduce_buffer = torch.zeros_like(self, dtype=torch.float32).flatten()
    if self._hp_mapping is not None:
        lp_frag_address = self._hp_mapping.lp_fragment_address
        reduce_fragment = torch.narrow(reduce_buffer, 0, lp_frag_address.start, lp_frag_address.numel)
        if optim_state_key is None:
            hp_fragment = self._hp_mapping.hp_fragment
        else:
            hp_fragment = self._hp_mapping.get_optim_state_fragment(optim_state_key)

        reduce_fragment.data.copy_(hp_fragment.data)
    dist.all_reduce(reduce_buffer, group=self._dp_group)
    return reduce_buffer.reshape_as(self)


def get_full_hp_grad(self):
    reduce_buffer = torch.zeros_like(self, dtype=torch.float32).flatten()
    if self._hp_mapping is not None:
        hp_mapping = self._hp_mapping

        if hp_mapping.use_offload:
            gradient_dict = hp_mapping.offload_gradient_dict
        else:
            gradient_dict = hp_mapping.gradient_dict

        if hp_mapping.param_group_index not in gradient_dict or gradient_dict[hp_mapping.param_group_index] is None:
            raise ValueError("Gradients are only available immediately after backward and before engine step")

        lp_grad_fragment = gradient_dict[hp_mapping.param_group_index][self._index_in_param_group]
        hp_grad_fragment = lp_grad_fragment.to(torch.float32).flatten()

        lp_frag_address = self._hp_mapping.lp_fragment_address
        reduce_fragment = torch.narrow(reduce_buffer, 0, lp_frag_address.start, lp_frag_address.numel)

        if self.view(-1).shape == hp_grad_fragment.shape:
            reduce_buffer.data.copy_(hp_grad_fragment.data)
        else:
            reduce_fragment.data.copy_(hp_grad_fragment.data)

    dist.all_reduce(reduce_buffer, group=self._dp_group)
    return reduce_buffer.reshape_as(self)


def safe_get_full_fp32_param(param):
    """Assemble and return the fp32 parameter of a low-precision (e.g., fp16) parameter.

        Args:
            param (``torch.nn.Parameter``): A model parameter
    """
    # ZeRO stage 3 param
    if hasattr(param, 'ds_id'):
        return param._z3_optimizer.get_full_hp_param(param)

    # ZeRO stage 1, 2, and bf16_optimizer params
    if hasattr(param, '_hp_mapping'):
        return param.get_full_hp_param()
    return None


def safe_get_full_optimizer_state(param, optim_state_key):
    """Assemble and return the fp32 optimizer state of a low-precision (e.g., fp16) parameter.

        Args:
            param (``torch.nn.Parameter``): A model parameter
    """
    # ZeRO stage 3 param
    if hasattr(param, 'ds_id'):
        return param._z3_optimizer.get_full_hp_param(param, optim_state_key)

    # ZeRO stage 1, 2, and bf16_optimizer params
    if hasattr(param, '_hp_mapping'):
        return param.get_full_hp_param(optim_state_key)
    return None


# TODO: Figure out the correct return dtype
def safe_get_full_grad(param):
    """Assemble and return the fp32 gradient of a low-precision (e.g., fp16) parameter.

        Args:
            param (``torch.nn.Parameter``): A model parameter
    """
    if param.grad is not None:
        return param.grad

    # ZeRO stage 3 param
    if hasattr(param, 'ds_id'):
        return param._z3_optimizer.get_fp32_grad_for_param(param)

    # ZeRO stage 1, 2, and bf16_optimizer params
    if hasattr(param, '_hp_mapping'):
        return param.get_full_hp_grad()

    return None


def get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, gradient_dict, offload_gradient_dict, use_offload,
                            param_group_index, partition_start, partition_size, optimizer_state_dict):
    lp_end = lp_param.numel() + lp_start
    hp_start = partition_start
    hp_end = partition_start + partition_size

    fragment_start = max(lp_start, hp_start)
    fragment_end = min(lp_end, hp_end)
    assert fragment_start < fragment_end, \
        f'fragment start {fragment_start} should be < fragment_end {fragment_end}'

    fragment_numel = fragment_end - fragment_start
    hp_frag_address = fragment_address(start=fragment_start - hp_start, numel=fragment_numel)
    hp_fragment_tensor = flat_hp_partition.narrow(0, hp_frag_address.start, hp_frag_address.numel)
    optim_fragment = {
        key: value.narrow(0, hp_frag_address.start, hp_frag_address.numel)
        for key, value in optimizer_state_dict.items()
        if torch.is_tensor(value) and value.shape == flat_hp_partition.shape
    }

    lp_frag_address = fragment_address(start=fragment_start - lp_start, numel=fragment_numel)
    lp_fragment_tensor = lp_param.flatten().narrow(0, lp_frag_address.start, lp_frag_address.numel)

    return tensor_fragment(lp_fragment=lp_fragment_tensor,
                           lp_fragment_address=lp_frag_address,
                           hp_fragment=hp_fragment_tensor,
                           hp_fragment_address=hp_frag_address,
                           optim_fragment=optim_fragment,
                           gradient_dict=gradient_dict,
                           offload_gradient_dict=offload_gradient_dict,
                           use_offload=use_offload,
                           param_group_index=param_group_index)


'''
Logic for lp_param to hp_param mapping

lp      lp0 lp1 lp2         lp3  lp4            <-------  indices/names
lp      [  ][  ][          ][   ][         ]    <-------- tensors
flat_lp [                                  ]     <-------- flat lp params
flat_hp            [                 ]   <------------------ flat hp partition on current rank
full_hp [                                        ] <------- full flat hp params


lp2
 full numel = 16
 lp_frag
   numel = 12
   frag_start = 3
   frag_end  = 15
 hp_frag
    numel = 12
    frag_start = 0
    frag_end = 11

 hp_frag.copy_(lp_frag)


lp3:
  full numel = 4
  lp_frag
     numel = 4
     start = 0
     end = 3
  hp_frag
     numel = 4
     start = 12
     end = 15


lp4:
   full numel = 12
   lp_frag
     numel = 4
     start = 0
     end = 3
  hp_frag
     numel = 4
     start = 16
     end = 19



Visual depiction of above
lp              {         }
flat_lp [                                ]
flat_hp            (                 )


flat_lp [       {  (      }          )   ]
                lx  hx   ly          hy
                    ly-hx


lp                             {       }
flat_lp [                                ]
flat_hp            (                 )


flat_lp [          (            {     ) }  ]
                   hx           lx   hy ly
                                   hy-lx

lp                        {   }
flat_lp [                                ]
flat_hp            (                 )


flat_lp [          (       {   }      )   ]
                   hx      lx  ly    hy
                             ly-lx

lp -> (lx, hy)
flat_hp -> (hx, hy)
'''
