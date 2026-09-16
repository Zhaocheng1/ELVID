import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#
# class Expertset(object):
#     def __init__(self, num_experts, gates):
#         self._gates = gates
#         self._num_experts = num_experts
#         sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
#         _, self._expert_index = sorted_experts.split(1, dim=1)
#         self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
#         self._part_sizes = (gates > 0).sum(0).tolist()
#         gates_exp = gates[self._batch_index.flatten()]#w我想用这个作为专家网络用于融合任务
#         self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
#     def es(self, inp):
#         inp_exp = inp[self._batch_index].squeeze(1)
#         return torch.split(inp_exp, self._part_sizes, dim=0)
#     def ee(self, expert_out, multiply_by_gates=True):
#         stitched = torch.cat(expert_out, 0).exp()
#         if multiply_by_gates:
#             stitched = stitched.mul(self._nonzero_gates)
#         zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
#         ensemble = zeros.index_add(0, self._batch_index, stitched.float())
#         ensemble[ensemble == 0] = np.finfo(float).eps
#         return ensemble.log()
#     def expert_to_gates(self):
#         return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
# # class Expertset:
#     def __init__(self, num_experts, gates):
#         self.num_experts = num_experts
#         self.gates = gates  # [N, E]
#
#     def es(self, x):  # x: [N, in_features]
#         expert_inputs = []
#         for i in range(self.gates.shape[1]):
#             expert_inputs.append(x * self.gates[:, i:i+1])  # 按专家权重加权输入
#         return expert_inputs
#
#     def ee(self, expert_outputs):  # List of [N, out_features]
#         out = 0
#         for i in range(len(expert_outputs)):
#             out = out + expert_outputs[i] * self.gates[:, i:i+1]  # 按权重融合输出
#         return out
class Expertset:
    def __init__(self, num_experts, gates):
        self._gates = gates  # [N, E]
        self._num_experts = num_experts

        nonzero_indices = torch.nonzero(gates, as_tuple=False)  # [M, 2]
        self._batch_index = nonzero_indices[:, 0]  # 样本索引
        self._expert_index = nonzero_indices[:, 1]  # 对应专家

        self._part_sizes = [(self._expert_index == i).sum().item() for i in range(num_experts)]
        self._nonzero_gates = gates[self._batch_index, self._expert_index].unsqueeze(1)

    def es(self, inp):
        inp_exp = inp[self._batch_index]  # [M, C]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def ee(self, expert_outs, multiply_by_gates=True):
        stitched = torch.cat(expert_outs, dim=0)  # [M, C]
        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates  # [M, 1] * [M, C] = broadcast
        output = torch.zeros(self._gates.size(0), stitched.size(1), device=stitched.device)
        output = output.index_add(0, self._batch_index, stitched)
        output[output == 0] = torch.finfo(torch.float32).eps
        return output