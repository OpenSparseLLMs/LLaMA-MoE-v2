import torch
import torch.nn as nn
import torch.nn.functional as F

top_k = 2

hidden_states = torch.randn(2, 4, 16)

hidden_dim = hidden_states.shape[-1]
num_experts = 4

experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)])

gate = nn.Linear(hidden_dim, num_experts, bias=False)

batch_size, sequence_length, hidden_dim = hidden_states.shape
hidden_states = hidden_states.view(-1, hidden_dim)
# router_logits: (batch * sequence_length, n_experts)
router_logits = gate(hidden_states)

# print(router_logits, router_logits.shape)  # [8,4]

logits = router_logits

scores = F.softmax(logits, dim=1, dtype=torch.float)
routing_weights, selected_experts = torch.topk(scores, top_k, dim=-1)  # top_k: 2 choose
routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
# we cast back to the input dtype
routing_weights = routing_weights.to(hidden_states.dtype)

final_hidden_states = torch.zeros(
    (batch_size * sequence_length, hidden_dim),
    dtype=hidden_states.dtype,
    device=hidden_states.device,
)

# One hot encode the selected experts to create an expert mask
# this will be used to easily index which expert is going to be sollicitated
expert_mask = torch.nn.functional.one_hot(
    selected_experts, num_classes=num_experts
).permute(2, 1, 0)

"""
selected_experts:
tensor([[3, 2],
        [0, 2],
        [2, 3],
        [1, 3],
        [2, 3],
        [3, 2],
        [2, 0],
        [0, 1]])

Each row denotes an expert, and each column denotes a token.
There are two lines as top-2 tokens are selected.
expert_mask:
tensor([[[0, 1, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 0]],

        [[0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1]],

        [[0, 0, 1, 0, 1, 0, 1, 0],
         [1, 1, 0, 0, 0, 1, 0, 0]],

        [[1, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0]]])
"""

# Loop over all available experts in the model and perform the computation on each expert
for expert_idx in range(num_experts):
    expert_layer = experts[expert_idx]
    idx, top_x = torch.where(expert_mask[expert_idx])
    print("top_x: ", top_x, top_x.shape)  # top_x: tensor([1, 2, 7])  torch.Size([3])

    if expert_idx == 0:
        top_x_list = []
        idx_list = []
        top_x = torch.empty(0, dtype=torch.long)

    if top_x.shape[0] == 0:
        # print("Warning!!! No expert selected")   # 🔍
        continue

    # in torch it is faster to index using lists than torch tensors
    top_x_list = top_x.tolist()
    idx_list = idx.tolist()

    # import pdb; pdb.set_trace()
    # Index the correct hidden states and compute the expert hidden state for
    # the current expert. We need to make sure to multiply the output hidden
    # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
    current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
    current_hidden_states = (
        expert_layer(current_state)
        * routing_weights[
            top_x_list, idx_list, None
        ]  # may error when routing_weights[top_x_list, idx_list, None]
    )

    import pdb

    pdb.set_trace()
    # However `index_add_` only support torch tensors for indexing so we'll use
    # the `top_x` tensor here.
    final_hidden_states.index_add_(
        0, top_x, current_hidden_states.to(hidden_states.dtype)
    )  # why? when no token is choosen by this expert.

final_hidden_states = final_hidden_states.reshape(
    batch_size, sequence_length, hidden_dim
)

final_hidden_states.sum().backward()  # 🔍

import pdb

pdb.set_trace()
