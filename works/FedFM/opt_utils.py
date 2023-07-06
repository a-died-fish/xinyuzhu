import numpy as np
from collections import OrderedDict
import torch

def get_aggregated_weights_np_list(nets_this_round, fed_avg_freqs):
    for net_id, net in enumerate(nets_this_round.values()):
        if net_id == 0:
            weights_np_list = [val.cpu().numpy()*fed_avg_freqs[net_id] for _, val in net.state_dict().items()]
        else:
            for i, val in enumerate(net.state_dict().values()):
                weights_np_list[i] += val.cpu().numpy()*fed_avg_freqs[net_id]

    return weights_np_list


def aggregation_adagrad(pre_weights_np_list, aggregated_weights_np_list, m_t, v_t):
    delta_t = [x - y for x, y in zip(aggregated_weights_np_list, pre_weights_np_list)]
    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    
    beta_1 = 0.0
    m_t = [beta_1 * x + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]
    
    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [x + np.multiply(y, y) for x, y in zip(v_t, delta_t)]

    eta = 0.1
    tau = 1e-3
    new_weights_np_list = [
            x + eta * y / (np.sqrt(z) + tau)
            for x, y, z in zip(pre_weights_np_list, m_t, v_t)
        ]
    
    return new_weights_np_list, m_t, v_t

def aggregation_adam(pre_weights_np_list, aggregated_weights_np_list, m_t, v_t):
    delta_t = [x - y for x, y in zip(aggregated_weights_np_list, pre_weights_np_list)]
    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    
    beta_1 = 0.9
    m_t = [beta_1 * x + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]
    
    beta_2 = 0.99
    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [
            beta_2 * x + (1 - beta_2) * np.multiply(y, y)
            for x, y in zip(v_t, delta_t)
        ]

    eta = 0.01
    tau = 1e-3
    new_weights_np_list = [
            x + eta * y / (np.sqrt(z) + tau)
            for x, y, z in zip(pre_weights_np_list, m_t, v_t)
        ]
    
    return new_weights_np_list, m_t, v_t

def aggregation_yogi(pre_weights_np_list, aggregated_weights_np_list, m_t, v_t):
    delta_t = [x - y for x, y in zip(aggregated_weights_np_list, pre_weights_np_list)]
    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    
    beta_1 = 0.9
    m_t = [beta_1 * x + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]
    
    beta_2 = 0.99
    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [
            x - (1.0 - beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
            for x, y in zip(v_t, delta_t)
        ]

    eta = 0.01
    tau = 1e-3
    new_weights_np_list = [
            x + eta * y / (np.sqrt(z) + tau)
            for x, y, z in zip(pre_weights_np_list, m_t, v_t)
        ]
    
    return new_weights_np_list, m_t, v_t

def set_model_using_np_list(net, weights_np_list):
    net_keys = net.state_dict().keys()
    state_dict = {}
    for key, weights in zip(net_keys, weights_np_list):
        state_dict[key] = torch.tensor(weights)
    net.load_state_dict(state_dict, strict=True)