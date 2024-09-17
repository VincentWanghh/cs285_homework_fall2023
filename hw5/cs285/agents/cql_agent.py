from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )
        # CQL惩罚对非最优动作的学习，
        # 也就是说对于一个给定的状态s_t, 如果网络生成的所有q(s_t, a_t）都彼此相差不大，
        # 这说明我在进行argmax操作选择最优动作时有很大的误差(有可能因为critic网络的预测误差选择了q值最大的“最优”动作， 而错过了实际上的最优动作)
        # 此时通过引入penalty增大loss，促使网络预测结果发生大改变。
        # 而当对于一个给定的状态s_t, 如果网络生成的所有q(s_t, a_t）有明显的最大值（也就是说有一个a_t的q值远远大于其他所有a_t的q值），
        # 这说明在进行argmax操作选择最优动作时误差较小， penalty也随之降低  
        # TODO(student): modify the loss to implement CQL
        # Hint: `variables` includes qa_values and q_values from your CQL implementation
        qa_values = variables["qa_values"]
        q_values = variables["q_values"]# q_values.shape = (batch_size, action_size)
        penalty = torch.log(torch.exp(qa_values / self.cql_temperature).sum(dim=-1)) # shape = (batch_size,)
        loss = loss + self.cql_alpha * (penalty - q_values).mean()
        metrics["critic_loss"] = loss.item()
        return loss, metrics, variables
