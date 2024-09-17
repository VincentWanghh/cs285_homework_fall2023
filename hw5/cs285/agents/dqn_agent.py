from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn, distributions

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int], num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        q_values = self.critic(observation)
        best_action_index = torch.argmax(q_values,-1)
        n, m = q_values.shape
        probs = torch.ones(n,m) * epsilon / (m - 1)
        probs[torch.arange(n),best_action_index] = 1 - epsilon
        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        dist = distributions.Categorical(probs = probs)
        action = dist.sample()

        return ptu.to_numpy(action).squeeze(0).item()

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Compute the loss for the DQN critic.

        Returns:
         - loss: torch.Tensor, the MSE loss for the critic
         - metrics: dict, a dictionary of metrics to log
         - variables: dict, a dictionary of variables that can be used in subsequent calculations
        """

        # TODO(student): paste in your code from HW3, and make sure the return values exist
        # raise NotImplementedError
        with torch.no_grad():
            # TODO(student): compute target values
            next_qa_values = self.critic(next_obs)
            if self.use_double_q:
                next_best_action_index = torch.argmax(next_qa_values,-1,keepdim=True)
                target_next_q_values = self.target_critic(next_obs)
                next_q_values = torch.gather(target_next_q_values, -1, next_best_action_index).squeeze()
                # raise NotImplementedError
            else:
                next_q_values = torch.max(next_qa_values, -1)[0]#--------------------------------------注意这个不同，用self.critic还是self.target_critic--------------------------------------------------------#
            done = done.to(torch.int8)
            target_values = reward + (1 - done) * self.discount * next_q_values

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)
        q_values = torch.gather(qa_values, -1, torch.unsqueeze(action.long(),1)).squeeze() # Compute from the data actions; see torch.gather
        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        loss, metrics, _ = self.compute_critic_loss(obs, action, reward, next_obs, done)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        metrics["grad_norm"] = grad_norm.item()
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return metrics

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): paste in your code from HW3
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()
        return critic_stats
