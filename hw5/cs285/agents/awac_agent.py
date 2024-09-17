from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)
        #不论有多少个活动关节，只要动作是离散的，则将所有可能的动作组合枚举出来并进行编号，由神经网络预测哪个动作（编号）最大
        #所以网络输出形状为（关节数， 每个关节的动作离散取值），即（action_dim, num_per_action） 本实验中action_dim=1, num_per_action=5(上下左右原地共五种动作可能)
        self.actor = make_actor(observation_shape, num_actions) # num_per_action = 5,
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            dist = self.actor(next_observations) # shape = (batch_size, num_per_action)
            action_by_policy = dist.sample() # shape = (batch_size,)
            next_qa_values = self.target_critic(next_observations) # shape = (batch_size, num_per_action)

            # Use the actor to compute a critic backup
            next_qs = torch.gather(next_qa_values, -1, torch.unsqueeze(action_by_policy,dim=1)).squeeze() # shape = (batch_size)

            # TODO(student): Compute the TD target
            target_values = rewards + (1 - dones.int()) * self.discount * next_qs

        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)
        q_values = torch.gather(qa_values, -1, torch.unsqueeze(actions, 1)).squeeze()
        assert q_values.shape == target_values.shape

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

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        with torch.no_grad():
            qa_values = self.critic(observations)###################################target+critic
        q_values = torch.gather(qa_values, -1, torch.unsqueeze(actions, 1)).squeeze()
        # values = torch.mean(qa_values * action_dist, dim=-1) --------------------------  (regard to V_s)try to see the difference between mean and argmax action
        action_by_policy = action_dist.sample()
        values = torch.gather(qa_values, -1, torch.unsqueeze(action_by_policy, 1)).squeeze()
        advantages = q_values - values
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        dist = self.actor(observations)
        log_prob = dist.log_prob(actions)
        advantages = self.compute_advantage(observations, actions, dist)
        weight = torch.exp(advantages / self.temperature)
        loss = - (log_prob * weight).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm or float('inf'))
        self.actor_optimizer.step()

        return loss.item(), advantages.mean(), actor_grad_norm

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss, advantages, actor_grad_norm = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss
        metrics["actor_advantage"] = advantages
        metrics["actor_grad_norm"] = actor_grad_norm
        return metrics
