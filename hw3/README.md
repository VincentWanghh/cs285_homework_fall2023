## 2 Deep Q-learning

### 2.4 Basic Q-learning
DQN algorithm implementation
- CartPole-v1 eval return
`python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml`
![CartPole](result_picture/CartPole.png)

- LunarLander-v2 eval return over seed 1, 2, 3
![LunarLander](result_picture/LunarLander.png)

- CartPole-v1 with learning rate 0.05(blue) and 0.001(orange)
  1. Predicted Q-values
  ![Predicted Q-values](result_picture/Qvalues.png)
  2. Critic Loss
  ![Critic error](result_picture/criticLoss.png)
  3. Eval Return
  ![Critic error](result_picture/CartPole_evalReturn.png)

    Q values and critic loss seem to be overestimated with learning rate 0.05

### 2.5 Double Q-Learning
#### LunarLander-v2
- eval return(double q learning v.s. policy gradient)
double q learning
![double_qlearning_3seed](result_picture/double_qlearning_3seed.png)
policy gradient
![PG](result_picture/PG.png)
  Double q learning is fairly stable compared to PG.

- eval return over seed 1, 2, 3(red: double q learning; blue: q learning)
seed 1
![seed1](result_picture/eval_eturn_double_QL_seed1.png)
seed 2
![seed2](result_picture/eval_eturn_double_QL_seed2.png)
seed 3
![seed3](result_picture/eval_eturn_double_QL_seed3.png)
The reason for the difference might be the output of q values network in double q learning is more stable than the output in q learning.

- MsPacman eval return v.s. train return
eval return
![eval return](result_picture/mspacman_eval_return.png)
train return
![train return](result_picture/train_return.png)
Early in training, eval return is more stable and higher than train return. This is because when sampling an action from the DQN agent, training uses a higher epsilon(1 or 0.5) than the default epsilon(0.02) used by evaluating.