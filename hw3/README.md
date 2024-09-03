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

## 3 Continuous Actions with Actor-Critic

 ### 3.1.1 Bootstrapping-Pendulum
 -  ![Pendulum_Qvalue](result_picture/Pendulum_Qvalue.png)
 Q-values stabilize at about 900.

 ### 3.1.3 Actor with REINFORCE
 -  ![IP1000](result_picture/IP1000.png)
 Eval returns are close to 1000.(Inverted Pendulum)
 -  ![Rein110](result_picture/Rein110.png)
 Using 10 samples(red) to estimate object function has lower variance and gains more returns than just using one sample(blue) to estimate it.(Halfcheetah)

  ### 3.1.4 Actor with REPARAMETRIZE
-  ![IP_REPvsREIN](result_picture/IP_REPvsREIN.png)
(Inverted Pendulum)  
Both REPARAMETRIZE(blue) and REINFORCE(orange) achieve similar reward and REPARAMETRIZE performs better.
 -  ![REPvsREIN1vs10](result_picture/REPvsREIN1vs10.png)
 (Halfcheetah)  
 reparametrize(red) has the best performance compared with reinforcement1(orange) and reinforcement10(blue).
-   ![eval3](result_picture/eval3.png)
    ![qvalue3](result_picture/qvalue3.png)
 (Halfcheetah)
Orange: num_critic_updates->1, num_action_samples->1  
Blue: num_critic_updates->10, num_action_samples->1  
Red: num_critic_updates->1, num_action_samples->10  
Both eval_return and qValues are the highest in orange curve.

 -  ![human_sac](result_picture/human_sac.png)

   ### 3.1.5 Stabilizing Target Values
-   ![QQQreturn](result_picture/QQQreturn.png)
 ![QQQvalue](result_picture/QQQvalue.png)
(Hopper)  
clipQ(red) has the best performance and lowest q_values, which means clipq's Q-values has the most accurate estimation. The single-Q(orange)'s performance is the worst, and the most overestimated q-values. doubleQ(blue)'s perforance is medium.
-   ![pg_evalreturn](result_picture/pg_evalreturn.png)
    ![rep_evalreturn](result_picture/rep_evalreturn.png)
    ![rep_qvalues](result_picture/rep_qvalues.png)
    Orange: Policy Gradient  
    Blue: Reparametrize(mean q values)  
    Red: Reparametrize(min q values)  
    Obviously the performance of the Red curve is much much much better than the others!!!!
    Off-policy algorithm is more effective than on-policy algorithm. In terms of time consuming, off-policy took about 25h while on-policy took about 120h on my laptop.
