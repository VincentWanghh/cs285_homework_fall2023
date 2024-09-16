# Exploration Strategies and Offline Reinforcement learning
## Problem1 Exploration
### Random Policy
- Easy environment
![easy](result_picture/PointmassEasy-v0_random.png)
- Medium environment
![medium](result_picture/PointmassMedium-v0_random.png)
- Hard environment
![hard](result_picture/PointmassHard-v0_random.png)

### Random Network Distillation Algorithm
- Easy environment
  (for plot from up to down)  
  rnd error w/o normalization  
  rnd error w/ normalization  
  rnd error normalization w/ running mean and std  
![easy](result_picture/PointmassEasy-v0_rnd1.0UNnormalization.png)
![easy](result_picture/PointmassEasy-v0_rnd1.0normalization.png)
![easy](result_picture/PointmassEasy-v0_rnd1.0ZFiliter.png)

- Medium environment
  (for plot from up to down)  
  rnd error w/o normalization  
  rnd error w/ normalization  
  rnd error normalization w/ running mean and std  
![Medium](result_picture/PointmassMedium-v0_rnd1.0UNnormalization.png)
![Medium](result_picture/PointmassMedium-v0_rnd1.0normalization.png)
![Medium](result_picture/PointmassMedium-v0_rnd1.0ZFiliter.png)

- Hard environment
  (for plot from up to down)  
  rnd error w/o normalization  
  rnd error w/ normalization  
  rnd error normalization w/ running mean and std  
![Hard](result_picture/PointmassHard-v0_rnd1.0UNnormalization.png)
![Hard](result_picture/PointmassHard-v0_rnd1.0normalization.png)
![Hard](result_picture/PointmassHard-v0_rnd1.0ZFiliter.png)


## Problem2 Offline RL
### Conversative Q learning


In view of all the eval agents reach the goal on cql alpha 0.1 in Medium environment, I choose to try different cql alpha in Hard environment to reveal the effect on cql alpha.
Specifically, I choose cql alpha in [0, 0.1, 0.2, 0.5, 1, 2, 5, 10] sequentially.

### Policy Constraint Methods: IQL and AWAC




### Data ablations



## Problem3 Online Fine-tuning