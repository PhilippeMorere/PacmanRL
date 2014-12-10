PacmanRL
========

Reinforcement learning algorithms applied to the Pacman game.

The Pacman game is the result of [Pushkar's work](https://github.com/pushkar/ud820-proj) for the Udacity class ud820.

### Run
```
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```
The `-l` parameter can be changed to use other maps (available in the layout folder).
The `-p` parameter can be changed to use other types of q-learners.

### Learners
* PacmanQAgent: basic Q-learning agent. It uses full states (the whole map), and thus doesn't scale when using big maps.
* ApproximateQAgent: Uses a few predifined features to describe a state. What is learnt is how to combine these features to take a decision. This one can be used on big maps. [under development]
* PerceptronQAgent: Learns using a perceptrion. The input is what is in the cells near-by. [under development]
