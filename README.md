PacmanRL
========

Reinforcement learning algorithms applied to the Pacman game.

The Pacman game is the result of [Pushkar's work](https://github.com/pushkar/ud820-proj) for the Udacity class ud820.

The learners can be found in the `qlearningAgents.py` file.

### Run
```
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```
* The `-l` parameter can be changed to use other maps (available in the layout folder).
* The `-p` parameter can be changed to use other types of q-learners.
* `-x` sets the number of training examples.
* `-n` sets the total number of examples (train +  test)


### Learners
* PacmanQAgent: basic Q-learning agent. It uses full states (the whole map), and thus doesn't scale when using big maps.
Run it:
```
python pacman.py -p PacmanQAgent -x 800 -n 810 -l smallGrid
```
* ApproximateQAgent: Uses a few predifined features to describe a state. What is learnt is how to combine these features to take a decision. This one can be used on big maps. The main drawback of this method is that Pacman is almost blind, as it only know what's happening in his 4 adjacent cells. It can easily get stuck in a corner because it doesn't know where the food is.
Run it:
```
python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid
```
* PerceptronQAgent: Learns using a perceptron. The input is what is in the cells near-by. Unlike with the previous learner, Pacman has access to a range of cells around him (defined via parameter). [under development]
Run it:
```
python pacman.py -p PerceptionQAgent -x 2000 -n 2010 -d
```
