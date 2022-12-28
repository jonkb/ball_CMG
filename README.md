# Dynamic simulation of a ball with a CMG mounted inside
This program simulates the dynamics of a theoretical robot composed of a ball with a Control Moment Gyroscope (CMG) mounted inside.

I'm in the process of documenting the methods here: https://www.overleaf.com/read/pcmrhbxdtwrf

The strategy behind the dynamics was mostly adapted from the following paper by Putkaradze & Rogers:
https://link.springer.com/article/10.1007/s11012-018-0904-5#Sec11

# Capabilities
1. Derive the EOM
2. Simulate the path of the robot
3. Plot simulation results
4. Optimize the input angle (alpha) to attempt to trace a given path

# Dependencies
See /requirements.txt

# To Do
* Display simulation parameters after loading from file (especially MPCparams)
* Fix the relative weighting of position and velocity error in MPC cost

