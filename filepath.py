import os
import sys

AIRFOILS_PATH = "/home/t-isaacju/data/01_cindm/dataset/airfoils_dataset/"
sys.path.append(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.append(os.path.join(os.path.dirname("__file__"), "..", ".."))

NBODY_PATH = "dataset/nbody_dataset/"
pos = "current_wp"
current_wp = os.getcwd()
if pos == "snap":
    EXP_PATH = "/dfs/project/plasma/results/"
elif pos == "current_wp":
    EXP_PATH = current_wp + "/results/"
else:
    EXP_PATH = "./results/"
