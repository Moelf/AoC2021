import numpy as np

f_name = "input2"
lines = None
with open(f_name, 'r') as f:
    lines = f.readlines()

# Action Mapping
action = {
    'forward' : [0, 1],
    'down' : [1, 1],
    'up' : [1, -1]
}

# Dead Reckoning
position = np.array([0, 0])
for l in lines:
    p = l.split(" ")
    action_key = p[0]
    movement = float(p[1])
    position[action[action_key][0]] += movement * action[action_key][1]

print(f"prod: {np.prod(position)}")

print("Part 2:")

# Change Action mapping since down and up affect aims now
action = {
    'forward' : [0, 1],
    'down' : [2, 1],
    'up' : [2, -1]
}

# Accordingly position aim
position_with_aim = np.array([0, 0, 0])
for l in lines:
    p = l.split(" ")
    action_key = p[0]
    movement = float(p[1])

    # Add to aim and position
    position_with_aim[action[action_key][0]] += movement * action[action_key][1]
    if action_key == "forward":
        position_with_aim[1] += movement * position_with_aim[2]

print(f"prod: {np.prod(position_with_aim[:-1])}")


