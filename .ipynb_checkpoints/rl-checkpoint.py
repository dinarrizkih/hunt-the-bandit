import turtle
import numpy as np
import yaml
import random
import time

# Setup turtle screen
screen = turtle.Screen()
screen.title("Tom & Jerry - Reinforcement Learning")
screen.setup(800, 800)
screen.addshape("jerry.gif")
screen.addshape("tom.gif")

# Setup score display
score_turtle = turtle.Turtle()
score_turtle.penup()
score_turtle.goto(-100, 325)
score_turtle.hideturtle()
score = 0
prey_score = 0
score_turtle.write(f"Tom's Score: {score}  Jerry's score: {prey_score}", font=("Arial", 10, "bold"))

# Create Jerry's turtle
target_turtle = turtle.Turtle()
target_turtle.speed(1000)
target_turtle.shape("jerry.gif")
target_turtle.penup()
target_turtle.goto(-325, -325)

target_turtle.pendown()
target_turtle.goto(-325, 325)
target_turtle.goto(325, 325)
target_turtle.goto(325, -325)
target_turtle.goto(-325, -325)

target_turtle.penup()
target_turtle.goto(300, 300)

# Create Tom's turtle
agent_turtle = turtle.Turtle()
agent_turtle.shape("tom.gif")
agent_turtle.speed(1000)
agent_turtle.penup()
agent_turtle.goto(-300, -300)

# Create obstacles
obstacles = [(0, 0), (50, 0), (100, 0), (0, 50), (0, 100), (-100, 0), (-50, -50), (-100, 250), (250, -200), 
             (200, -200), (250, -150), (250, -50), (250, 0), (250, 100), (250, 250), (200, 250), (150, 250), 
             (-200, -250), (-150, -250), (-200, -150), (-200, -100), (-250, -50), (-250, 0), (-250, 50), 
             (-250, 150), (-250, 250)]

for obstacle in obstacles:
    obstacle_turtle = turtle.Turtle()
    obstacle_turtle.speed(1000)
    obstacle_turtle.shape("square")
    obstacle_turtle.shapesize(2.5)
    obstacle_turtle.color("red")
    obstacle_turtle.penup()
    obstacle_turtle.goto(obstacle)

# Define states
states = [(x, y) for x in range(-300, 301, 50) for y in range(-300, 301, 50)]

# Define goal states (states without obstacles)
goal_states = [state for state in states if state not in obstacles]

# Define actions
actions = ['up', 'down', 'left', 'right']

# Initialize Q-tables
predator_table = {}
prey_table = {}

# Load Q-tables from file if available
try:
    with open('condition.yaml', 'r') as f:
        predator_table = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    predator_table = {}

try:
    with open('prey.yaml', 'r') as f:
        prey_table = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    prey_table = {}

# Define hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Q-Learning algorithm
predator_state = (-300, -300)  # Start from the initial state
prey_state = (300, 300)

for s in goal_states:
    for g_state in goal_states:
        min_duration = 0
        min_duration_time = 0

        while min_duration_time != 1000:
            randomm = True
            show = True

            if not randomm:
                predator_state = s
                prey_state = g_state

            if show:
                agent_turtle.goto(predator_state)
                score_turtle.clear()
                score_turtle.write(f"Tom's Score: {score}  Jerry's Score: {prey_score}", font=("Arial", 10, "bold"))
                agent_turtle.speed(10)
                target_turtle.speed(10)
                target_turtle.goto(prey_state)

            done = False
            start = time.time()
            step = 0

            while not done:
                step += 1

                # Train Prey
                x, y = predator_state
                goal_x, goal_y = prey_state
                condition = (x, y, goal_x, goal_y)
                if condition not in prey_table:
                    prey_table[condition] = {action: 0 for action in actions}

                # Choose an action using epsilon-greedy policy
                if np.random.uniform() < epsilon:
                    action = np.random.choice(actions)  # Explore
                else:
                    action = max(prey_table[condition], key=prey_table[condition].get)  # Exploit

                # Get the next state
                if action == 'up':
                    next_state = (goal_x, goal_y + 50)
                elif action == 'down':
                    next_state = (goal_x, goal_y - 50)
                elif action == 'left':
                    next_state = (goal_x - 50, goal_y)
                else:  # 'right'
                    next_state = (goal_x + 50, goal_y)

                # Get the reward
                if next_state not in states:
                    next_state = prey_state
                    reward = -5  # Penalty for hitting the wall
                elif ((next_state[0] - predator_state[0]) ** 2 + (next_state[1] - predator_state[1]) ** 2) <= 5000:
                    reward = -10  # Reached the target turtle
                elif next_state in obstacles:
                    next_state = prey_state
                    reward = -5  # Hit the obstacle turtle
                else:
                    current_distance = ((prey_state[0] - predator_state[0]) ** 2 + (prey_state[1] - predator_state[1]) ** 2) ** (1 / 2)
                    next_distance = ((next_state[0] - predator_state[0]) ** 2 + (next_state[1] - predator_state[1]) ** 2) ** (1 / 2)
                    reward = (next_distance - current_distance) / 10

                x, y = predator_state
                goal_x, goal_y = next_state
                next_condition = (x, y, goal_x, goal_y)
                if next_condition not in prey_table:
                    prey_table[next_condition] = {action: 0 for action in actions}

                # Update the Q-table
                prey_table[condition][action] += alpha * (
                        reward + gamma * max(prey_table[next_condition].values()) -
                        prey_table[condition][action])

                # Update the state and move the agent turtle
                prey_state = next_state
                if show:
                    target_turtle.goto(prey_state)

                # Train Predator
                x, y = predator_state
                goal_x, goal_y = prey_state
                condition = (x, y, goal_x, goal_y)
                if condition not in predator_table:
                    predator_table[condition] = {action: 0 for action in actions}

                # Choose an action using epsilon-greedy policy
                if np.random.uniform() < epsilon:
                    action = np.random.choice(actions)  # Explore
                else:
                    action = max(predator_table[condition], key=predator_table[condition].get)  # Exploit

                # Get the next state
                if action == 'up':
                    next_state = (x, y + 50)
                elif action == 'down':
                    next_state = (x, y - 50)
                elif action == 'left':
                    next_state = (x - 50, y)
                else:  # 'right'
                    next_state = (x + 50, y)

                # Get the reward
                if next_state not in states:
                    next_state = predator_state
                    reward = -5  # Penalty for hitting the wall
                elif next_state == prey_state:
                    reward = 10  # Reached the target turtle
                elif next_state in obstacles:
                    next_state = predator_state
                    reward = -5  # Hit the obstacle turtle
                else:
                    reward = -1

                x, y = next_state
                goal_x, goal_y = prey_state
                next_condition = (x, y, goal_x, goal_y)
                if next_condition not in predator_table:
                    predator_table[next_condition] = {action: 0 for action in actions}

                # Update the Q-table
                predator_table[condition][action] += alpha * (
                        reward + gamma * max(predator_table[next_condition].values()) -
                        predator_table[condition][action])

                # Update the state and move the agent turtle
                predator_state = next_state
                if show:
                    agent_turtle.goto(predator_state)

                if step == 70:
                    done = True
                    if randomm:
                        prey_state = random.choice(goal_states)
                    if show:
                        prey_score += 1
                        target_turtle.hideturtle()
                        target_turtle.goto(prey_state)
                        target_turtle.showturtle()

                if ((prey_state[0] - predator_state[0]) ** 2 + (prey_state[1] - predator_state[1]) ** 2) <= 5000:
                    done = True
                    if randomm:
                        prey_state = random.choice(goal_states)
                    if show:
                        score += 1
                        target_turtle.hideturtle()
                        target_turtle.goto(prey_state)
                        target_turtle.showturtle()

            duration = time.time() - start
            if duration > min_duration:
                min_duration = duration
                min_duration_time = 0
            else:
                min_duration_time += 1

            # Save Q-tables periodically
            if min_duration_time % 100 == 0:
                with open('condition.yaml', 'w') as f:
                    yaml.dump(predator_table, f)
                with open('prey.yaml', 'w') as f:
                    yaml.dump(prey_table, f)

# Save Q-tables at the end
with open('condition.yaml', 'w') as f:
    yaml.dump(predator_table, f)
with open('prey.yaml', 'w') as f:
    yaml.dump(prey_table, f)

# Keep the screen open until it's closed manually
turtle.done()
