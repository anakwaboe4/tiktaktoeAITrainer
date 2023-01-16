from tensorflow.python.ops.math_ops import truediv
import os
import itertools
import concurrent.futures
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.optimizer.set_jit(True)
import pickle
import numpy as np
import concurrent
import numpy.random as npr
from tensorflow.keras.layers import Dense


import random
population_size = 10
# Set the number of self-play games to run per neural network
num_games = 10

def select_best(population, scores):
  # Zip the population and scores lists together
  population_with_scores = zip(population, scores)

  # Sort the population based on their scores
  sorted_population = sorted(population_with_scores, key=lambda x: x[1], reverse=True)

  # Return the best neural network
  return sorted_population[0][0]

def mutate(nn):
  for layer in nn.layers:
    if isinstance(layer, Dense):
        weights = layer.get_weights()
        mutated_weights = []
        for weight in weights:
            if random.uniform(0, 1) < 0.1:
                mutated_weight = weight + random.uniform(-0.5, 0.5)
                mutated_weights.append(mutated_weight)
            else:
                mutated_weights.append(weight)
        layer.set_weights(mutated_weights)
  return nn

def crossover(parent1, parent2):
  # Choose a crossover point at random
  crossover_point = random.randint(1, len(parent1.get_weights()))

  # Create the offspring by combining the weights of the parents at the crossover point
  offspring1_weights = parent1.get_weights()[:crossover_point] + parent2.get_weights()[crossover_point:]
  offspring2_weights = parent2.get_weights()[:crossover_point] + parent1.get_weights()[crossover_point:]

  # Create the offspring neural networks
  offspring1 = initialize_neural_network()
  offspring1.set_weights(offspring1_weights)
  offspring2 = initialize_neural_network()
  offspring2.set_weights(offspring2_weights)

  return offspring1, offspring2

def breed(parent1, parent2):
  # Perform crossover to create two offspring
  offspring1, offspring2 = crossover(parent1, parent2)

  # Mutate the offspring with a small probability
  if random.uniform(0, 1) < 0.1:
    offspring1 = mutate(offspring1)
  if random.uniform(0, 1) < 0.1:
    offspring2 = mutate(offspring2)

  return offspring1, offspring2

def normalize(scores):
        # Calculate the sum of the scores
        newscore= []
        for score in scores: 
          if score > 0: newscore.append(score + 1.5 *num_games)
          else: newscore.append(score + num_games)
        total = sum(newscore)
        if total == 0: 
          total = 1
          if newscore.count(0) == len(newscore):
            temp = []
            for score in newscore: temp.append(1/len(newscore))
            newscore=temp
        # Normalize the scores
        return [(score)  / total for score in newscore]


def breed_population(population, population_size,scores):
  next_generation = []
  normalized_scores = normalize(scores)
  while len(next_generation) < population_size:
    # Select a survivor using roulette wheel selection
    survivor = select_survivor_roulette(population, normalized_scores)
    # Apply mutations to the survivor to create the next generation
    if random.random() < 0.2:
      mutated_nn = mutate(survivor)
      next_generation.append(mutated_nn)
    else:
      next_generation.append(survivor)


  return next_generation

def select_survivor_roulette(population, scores):
  # Normalize the scores
  return population[npr.choice(len(population), p=scores)]

def calculate_score(board):
  # Check if 'X' has won
  if check_winner(board) == 'X':
    return 1

  # Check if 'O' has won
  if check_winner(board) == 'O':
    return -1

  # The game is a draw
  return 0

def get_available_moves(board):
  moves = []
  for i in range(3):
    for j in range(3):
      if board[i][j] == ' ':
        moves.append(i * 3 + j)
  return moves

def choose_random_move(board):
  # Get a list of all the available moves
  moves = get_available_moves(board)

  # Choose a random move from the list
  #move = random.choice(moves)
  move=moves[0]
  return move

def make_move(board, move, player):
  if move < 0 or move > 8:
    raise ValueError("Invalid move")
  row = move // 3
  col = move % 3
  board[row][col] = player
  #print(f"Making move: {move}")
  #print(f"Board after move: {board}")

def create_input_data(board):
  # Create an input with shape (None, 9)
  input_data = np.zeros((1, 9))

  # Set the input data for each cell
  for i in range(3):
    for j in range(3):
      if board[i][j] == "X":
        input_data[0][i * 3 + j] = 1
      elif board[i][j] == "O":
        input_data[0][i * 3 + j] = -1

  # Return the input data
  return input_data
def create_input_data_opp(board):
  # Create an input with shape (None, 9)
  input_data = np.zeros((1, 9))

  # Set the input data for each cell
  for i in range(3):
    for j in range(3):
      if board[i][j] == "X":
        input_data[0][i * 3 + j] = -1
      elif board[i][j] == "O":
        input_data[0][i * 3 + j] = 1

  # Return the input data
  return input_data
def check_move(board, move):
  if move < 0 or move > 8:
    return False
  row = move // 3
  col = move % 3
  if board[row][col] != "O" and board[row][col] != "X": return True
  else: return False

def choose_move(board, nn):
    input_data = create_input_data(board)
    prediction = nn[0].predict(input_data,verbose=0)
    print(prediction)
    best_moves = sorted(range(len(prediction)), key=lambda i: prediction[i])[-9:]
    for i in best_moves:
        if check_move(board, i):
            return i
    raise ValueError("No valid move found")

def choose_move_opp(board, nn):
    input_data = create_input_data_opp(board)
    prediction = nn.predict(input_data,verbose=0)[0]
    best_moves = sorted(range(len(prediction)), key=lambda i: prediction[i])[-9:]
    for i in best_moves:
        if check_move(board, i):
            return i
    raise ValueError("No valid move found")


def check_winner(board):
  # Check rows
  for row in board:
    if row[0] != ' ' and row[0] == row[1] and row[1] == row[2]:
      return row[0]

  # Check columns
  for col in range(3):
    if board[0][col] != ' ' and board[0][col] == board[1][col] and board[1][col] == board[2][col]:
      return board[0][col]

  # Check diagonals
  if board[0][0] != ' ' and board[0][0] == board[1][1] and board[1][1] == board[2][2]:
    return board[0][0]
  if board[0][2] != ' ' and board[0][2] == board[1][1] and board[1][1] == board[2][0]:
    return board[0][2]

  # No winner
  return None

def initialize_neural_network():
  # Define the model
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, input_shape=(9,), activation='relu'),
      tf.keras.layers.Dense(9, activation='softmax')
  ])
  

  # Compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model

def initialize_population(size):
  population = []
  for i in range(size):
    nn = initialize_neural_network()
    population.append(nn)
  return population

# Save the population of neural networks to a file
def save_population(population, filename):
  with open(filename, 'wb') as f:
    pickle.dump(population, f)

# Load the population of neural networks from a file
def load_population(filename):
  with open(filename, 'rb') as f:
    population = pickle.load(f)
  return population

# Export the best neural network
def export_best(best_nn, filename):
  best_nn.save(filename)

# Load the best neural network
def import_best(filename):
  best_nn = tf.keras.models.load_model(filename)
  return best_nn

def initialize_board():
  board = [
      [' ', ' ', ' '],
      [' ', ' ', ' '],
      [' ', ' ', ' ']
  ]
  return board

def game_not_over(board):
  # Check if there is a winner
  if check_winner(board) == 'X':
    return False
  if check_winner(board) == 'O':
    return False

  # Check if the board is full
  for row in board:
    for element in row:
      if element == ' ':
        return True

  # If the board is full and there is no winner, the game is a draw
  return False

def evaluate_nn(nn, num_games):
  score = 0
  for j in range(num_games):
    # Initialize the board
    board = initialize_board()

    # Play the game until it is over
    while game_not_over(board):
      # Choose the next move using the neural network
      move = choose_move(board, nn)
      # Make the move
      make_move(board, move, "X")

      # If the game is not over, have the opponent make a random move
      if game_not_over(board):
        opponent_move = choose_random_move(board)
        make_move(board, opponent_move, "O")

    # The game is over, so update the score based on the result of the game
    score += calculate_score(board)
  return score

def evaluate_population_rando(population, num_games, use_threading=False):
  if use_threading:
    # Get the number of CPU cores
    num_cores = os.cpu_count()

    # Split the population into num_cores parts
    population_parts = [population[i::num_cores] for i in range(num_cores)]
  
    # Create a list of tasks to evaluate each part of the population
    with concurrent.futures.ThreadPoolExecutor() as executor:
      tasks = []
      for part in population_parts:
          task = executor.submit(evaluate_nn, part, num_games)
          tasks.append(task)

      scores = []
      for task in concurrent.futures.as_completed(tasks):
          scores.append(task.result())

    scores = list(itertools.chain.from_iterable(scores))
  else:
    scores = []
    for nn in population:
      scores.append(evaluate_nn(nn, num_games))
  return scores
def evaluate_population(population, use_threading=True):
    if use_threading:
        # Get the number of CPU cores
        num_cores = os.cpu_count()

        # Split the population into num_cores parts
        population_parts = [population[i::num_cores] for i in range(num_cores)]

        # Create a list of tasks to evaluate each part of the population
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = []
            for part in population_parts:
                task = executor.submit(evaluate_nn_multithread, part, population)
                tasks.append(task)

            scores = []
            for task in concurrent.futures.as_completed(tasks):
                scores.append(task.result())

        scores = list(itertools.chain.from_iterable(scores))
    else:
        scores = self_play_population(population)
    return scores

def evaluate_nn_multithread(nn, population):
    nn_score = 0
    for opp in population:
        if nn != opp:
            result = play_game(nn, opp)
            if result == 1:
                nn_score += 1
            elif result == -1:
                nn_score -= 1
    return nn_score

def play_game(nn, opp):
  score = 0
  
  # Initialize the board
  board = initialize_board()

  # Play the game until it is over
  while game_not_over(board):
    # Choose the next move using the neural network
    move = choose_move(board, nn)
    # Make the move
    make_move(board, move, "X")
    # If the game is not over, have the opponent make a random move
    if game_not_over(board):
      opponent_move = choose_move_opp(board, opp)
      make_move(board, opponent_move, "O")
  # The game is over, so update the score based on the result of the game
  score += calculate_score(board)
  return score   

  
# Initialize a population of neural networks
population = initialize_population(size=population_size)

# Set the target win rate
target_win_rate = 0.9

# Set the generation counter
generation = 0

while True:
  # Evaluate the performance of each neural network by having it play num_games self-play games
  scores = []
  printscores= []
  scores = evaluate_population_rando(population, num_games, use_threading=True)
  #scores = evaluate_population(population, use_threading=False )
  average_score = sum(scores) / len(scores)
  print(f"Average score: {average_score}")
  print(f"Scores: {scores}")

  # Save the population to a file
  save_population(population, f"generation{generation}test.txt")
    # Save the best AI to a file
  best_nn = select_best(population, scores)
  best_nn.save(f"best_ai_generation{generation}test.h5")

  # Calculate the win rate
  win_rate = sum(scores) / len(scores) / num_games

  # If the win rate is above the target, stop training
  if win_rate >= target_win_rate:
    break

  # Select the parents using roulette wheel selection
  # Breed the next generation of neural networks
  next_generation = breed_population(population, population_size,scores)

  # Replace the current population with the next generation
  population = next_generation

  # Increment the generation counter
  generation += 1

# The training is complete, so choose the best neural network
best_nn = select_best(population, scores)
best_nn.save("best_ai.h5")
