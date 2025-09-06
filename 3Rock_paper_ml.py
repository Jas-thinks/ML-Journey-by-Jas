import random
from sklearn.linear_model import LogisticRegression

# Possible moves
moves = ["rock", "paper", "scissors"]

# Data storage for training
X, y = [], []
model = LogisticRegression()

print("Let's play Rock-Paper-Scissors! (type 'quit' to stop)")

while True:
    user = input("Your move (rock/paper/scissors): ").lower()
    if user == "quit":
        break
    if user not in moves:
        print("Invalid move. Try again!")
        continue
    
    # NOTE for learners:
    # 👉 In ML, models usually need MANY samples to work well.
    # But here, we are training with VERY FEW moves.
    # This shows that ML can even "adapt on the fly" with small data,
    # which is similar to how **online learning algorithms** work.
    
    if len(X) > 3:
        model.fit(X, y)  # trains every round with all past moves
        
        # 👉 We predict based only on the LAST move.
        # This is a very naive approach, but surprisingly,
        # humans often repeat or switch in patterns.
        # Fun fact: Real RPS AI bots use **Markov Chains** to model this!
        pred = model.predict([X[-1]])[0]
        
        # Computer tries to COUNTER your predicted move
        comp = {"rock": "paper", "paper": "scissors", "scissors": "rock"}[pred]
    else:
        comp = random.choice(moves)  # random start when no training yet
    
    print(f"Computer chose: {comp}")
    
    # Decide winner
    if user == comp:
        print("It's a tie!")
    elif (user == "rock" and comp == "scissors") or \
         (user == "paper" and comp == "rock") or \
         (user == "scissors" and comp == "paper"):
        print("You win!")
    else:
        print("Computer wins!")

    # 👉 We are encoding moves as numbers (0,1,2)
    # Because ML models don't understand words directly.
    # This is called **Feature Encoding** (categorical → numeric).
    X.append([moves.index(user)])
    y.append(user)  # labels remain as text, since sklearn handles them

