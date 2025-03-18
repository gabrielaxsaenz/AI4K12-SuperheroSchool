import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Convert choices to numerical values
def choice_to_num(choice):
    return {'A': 0, 'B': 1, 'C': 2, 'D': 3}[choice.upper()]

# Convert numerical values back to choices
def num_to_choice(num):
    return {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[np.argmax(num)]

# Collect user choices
choices = []
for i in range(10):
    choice = input(f"Enter choice {i+1} (A, B, C, or D): ").strip().upper()
    while choice not in ['A', 'B', 'C', 'D']:
        print("Invalid input! Please enter 'A', 'B', 'C', or 'D'.")
        choice = input(f"Enter choice {i+1} (A, B, C, or D): ").strip().upper()
    choices.append(choice_to_num(choice))

# Prepare data for training
X = np.array([choices[:-1]])  # Input: First 9 choices
y = np.array([choices[-1]])   # Output: 10th choice

# One-hot encode the data
encoder = OneHotEncoder(sparse=False, categories=[range(4)])
X_encoded = encoder.fit_transform(X)
y_encoded = encoder.transform(y.reshape(-1, 1))

# Build a simple neural network
model = Sequential([
    Dense(16, input_shape=(4 * 9,), activation='relu'),  # Hidden layer
    Dense(4, activation='softmax')  # Output layer (4 choices)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_encoded, y_encoded, epochs=50, verbose=0)

# Predict the next choice
next_input = np.array([choices[1:]])  # Use choices 2-10 as input
next_input_encoded = encoder.transform(next_input)
predicted_choice_encoded = model.predict(next_input_encoded)
predicted_choice = num_to_choice(predicted_choice_encoded)

print(f"\nBased on your previous choices, the neural network predicts your next choice will be: {predicted_choice}")