import numpy as np
import random
import argparse
import os

class TICalcNeuralNetwork:
    def __init__(self, learning_rate=0.001, epochs=25):
        """Initialize neural network with parameters matching the TI BASIC program"""
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Network dimensions (4 input nodes, 16 hidden nodes, 4 output nodes)
        self.input_size = 4
        self.hidden_size = 16
        self.output_size = 4
        
        # Weights and biases - matching the structure in the TI BASIC code
        # Initialize weights with values scaled similarly to TI-BASIC
        self.I = (np.random.rand(self.hidden_size, self.input_size) - 0.5) / 2  # Hidden layer weights
        self.J = (np.random.rand(self.output_size, self.hidden_size) - 0.5) / 4  # Output layer weights
        self.L4 = np.zeros(self.hidden_size)  # Hidden layer bias
        self.L5 = np.zeros(self.output_size)  # Output layer bias
        
        # Categories
        self.categories = ["DARK", "NEWT", "DEER", "DIRT"]
    
    def sigmoid(self, x):
        """Sigmoid activation function with safety checks matching TI BASIC version"""
        x = np.clip(x, -10, 10)  # Restrict to range [-10, 10] for numerical stability
        return 1 / (1 + np.exp(-x))
    
    def encode_word(self, word):
        """Convert a word into numerical input values (4 letters max)"""
        input_values = np.zeros(self.input_size)
        
        # Pad or truncate to 4 characters
        word = word.upper()[:self.input_size] # Ensure it uses self.input_size
        
        for i, char in enumerate(word):
            if 'A' <= char <= 'Z':
                # Convert letter to value between 1/26 and 26/26
                input_values[i] = (ord(char) - ord('A') + 1) / 26.0
        
        return input_values
    
    def forward(self, inputs):
        """Forward pass through the network"""
        # Hidden layer
        hidden_inputs_raw = np.dot(self.I, inputs) + self.L4
        hidden_outputs = self.sigmoid(hidden_inputs_raw)
        
        # Output layer
        final_inputs_raw = np.dot(self.J, hidden_outputs) + self.L5
        final_outputs = self.sigmoid(final_inputs_raw)
        
        return hidden_outputs, final_outputs

    def train(self, word_str, category_index, augment=True):
        """Train network on a single word, with corrected backpropagation and improved augmentation."""
        
        # Create target output (one-hot encoding)
        targets = np.zeros(self.output_size)
        targets[category_index] = 1
        
        # Data augmentation - repeat with noise and scrambling if augment is True
        repeats = 5 if augment else 1
        
        for i_repeat in range(repeats):
            current_word_str = word_str
            
            if augment:
                if i_repeat == 1 or i_repeat == 2: # Scramble for 2 iterations
                    list_word = list(word_str)
                    random.shuffle(list_word)
                    current_word_str = "".join(list_word)
                # For i_repeat == 0, use original word_str
                # For i_repeat == 3 or 4, use original word_str and add noise to encoded inputs later
            
            inputs = self.encode_word(current_word_str)

            if augment and (i_repeat == 3 or i_repeat == 4): # Add noise for last 2 augmented iterations
                # Adding noise similar to TI-BASIC's approach
                noise_application_prob = 0.5 # Chance to apply noise to a character
                for k in range(self.input_size):
                    if random.random() < noise_application_prob : # Check if char existed before noise
                         if inputs[k] > 0 or k < len(current_word_str): # only noise existing chars or their slots if word was short
                            inputs[k] = random.random() / 5.0


            # Forward pass
            hidden_outputs, final_outputs = self.forward(inputs)
            
            # --- Corrected Backpropagation ---
            
            # 1. Calculate output layer delta (error * derivative of sigmoid)
            # error_raw = final_outputs - targets
            # delta_output = error_raw * final_outputs * (1 - final_outputs)
            delta_output = (final_outputs - targets) * final_outputs * (1.0 - final_outputs)

            # 2. Calculate hidden layer delta
            error_propagated_to_hidden = np.dot(self.J.T, delta_output)
            delta_hidden = error_propagated_to_hidden * hidden_outputs * (1.0 - hidden_outputs)
            
            # 3. Update output layer weights and biases
            self.J -= self.learning_rate * np.outer(delta_output, hidden_outputs)
            self.L5 -= self.learning_rate * delta_output
            
            # 4. Update hidden layer weights and biases
            self.I -= self.learning_rate * np.outer(delta_hidden, inputs)
            self.L4 -= self.learning_rate * delta_hidden
            
    def train_model(self, verbose=True):
        """Train the model on the full dataset"""
        # Training data
        training_data = [
            ("DARK", 0),
            ("NEWT", 1),
            ("DEER", 2),
            ("DIRT", 3)
        ]
        
        if verbose:
            print(f"Starting training for {self.epochs} epochs with learning rate {self.learning_rate}")
        
        for epoch in range(self.epochs):
            # Train on each word
            random.shuffle(training_data) # Shuffle data each epoch
            for word, category in training_data:
                self.train(word, category, augment=True) # Augmentation is on by default
            
            if verbose and (epoch + 1) % (self.epochs // 10 if self.epochs >=10 else 1) == 0 : # Print progress 10 times
                print(f"Epoch {epoch + 1}/{self.epochs} completed")
        
        if verbose:
            print("Training complete!")
    
    def predict(self, word):
        """Predict the category of a word"""
        inputs = self.encode_word(word)
        _, outputs = self.forward(inputs)
        
        # Find the highest probability class
        category_index = np.argmax(outputs)
        
        # Calculate confidence as in TI-BASIC
        sum_outputs = np.sum(outputs)
        if sum_outputs == 0:
            confidence = 0
        else:
            confidence = int((outputs[category_index] / sum_outputs) * 100)
        
        return self.categories[category_index], confidence
    
    def test_accuracy(self, test_words_map=None):
        """Test the model's accuracy on test data.
        test_words_map: A dictionary like {"INPUT_WORD": "EXPECTED_CATEGORY_WORD"}
        """
        if test_words_map is None:
            # Default test set - original words and some scrambled versions
            test_words_map = {
                "DARK": "DARK", "NEWT": "NEWT", "DEER": "DEER", "DIRT": "DIRT",
                "KARD": "DARK", "WTNE": "NEWT", "REED": "DEER", "TRID": "DIRT", # Simple scrambles
                "ADKR": "DARK", "NTWE": "NEWT", "EDRE": "DEER", "ITDR": "DIRT", # More scrambles
                "BARK": "DARK", # Similar words (optional, may misclassify to trained categories)
                "TEWN": "NEWT",
                "DRK": "DARK", # Shorter input
                "NWT": "NEWT"
            }
        
        correct_predictions = 0
        total_testable_words = len(test_words_map)
        results = []
        
        print("\nTesting model accuracy:")
        for word, expected_category_word in test_words_map.items():
            predicted_category_word, confidence = self.predict(word)
            
            is_match = (predicted_category_word == expected_category_word)
            if is_match:
                correct_predictions += 1
            
            result = f"Input: '{word}' → Predicted: '{predicted_category_word}' (Confidence: {confidence}%) - Expected: '{expected_category_word}' [{ 'MATCH' if is_match else 'MISS' }]"
            results.append(result)
            print(result)
        
        if total_testable_words > 0:
            accuracy = (correct_predictions / total_testable_words) * 100
            print(f"\nAccuracy on this test set: {accuracy:.2f}% ({correct_predictions}/{total_testable_words})")
        else:
            print("\nNo testable words provided for accuracy calculation.")
        
        return results

    def generate_ti_basic_lists(self):
        """Generate TI BASIC code to load trained weights (without comments)"""
        ti_basic_code = f"""
Lbl P
ClrHome
Output(1,1,"LOADING PRETRAINED")
Output(2,1,"WEIGHTS...")
{{16,4}}→dim([I])
{{4,16}}→dim([J])
{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}→L₄
{{0,0,0,0}}→L₅
"""
        
        # Add code to set each element of I
        ti_basic_code += "\n" # Removed comment
        for row in range(self.hidden_size):
            for col in range(self.input_size):
                # Use '⁻' for negative numbers for TI-BASIC literal
                val_str = f"{self.I[row, col]:.6f}"
                if self.I[row, col] < 0:
                    val_str = "⁻" + val_str[1:]
                ti_basic_code += f"{val_str}→[I]({row+1},{col+1})\n"
        
        # Add code to set each element of J
        ti_basic_code += "\n" # Removed comment
        for row in range(self.output_size):
            for col in range(self.hidden_size):
                val_str = f"{self.J[row, col]:.6f}"
                if self.J[row, col] < 0:
                    val_str = "⁻" + val_str[1:]
                ti_basic_code += f"{val_str}→[J]({row+1},{col+1})\n"
        
        # Add code to set L4 and L5
        ti_basic_code += "\n" # Removed comment
        for i, val in enumerate(self.L4):
            val_str = f"{val:.6f}"
            if val < 0:
                val_str = "⁻" + val_str[1:]
            ti_basic_code += f"{val_str}→L₄({i+1})\n"
        
        ti_basic_code += "\n" # Removed comment
        for i, val in enumerate(self.L5):
            val_str = f"{val:.6f}"
            if val < 0:
                val_str = "⁻" + val_str[1:]
            ti_basic_code += f"{val_str}→L₅({i+1})\n"
        
        ti_basic_code += """
Output(3,1,"LOAD COMPLETE")
Pause 
Goto A
"""
        
        return ti_basic_code

    def save_weights_to_file(self, filename="nn_weights.txt"):
            """Save weights to a text file (in UTF-8)."""
            # filename is expected to be a valid string due to argparse default
            
            parent_dir = os.path.dirname(filename)
            
            # If parent_dir is not an empty string, it means filename includes a path.
            # Create the parent directory if it doesn't exist.
            if parent_dir: 
                os.makedirs(parent_dir, exist_ok=True)
            # If parent_dir is an empty string, the file is in the current directory.
            # The current directory is assumed to exist, so no os.makedirs call is needed for it.
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.generate_ti_basic_lists())
            print(f"Weights saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Train a neural network and export weights for TI calculator.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.04)')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs (default: 20000)')
    parser.add_argument('--output', type=str, default="nn_weights.txt", help='Output file for weights (default: nn_weights.txt)')
    parser.add_argument('--test', action='store_true', help='Run tests after training')
    parser.add_argument('--no-verbose', action='store_true', help='Suppress verbose output during training')
    
    args = parser.parse_args()
    
    # Initialize and train the network
    nn = TICalcNeuralNetwork(learning_rate=args.lr, epochs=args.epochs)
    nn.train_model(verbose=not args.no_verbose)
    
    # Save the weights
    nn.save_weights_to_file(args.output)
    
    # Test if requested
    if args.test:
        nn.test_accuracy() # Uses the default test set which includes some scrambled words

if __name__ == "__main__":
    main()