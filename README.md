# Hermes Neural Messenger: TI-84 Word Classifier & PC Trainer

## Project Overview

Welcome to **Project Hermes**, a system designed to bring the power of neural network word classification to your TI-84 Plus series graphing calculator. Like the swift messenger of Olympus, Hermes aims to deliver quick interpretations (classifications) of 4-letter words directly on your handheld device.

The "messages" Hermes delivers – the trained intelligence of the neural network – can be crafted with greater power and precision on a PC using the companion Python script. This script allows for robust training and then packages the learned knowledge into a format the TI-84 version of Hermes can understand and utilize.

**Core Categories Hermes Interprets:** The network is trained to sort words into one of four categories:
* DARK
* NEWT
* DEER
* DIRT

## Features

### Hermes on the TI-84 (`NEURALNET_CLASSIFIER_V10.0.8xp`)

* **Swift On-Calculator Word Classification:** Input a 4-letter word, and Hermes delivers its predicted category and a confidence score, right on your calculator.
* **On-Calculator Training (Quick Initiation):** A basic training routine allows Hermes to quickly learn directly on the calculator (100 epochs, learning rate 0.04). This is for rapid familiarization, but messages from the PC are more refined.
* **Receive "Messages" (Load Pre-trained Models):** Hermes can load sophisticated training messages (weights and biases) prepared on a PC. A default message is included for immediate use.
* **Intuitive Guidance (Menu System):** User-friendly menus guide you through training Hermes, using its classification abilities, or loading new knowledge.
* **Hermes's Mind (Neural Network Architecture):** A 4-16-4 feedforward neural network:
    * 4 input neurons (to receive the 4 characters of a word).
    * 16 neurons in the hidden layer (where interpretation happens).
    * 4 output neurons (delivering the final category).
* **Activation Function (The Spark of Thought):** Sigmoid activation function ($f(x) = \frac{1}{(1 + e^{-x})}$), with output clamping for stable thought processes on the calculator.
* **Adaptive Learning (Data Augmentation):** Basic noise-based data augmentation is applied during on-calculator training, helping Hermes generalize.
* **Display:** Clear, text-based communication of menus, progress, and results.

*(Note: The program name on the calculator might be truncated (e.g., to `HERMES` or `NNCLAS10`). The internal display name "NEURALNET CLASSIFIER V10.0" will still be shown by Hermes itself.)*

### Hermes's Training Scribe (Python Script: `python_trainer.py`)

* **Crafting Messages on Olympus (PC-Based Model Training):** Train Hermes's neural network on your computer, allowing for more extensive training epochs and fine-tuned learning rates, resulting in more insightful classification "messages."
* **Configurable Training Parameters:** Adjust the learning rate and number of epochs via command-line arguments to refine the training message.
* **TI-BASIC "Scroll" Generation:** Automatically generates the TI-BASIC code (a "scroll" of knowledge) needed to update the weights (`[I]`, `[J]`) and biases (`L₄`, `L₅`) in Hermes's mind on the calculator.
* **Enhanced Learning Techniques (Advanced Data Augmentation):** Employs word scrambling and noise injection during PC training, creating more robust and adaptable knowledge for Hermes.
* **Testing the Message's Clarity (Accuracy Testing):** Includes a function to test the accuracy of the trained model (the "message") on a predefined test set.
* **Preserving the Scroll (Save/Load Weights):** Saves the generated TI-BASIC code to a text file.

## Why Two Scripts? The Tale of Two Realms

* **Calculator Realm (The Mortal Plane):** Training complex neural networks is a Herculean task for the TI-84, a realm of immediacy but limited computational power. Extensive training here would be a slow odyssey.
* **PC Realm (Olympus):** A PC possesses the Olympian power needed to train the network for tens of thousands of epochs, crafting highly refined "messages" (models) with speed and precision.
* **Hermes, The Bridge:** This system allows the intensive message crafting (training) to occur in the PC realm, while the swift delivery and application of that knowledge happen portably on the TI-84, with Hermes acting as the bridge.

## System Requirements

### For Hermes on the TI-84:

* **Calculator:** A TI-84 Plus CE, TI-84 Plus C Silver Edition, or a compatible model from the TI-84 Plus series. While the core logic might run on older TI-83/84 models supporting matrices and list operations, display formatting and some commands are optimized for newer versions.
* **Transfer Software (The Chariot):** TI Connect™ CE software (or an alternative like TiLP) to convey the `.8xp` program file (Hermes himself) to your calculator.
* **Calculator Memory:** Sufficient free RAM and Archive memory for Hermes and his associated knowledge matrices/lists.

### For Hermes's Training Scribe (Python):

* **Python:** Python 3.x (e.g., Python 3.7 or newer).
* **Libraries:**
    * NumPy: For numerical operations, the bedrock of neural calculations. Install using pip:
        ```bash
        pip install numpy
        ```

## Setup and Installation

### Bringing Hermes to your TI-84:

1.  **Download:** Obtain the `NEURALNET_CLASSIFIER_V10.0.8xp` program file.
2.  **Transfer:** Use TI Connect™ CE (or similar software) to send the `.8xp` file from your computer to your calculator's RAM or Archive memory.

### Preparing Hermes's Training Scribe on PC:

1.  **Download:** Obtain the `python_trainer.py` script file.
2.  **Install Python:** If you don't have Python 3 installed, download and install it from [python.org](https://www.python.org/).
3.  **Install NumPy:** Open your terminal or command prompt and run:
    ```bash
    pip install numpy
    ```

## Interacting with Hermes

### Hermes on the TI-84 (`NEURALNET CLASSIFIER V10.0`):

1.  **Summon Hermes:** Press the `[prgm]` key on your calculator, select the program (e.g., `HERMES`, `NNCLAS10`, or the full name if it appears), and press `[enter]`.
2.  **Hermes's Main Menu:**
    * `TRAIN NEW MODEL`: Initiates a brief training session directly on the calculator, allowing Hermes to learn the basics. Weights and biases will be randomly initialized and then updated.
    * `SKIP`: Bypasses on-calculator training and proceeds to the classification menu. If no model has been trained or "message" loaded, Hermes will use randomly initialized (likely unhelpful) knowledge.
    * `EXIT`: Dismisses Hermes.
3.  **Hermes's Classifier Menu (appears after `SKIP` or `TRAIN NEW MODEL`):**
    * `CLASSIFY`:
        * Hermes will ask for a "WORD? max 4 chars".
        * Enter a 4-letter word (uppercase).
        * Hermes delivers the predicted category ("DARK", "NEWT", "DEER", or "DIRT") and a confidence percentage.
    * `LOAD PRETRAIN` (Receive a Refined Message):
        * Loads a set of pre-defined weights and biases into Hermes's network.
        * The script initially contains a default "message."
        * **This option is crucial for bestowing Hermes with the advanced knowledge crafted by the Python script (see "Workflow" below).**
    * `ABOUT`: Hermes shares information about his current version and capabilities:
        * "Version 10.0"
        * "- Feedforward neural net"
        * "- 4-16-4 arχtecture" (Note: 'arχtecture' uses a Greek Chi symbol as per the original script code.)
        * "- Data augmentation"
    * `EXIT`: Dismisses Hermes.

### Hermes's Training Scribe (`python_trainer.py`):

1.  **Open Terminal:** Launch your command prompt (Windows) or terminal (macOS/Linux).
2.  **Navigate:** Change directory to where you saved `python_trainer.py`.
    ```bash
    cd path/to/your/script
    ```
3.  **Instruct the Scribe:** Execute the script with optional arguments:
    ```bash
    python python_trainer.py [options]
    ```
    **Command-line Options:**
    * `--lr FLOAT`: Set the learning rate for training. (Default: `0.01`)
        * Example: `python python_trainer.py --lr 0.005`
    * `--epochs INT`: Set the number of training epochs. (Default: `20000`)
        * Example: `python python_trainer.py --epochs 50000`
    * `--output FILENAME`: Specify the name of the output file for the TI-BASIC "scroll" of weights. (Default: `nn_weights.txt`)
        * Example: `python python_trainer.py --output hermes_wisdom_v1.txt`
    * `--test`: Instructs the Scribe to run an accuracy test on the newly crafted "message" after training.
        * Example: `python python_trainer.py --test`
    * `--no-verbose`: Suppress the epoch-by-epoch progress output during training.
        * Example: `python python_trainer.py --no-verbose`

4.  **The Scribe's Output:** After training, the script will create a text file (e.g., `nn_weights.txt` or your custom name). This file contains the TI-BASIC code – the refined "message" for Hermes.

## Workflow: Delivering Refined Knowledge from PC to Hermes

This is the recommended path to empower Hermes with the most insightful classifications:

1.  **Craft the Message on PC:**
    * Run `python_trainer.py` with your desired learning rate and number of epochs. Training for more epochs (e.g., 20,000-100,000) with a suitable learning rate (e.g., 0.001-0.05) generally crafts more potent "messages."
    * Example: `python python_trainer.py --epochs 30000 --lr 0.008 --output hermes_scroll_v1.txt --test`
    * This will generate your "scroll" file (e.g., `hermes_scroll_v1.txt`).

2.  **Prepare the TI-BASIC Scroll:**
    * Open the generated text file (e.g., `hermes_scroll_v1.txt`) on your computer. It will contain many lines of TI-BASIC code.

3.  **Update Hermes's Knowledge on TI-84:**
    * On your TI-84 calculator, go to `[prgm]`, arrow over to `EDIT`, and select your Hermes program.
    * Scroll down in the code to find the label `Lbl P`. This section is where Hermes receives new knowledge.
    * **Carefully delete** all the lines of code that assign values to `[I](row,col)`, `[J](row,col)`, `L₄(i)`, and `L₅(i)`. These are the lines that look like:
        ```TI-BASIC
        Lbl P
        ClrHome
        Output(1,1,"LOADING PRETRAINED")
        Output(2,1,"WEIGHTS…")
        {16,4}→dim([I])
        {4,16}→dim([J])
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}→L₄
        {0,0,0,0}→L₅

        ; DELETE FROM HERE...
        ⁻1.922868→[I](1,1)
        ...
        ⁻0.189082→L₅(4)
        ; ...TO HERE

        Output(3,1,"LOAD COMPLETE")
        Pause 
        Goto A
        ```
    * The default knowledge section is quite long. Be precise in your deletions.

4.  **Inscribe the New Scroll:**
    * Manually type the TI-BASIC code from your generated text file (e.g., `hermes_scroll_v1.txt`) into the space you just cleared in the TI-84 program editor.
        * **Crucial:** When typing negative numbers, use the calculator's negative sign `[(-)]` key (which produces `⁻`), not the subtraction `[-]` key. The Python Scribe generates the correct `⁻` symbol.
        * This process can be tedious. Double-check your transcription. Using a TI-connectivity program that allows editing program code on the PC can make this transfer much swifter.

5.  **Hermes Awakens with New Knowledge:**
    * Exit the program editor on the calculator (usually `[2nd]` `[QUIT]`).
    * Run the Hermes program.
    * From the main menu, choose `SKIP`.
    * From the classifier menu, choose `LOAD PRETRAIN`. Hermes will now internalize your newly inscribed PC-trained "message."
    * Now, use the `CLASSIFY` option to witness Hermes's enhanced interpretative abilities!

## Technical Details of Hermes's Mind

* **Neural Network Architecture:**
    * **Input Layer (The Ears):** 4 neurons. Each neuron receives a numerical representation of a character (A=1/26, B=2/26, ..., Z=26/26). Input words are padded with 0s if shorter than 4 characters or truncated if longer.
    * **Hidden Layer (The Labyrinth of Thought):** 16 neurons.
    * **Output Layer (The Voice):** 4 neurons, one for each category ("DARK", "NEWT", "DEER", "DIRT").
* **Weights and Biases (The Synapses):**
    * `[I]`: 16x4 matrix for weights connecting input to hidden layer.
    * `[J]`: 4x16 matrix for weights connecting hidden to output layer.
    * `L₄`: List of 16 biases for the hidden layer neurons (referred to as `self.L4` in Python Scribe).
    * `L₅`: List of 4 biases for the output layer neurons (referred to as `self.L5` in Python Scribe).
* **Activation Function (The Spark):** Sigmoid function $f(x) = \frac{1}{(1 + e^{-x})}$. In the TI-BASIC code, outputs of the sigmoid are clamped if $x > 10$ (output $\approx 0.9999$) or $x < -10$ (output $\approx 0.0001$) to prevent overflow/underflow and maintain stable thought. The Python Scribe mirrors this.
* **Backpropagation (The Learning Process):** Both Hermes (on-calc) and the Scribe (PC) implement backpropagation to adjust weights and biases based on prediction error.
* **Data Augmentation (Broadening Horizons):**
    * **TI-BASIC Hermes:** During its training (`Lbl T`), if `R<5`, it introduces noise to the input vector `L₁`.
    * **Python Scribe:** The `train` method in the Python script applies augmentation by scrambling word letters or adding noise to encoded inputs, crafting a more robust understanding.
* **Confidence Score (Hermes's Certainty):** Calculated as `int(M/sum(L₃)*100)`, where `M` is the highest activation in the output layer `L₃`, and `sum(L₃)` is the total activation.

## Note on Hermes's "About" Screen

The "About" screen in the TI-BASIC program displays "- 4-16-4 arχtecture". The 'χ' is the Greek letter Chi, as written in the original program code, a subtle nod to the classics within Hermes's own description of his structure.

## Future Journeys for Hermes

* Expand Hermes's vocabulary to include more categories and words.
* Enable Hermes to interpret words of varying lengths (a significant architectural evolution).
* Develop faster "chariots" (TI linking tools or methods) for delivering "scrolls" of knowledge.

## License

This project is licensed under the MIT License.
