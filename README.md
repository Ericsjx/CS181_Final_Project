# CS181_Final_Project(Reversi AI)

## Overview

This project implements several AI algorithms to play the game of Reversi (also known as Othello). The implemented agents include:

- Random Agent
- Greedy Agent
- Minimax Agent with Alpha-Beta Pruning
- Monte-Carlo Tree Search (MCTS) Agent
- Q-Learning Agent
- Approximate Q-Learning Agent

## Prerequisites

Ensure you have the following installed:

- Python 3.6 or higher
- Required Python packages (can be installed via `requirements.txt`)

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## How to Run

### Selecting Agents for Testing

1. Open the `main.py` file in a text editor.
2. Locate the section where the agents are defined.
3. Uncomment the lines for `agent1` and `agent2` to select the agents you want to compete against each other. For example:
   ```python
   # Uncomment the agents you want to use
   agent1 = RandomAgent()
   agent2 = GreedyAgent()
   ```
4. Save the changes to `main.py`.

### Running the Test

To start the test, open a terminal and run the following command:
```bash
python reversi_ai.py
```
This will launch the game with a graphical user interface (GUI) by default.

### Running Without GUI

If you prefer to run the test without the GUI, use the following command:
```bash
python reversi_ai.py --no-GUI
```

## File Structure

- `main.py`: Main script to set up and start the game.
- `reversi_ai.py`: Script to run the game, supporting GUI and non-GUI modes.
- `agents/`: Directory containing the implementations of various agents.
- `utils/`: Utility functions for the game.
- `requirements.txt`: List of required Python packages.

## Notes

- The default setting runs the game with a GUI interface for better visualization of the agents' decisions.
- Running without the GUI can be useful for quicker tests and performance measurements.

## Authors

This project was developed by [Your Team Name]. For any questions or feedback, please contact [Your Contact Information].

---

Please review and let me know if there are any additional details or changes you would like to include.
