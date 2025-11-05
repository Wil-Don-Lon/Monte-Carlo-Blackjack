# Blackjack Monte Carlo

An interactive blackjack game demonstrating Monte Carlo reinforcement learning with real-time AI training and visualization.

## Overview

This project implements a complete blackjack environment with a tabular Monte Carlo control agent that learns optimal play through self-play. The application features multiple modes, including manual play, AI-assisted play with hints, live AI demonstrations, and batch simulation with statistical analysis. The terminal-based Jupyter notebook was created by Medhansh Sankaran, which was then adapted into a PyGame interface by William Donnell-Lonon. 

## Features

- **Monte Carlo RL Agent**: First-visit Monte Carlo control with epsilon-greedy exploration
- **Interactive Training**: Watch the AI train in real-time with progress visualization
- **Multiple Game Modes**:
  - Manual play without assistance
  - Manual play with AI hints showing optimal moves
  - Watch AI play single hands with animation
  - Batch simulation mode with outcome distribution charts
- **Rich Visualizations**: Animated card dealing, live statistics, and relative bar charts
- **Educational Info Screen**: Explains Monte Carlo RL concepts and implementation details

## Requirements

- Python 3.7+
- pygame

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Wil-Don-Lon/blackjack-monte-carlo.git
cd blackjack-monte-carlo
```

2. Install dependencies:
```bash
pip install pygame
```

## Usage

Run the game:
```bash
python blackjack_mc.py
```

### Game Modes

**Train AI**: Trains the Monte Carlo agent on 1,000,000 episodes with live progress tracking. Training takes approximately 30-60 seconds depending on your hardware.

**Play (Manual)**: Standard blackjack where you make all decisions. Cards are dealt from an infinite deck and the dealer hits on soft 17 by default.

**Play with AI Hints**: Same as manual mode but the AI suggests the optimal action based on its learned policy after each card.

**Watch AI Play**: Observe the trained AI play hands automatically with an 800ms delay between actions for visibility. Includes batch simulation controls.

**Info & MC Notes**: View a summary of Monte Carlo reinforcement learning concepts and project credits.

## Batch Simulation

In "Watch AI Play" mode, you can run batch simulations to see statistical performance:

1. Select a batch size (5,000 / 25,000 / 100,000 games)
2. Click "RUN BATCH" to start the simulation
3. Watch the real-time bar chart update with wins, losses, and pushes
4. Stop early with "STOP BATCH" or let it complete

The batch mode uses the greedy policy (no exploration) and runs games without animation for speed.

## Implementation Details

### Environment

The blackjack environment follows standard rules:
- Player and dealer each start with 2 cards
- Face cards count as 10, Aces as 1 or 11 (whichever is better)
- Player can hit or stand
- Dealer hits until reaching 17+ (including soft 17 by default)
- Rewards: +1 for win, -1 for loss, 0 for push
- Natural blackjacks (Ace + 10-value) end the hand immediately

### Agent

The Monte Carlo agent uses:
- **First-visit Monte Carlo control**: Updates Q-values only on the first occurrence of each state-action pair per episode
- **Epsilon-greedy policy**: Explores random actions with probability 0.1 during training
- **Running average returns**: Efficient Q-value updates without storing full return histories
- **Tabular representation**: State space is (player_sum, dealer_upcard, usable_ace)

### Training

Default training runs 1,000,000 episodes, which provides strong convergence for the relatively small state space (roughly 200 states). The agent learns to:
- Hit on totals below 17 in most cases
- Stand on strong hands (18+)
- Adjust strategy based on dealer upcard
- Handle soft totals (with usable Ace) differently than hard totals

## Controls

- **Mouse**: Click buttons to navigate menus and make decisions
- **Escape/Close**: Quit the application

## Project Structure

```
blackjack_mc.py          # Main game file containing all classes and game loop
README.md                # This file
```

Key classes:
- `BlackjackEnv`: Episodic environment with step/reset interface
- `MCBlackjackAgent`: Tabular Monte Carlo control with epsilon-greedy policy
- `BlackjackGame`: Pygame application handling UI, rendering, and game flow
- `Card`: Visual card representation with suit glyphs and smooth animation
- `Button`: Interactive UI button component

## Technical Notes

### Font Handling

The application uses a robust Unicode font loading system to ensure card suit symbols (spades, hearts, diamonds, clubs) render correctly across Windows, macOS, and Linux. It attempts to load system fonts in order of preference and falls back gracefully if none are found.

### Performance

- Training 1M episodes: 30-60 seconds
- Batch simulation: ~1000 games/second (varies by hardware)
- UI runs at 60 FPS with smooth card animations

### Monte Carlo Specifics

This implementation uses **on-policy** first-visit MC control. The agent improves by:
1. Generating episodes using the current epsilon-greedy policy
2. Computing returns (discounted sum of rewards, gamma=1.0)
3. Updating Q-values with running averages
4. Improving policy greedily with respect to updated Q-values

The epsilon parameter (0.1) balances exploration and exploitation, ensuring the agent continues to discover better strategies even after initial convergence.

## Future Enhancements

Potential additions:
- Configurable dealer rules (hit/stand on soft 17)
- Card counting visualization
- Comparison with other RL algorithms (Q-learning, SARSA)
- Save/load trained policies
- Multi-deck shoe with penetration
- Betting and bankroll management
- Additional blackjack variants (European, Spanish 21)

## Credits

Interface designed by William Donnell-Lonon
Monte Carlo implementation architected by Medhansh Sankaran.

## License

This project is provided as-is for educational purposes. Feel free to modify and distribute with attribution.

## Acknowledgments

Monte Carlo methods form one of the foundational approaches in reinforcement learning, introduced by Sutton and Barto in their seminal work "Reinforcement Learning: An Introduction". This implementation demonstrates the effectiveness of simple tabular methods on small state spaces like blackjack.
