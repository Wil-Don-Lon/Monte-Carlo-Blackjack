# Blackjack Monte Carlo

An interactive blackjack game demonstrating Monte Carlo reinforcement learning with real-time AI training and visualization.

## Overview

This project implements a complete blackjack environment with a tabular Monte Carlo control agent that learns optimal play through self-play. The application features multiple modes, including manual play, AI-assisted play with hints, live AI demonstrations, and batch simulation with statistical analysis.

The project evolved from a command-line interface (CLI) implementation in Jupyter notebook by Medhansh Sankaran into a full-featured PyGame graphical interface by William Donnell-Lonon. The core Monte Carlo algorithm, blackjack environment, and agent logic remain faithful to the original implementation while adding rich visualizations and interactive features.

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

## Project Evolution

### Original CLI Implementation

The original implementation by Medhansh Sankaran was a terminal-based interactive program featuring:

- **Text-based menu system**: Five options including training, watching the agent, and manual play
- **Episode generation**: Complete episode sampling under the epsilon-greedy policy
- **State formatting**: Human-readable state representations showing player sum, dealer upcard, and usable ace status
- **Action suggestions**: Real-time recommendations from the trained policy during manual play
- **Agent performance tracking**: Win/loss/push statistics across multiple hands

The CLI version demonstrated the core MC algorithm with a clean, minimalist interface focused on the learning mechanics.

### PyGame Graphical Interface

William Donnell-Lonon adapted the CLI implementation into a full graphical application while preserving the original algorithm:

- **Visual card representations**: Animated playing cards with proper suit symbols and smooth positioning
- **Real-time training visualization**: Progress bars, episode counters, and decorative card animations during training
- **Interactive UI components**: Button system with hover effects and state management
- **Live statistics display**: Persistent win/loss/push tracking across sessions
- **Batch simulation charts**: Dynamic bar graphs showing outcome distributions with real-time updates
- **Multi-platform font handling**: Robust Unicode font loading for card suit symbols across operating systems
- **Smooth animations**: 60 FPS rendering with interpolated card movements

The graphical version maintains identical game logic and Monte Carlo implementation while making the learning process visually engaging and accessible to non-technical users.

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

The training process is identical between CLI and GUI versions, with the GUI adding visual feedback through progress bars and animated card backs.

## Controls

- **Mouse**: Click buttons to navigate menus and make decisions
- **Escape/Close**: Quit the application

## Project Structure

```
blackjack_mc.py          # Main PyGame application with GUI
blackjack_cli.py         # Original CLI implementation (if included)
README.md                # This file
```

Key classes:
- `BlackjackEnv`: Episodic environment with step/reset interface
- `MCBlackjackAgent`: Tabular Monte Carlo control with epsilon-greedy policy
- `BlackjackGame`: PyGame application handling UI, rendering, and game flow
- `Card`: Visual card representation with suit glyphs and smooth animation
- `Button`: Interactive UI button component

### Core Algorithm (shared between versions)

Both implementations share the same Monte Carlo logic:

```python
def train(self, env_factory, episodes=100000):
    for _ in range(episodes):
        env = env_factory()
        ep = self.generate_episode(env)
        
        visited = set()
        G = 0.0
        for t in reversed(range(len(ep))):
            s, a, r = ep[t]
            G = self.gamma * G + r
            if a is None:
                continue
            sa = (s, a)
            if sa not in visited:
                visited.add(sa)
                self.returns_sum[s][a] += G
                self.returns_count[s][a] += 1
                self.Q[s][a] = self.returns_sum[s][a] / self.returns_count[s][a]
```

This first-visit MC control algorithm with running averages forms the foundation of both the CLI and GUI versions.

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
- Heatmap visualizations of learned Q-values
- Policy comparison tool between different training runs

## Credits

Original CLI implementation and Monte Carlo algorithm: Medhansh Sankaran

PyGame interface and visualization system: William Donnell-Lonon

## License

This project is provided as-is for educational purposes. Feel free to modify and distribute with attribution.

## Acknowledgments

Monte Carlo methods form one of the foundational approaches in reinforcement learning, introduced by Sutton and Barto in their seminal work "Reinforcement Learning: An Introduction". This implementation demonstrates the effectiveness of simple tabular methods on small state spaces like blackjack.
