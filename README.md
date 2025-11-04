# Blackjack Monte Carlo

**Author:** INSERT AUTHOR HERE  
**License:** MIT (change if you prefer)

A Pygame-based blackjack sandbox featuring a tabular Monte Carlo control agent. Train the agent, play manually, get AI hints, watch the AI play, or run large batch simulations with a live, relative bar chart (Wins/Losses/Pushes). Includes an Info screen with newline-aware rendering for Monte Carlo RL notes. Single-file implementation.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run](#run)
- [Controls & Modes](#controls--modes)
- [Batch Simulation](#batch-simulation)
- [Info Screen](#info-screen)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [License](#license)

---

## Overview

This app demonstrates first-visit Monte Carlo control for blackjack using a compact, tabular implementation. The UI is built with Pygame and includes smooth card animations and a felt table look. The AI can be trained from scratch, and you can visually inspect its behavior or evaluate it at scale via batch runs.

---

## Features

- First-visit Monte Carlo control (on-policy, epsilon-greedy).
- Watch AI play hands visually or run headless batch simulations.
- Real-time, relative outcome bar chart with automatic label placement.
- Manual play mode with optional AI hints.
- Newline-aware Info screen for Monte Carlo RL notes.
- Unicode suit rendering with robust font fallbacks.

---

## Requirements

- Python 3.9+ (tested up to 3.13)
- Pygame 2.6+

---

## Installation

```bash
# create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install pygame
