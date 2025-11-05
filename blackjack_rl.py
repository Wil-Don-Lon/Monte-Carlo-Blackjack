import pygame
import random
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
pygame.init()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
WIDTH, HEIGHT = 1200, 800
FPS = 60

# Colors
FELT_GREEN = (53, 101, 77)
DARK_GREEN = (35, 65, 50)
GOLD = (255, 215, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 20, 60)
CARD_RED = (200, 0, 0)
BLUE = (70, 130, 180)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

# Cards
SUITS = ['♠', '♥', '♦', '♣']
SUIT_COLORS = {'♠': BLACK, '♥': CARD_RED, '♦': CARD_RED, '♣': BLACK}

# Layout
PLAYER_LABEL_Y = HEIGHT - 410
PLAYER_VALUE_Y = HEIGHT - 190
AI_SUGGEST_Y   = HEIGHT - 160

# Hand center targets (slightly higher and left for both hands)
HAND_CENTER_X = WIDTH // 2 - 40
DEALER_Y      = 190
PLAYER_Y      = HEIGHT - 360

# Training screen vertical offset
TRAIN_Y_SHIFT = 60

# ---------------------------------------------------------
# Fonts
# ---------------------------------------------------------
def load_unicode_font(size: int) -> pygame.font.Font:
    """
    Load a system font with reliable Unicode coverage for card suits.
    Tries common families first, then platform-specific paths, with a final fallback.
    """
    family_candidates = [
        "Segoe UI Symbol",    # Windows
        "Apple Symbols",      # macOS
        "DejaVu Sans",        # Linux
        "Arial Unicode MS",
        "Noto Sans Symbols",
        "FreeSerif",
    ]
    path = pygame.font.match_font(family_candidates, bold=False, italic=False)
    if path:
        return pygame.font.Font(path, size)

    fallback_paths = []
    if sys.platform.startswith("win"):
        fallback_paths += [
            r"C:\Windows\Fonts\seguisym.ttf",
            r"C:\Windows\Fonts\arialuni.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
    elif sys.platform == "darwin":
        fallback_paths += [
            "/System/Library/Fonts/Apple Symbols.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/Library/Fonts/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
    else:
        fallback_paths += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansSymbols-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        ]

    for p in fallback_paths:
        if os.path.exists(p):
            return pygame.font.Font(p, size)

    return pygame.font.Font(None, size)

# ---------------------------------------------------------
# Blackjack environment (tabular Monte Carlo target)
# ---------------------------------------------------------
def draw_card():
    """Return a random card value with face cards as 10; Ace as 1."""
    cards = [1,2,3,4,5,6,7,8,9,10,10,10,10]
    return random.choice(cards)

def draw_hand():
    return [draw_card(), draw_card()]

def usable_ace(hand):
    """Return True if an ace can be counted as 11 without busting."""
    return 1 in hand and sum(hand) + 10 <= 21

def hand_value(hand):
    """Return blackjack hand value with a usable ace counted as 11."""
    val = sum(hand)
    return val + 10 if usable_ace(hand) else val

def is_natural(hand):
    """Return True if the hand is a natural blackjack (Ace + 10-value)."""
    return sorted(hand) == [1, 10]

class BlackjackEnv:
    """
    Simple episodic blackjack environment.
    Observation: (player_value, dealer_upcard, usable_ace)
    Actions: 0=Stand, 1=Hit
    Rewards: +1 win, 0 push, -1 loss
    """
    def __init__(self, hit_on_soft_17: bool = False):
        self.hit_on_soft_17 = hit_on_soft_17
        self.reset()

    def reset(self):
        self.player = draw_hand()
        self.dealer = draw_hand()
        if is_natural(self.player) or is_natural(self.dealer):
            self.done = True
            self.reward = self._natural_outcome()
        else:
            self.done = False
            self.reward = 0
        return self._obs()

    def _natural_outcome(self):
        p_nat = is_natural(self.player)
        d_nat = is_natural(self.dealer)
        if p_nat and not d_nat:
            return 1
        if d_nat and not p_nat:
            return -1
        return 0

    def _dealer_policy(self, hand):
        """Dealer draws to 17; optional hit on soft 17."""
        while True:
            val = hand_value(hand)
            if val < 17:
                hand.append(draw_card())
                continue
            if val == 17 and usable_ace(hand) and self.hit_on_soft_17:
                hand.append(draw_card())
                continue
            break

    def step(self, action):
        """
        Perform an action. If action=Hit, player draws and may bust.
        If action=Stand, dealer plays out and hand is scored.
        """
        if self.done:
            return self._obs(), self.reward, self.done, {}

        # Hit
        if action == 1:
            self.player.append(draw_card())
            if hand_value(self.player) > 21:
                self.done = True
                self.reward = -1
            return self._obs(), self.reward, self.done, {}

        # Stand: dealer plays out, compare totals
        self._dealer_policy(self.dealer)
        self.done = True
        p, d = hand_value(self.player), hand_value(self.dealer)
        if d > 21 or p > d:
            self.reward = 1
        elif p < d:
            self.reward = -1
        else:
            self.reward = 0
        return self._obs(), self.reward, self.done, {}

    def _obs(self):
        return (hand_value(self.player), self.dealer[0], usable_ace(self.player))

# ---------------------------------------------------------
# Monte Carlo control agent (tabular, ε-greedy)
# ---------------------------------------------------------
class MCBlackjackAgent:
    def __init__(self, epsilon=0.1, gamma=1.0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: [0.0, 0.0])          # action-values for (state)->[Q_stand, Q_hit]
        self.returns_sum = defaultdict(lambda: [0.0, 0.0])
        self.returns_count = defaultdict(lambda: [0, 0])

    def policy(self, state):
        """ε-greedy policy from Q."""
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        q0, q1 = self.Q[state]
        return 0 if q0 >= q1 else 1

    def greedy_action(self, state):
        """Greedy action for inference/UI."""
        q0, q1 = self.Q[state]
        return 0 if q0 >= q1 else 1

    def generate_episode(self, env):
        """Simulate an episode under current ε-greedy policy."""
        episode = []
        state = env.reset()
        if env.done:
            # Natural outcome without actions
            return [(state, None, env.reward)]
        while True:
            action = self.policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    def train(self, env_factory, episodes=1_000_000):
        """First-visit Monte Carlo control with running-average returns."""
        for _ in range(episodes):
            env = env_factory()
            ep = self.generate_episode(env)
            visited = set()
            G = 0.0
            for t in reversed(range(len(ep))):
                s, a, r = ep[t]
                G = self.gamma * G + r
                if a is None:
                    continue  # no action taken on a terminal natural
                sa = (s, a)
                if sa not in visited:
                    visited.add(sa)
                    self.returns_sum[s][a] += G
                    self.returns_count[s][a] += 1
                    self.Q[s][a] = self.returns_sum[s][a] / self.returns_count[s][a]

# ---------------------------------------------------------
# UI Components
# ---------------------------------------------------------
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False
        self.enabled = True

    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered and self.enabled else self.color
        if not self.enabled:
            color = GRAY
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, GOLD, self.rect, 3, border_radius=10)
        text_surf = font.render(self.text, True, WHITE if self.enabled else DARK_GREEN)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.enabled:
            if self.rect.collidepoint(event.pos):
                return True
        return False

class Card:
    """Renderable card with a single suit glyph centered and value marks in corners."""
    def __init__(self, value, x, y):
        self.value = value
        self.suit = random.choice(SUITS)
        if self.value == 10:
            r = random.random()
            if r < 0.25:
                self.display_value = "J"
            elif r < 0.5:
                self.display_value = "Q"
            elif r < 0.75:
                self.display_value = "K"
            else:
                self.display_value = "10"
        elif self.value == 1:
            self.display_value = "A"
        else:
            self.display_value = str(self.value)
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.width = 80
        self.height = 120

    def update(self):
        """Smoothly move toward target position."""
        self.x += (self.target_x - self.x) * 0.2
        self.y += (self.target_y - self.y) * 0.2

    def draw(self, screen, font_large, font_small, hidden=False):
        if hidden:
            # Card back
            pygame.draw.rect(screen, RED, (self.x, self.y, self.width, self.height), border_radius=8)
            pygame.draw.rect(screen, GOLD, (self.x, self.y, self.width, self.height), 3, border_radius=8)
            for i in range(3):
                for j in range(4):
                    cx = self.x + 15 + i * 25
                    cy = self.y + 15 + j * 27
                    pygame.draw.circle(screen, DARK_GREEN, (int(cx), int(cy)), 8)
            return

        # Face
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height), border_radius=8)
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2, border_radius=8)

        color = SUIT_COLORS[self.suit]

        # Top-left value
        value_tl = font_small.render(self.display_value, True, color)
        screen.blit(value_tl, (self.x + 8, self.y + 6))

        # Center suit glyph
        center_suit = font_large.render(self.suit, True, color)
        suit_rect = center_suit.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(center_suit, suit_rect)

        # Bottom-right value
        value_br = font_small.render(self.display_value, True, color)
        screen.blit(
            value_br,
            (self.x + self.width - value_br.get_width() - 8,
             self.y + self.height - value_br.get_height() - 6)
        )

# ---------------------------------------------------------
# Game
# ---------------------------------------------------------
class BlackjackGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Blackjack Monte Carlo")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = load_unicode_font(52)
        self.font_medium = load_unicode_font(36)
        self.font_small  = load_unicode_font(24)

        # Environment and agent
        self.env = BlackjackEnv()
        self.agent = MCBlackjackAgent(epsilon=0.1, gamma=1.0)

        # Session state
        self.mode = "menu"   # "menu" | "training" | "playing" | "ai_playing" | "ai_batch" | "info"
        self.player_cards = []
        self.dealer_cards = []
        self.show_dealer_card = False
        self.result_text = ""
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.ai_suggestion = ""
        self.training_progress = 0
        self.is_training = False
        self.use_ai_hints = False
        self.training_episodes_done = 0
        self.training_target = 0

        # Batch sim state
        self.batch_total = 0
        self.batch_done = 0
        self.batch_update_every = 25
        self.batch_running = False
        self.batch_counts = {"Wins": 0, "Losses": 0, "Pushes": 0}

        # Info screen state
        self.prev_mode = None
        self.info_text = (
            "Monte Carlo RL (quick hits):\n"
            "• Learn action values from complete episodes by averaging returns.\n"
            "• First-visit MC: update each (state,action) only on its first occurrence per episode.\n"
            "• Policy improvement: act ε-greedily w.r.t. current Q.\n"
            "• Blackjack env: reward +1/-1/0 for win/loss/push; terminal at stand/bust.\n"
            "• In this app: on-policy, tabular Q with running averages.\n"
            "\n"
            "This project was created by William Donnell-Lonon and Medhansh Sankaran."
        )

        # Controls
        self.create_buttons()

    # ---------- UI Helpers ----------
    def create_buttons(self):
        button_y = 300
        spacing = 90
        self.menu_buttons = [
            Button(WIDTH//2 - 150, button_y + 0*spacing, 300, 60, "Train AI",            BLUE, (100, 160, 210)),
            Button(WIDTH//2 - 150, button_y + 1*spacing, 300, 60, "Play (Manual)",       BLUE, (100, 160, 210)),
            Button(WIDTH//2 - 150, button_y + 2*spacing, 300, 60, "Play with AI Hints",  BLUE, (100, 160, 210)),
            Button(WIDTH//2 - 150, button_y + 3*spacing, 300, 60, "Watch AI Play",       BLUE, (100, 160, 210)),
            Button(WIDTH//2 - 150, button_y + 4*spacing, 300, 60, "Info & MC Notes",     BLUE, (100, 160, 210)),
        ]
        self.hit_button       = Button(WIDTH//2 - 260, HEIGHT - 120, 120, 50, "HIT",        (34, 139, 34), (50, 205, 50))
        self.stand_button     = Button(WIDTH//2 - 120, HEIGHT - 120, 120, 50, "STAND",      RED, (255, 69, 69))
        self.new_hand_button  = Button(WIDTH//2 +  20, HEIGHT - 120, 200, 50, "NEW HAND",   BLUE, (100, 160, 210))
        self.back_button      = Button(50, HEIGHT - 80, 120, 50, "MENU", GRAY, LIGHT_GRAY)
        self.run_batch_button = Button(WIDTH//2 +  240, HEIGHT - 120, 220, 50, "RUN BATCH", BLUE, (100, 160, 210))
        self.batch_stop_button= Button(WIDTH//2 - 100, HEIGHT - 80, 200, 50, "STOP BATCH",  RED, (255, 69, 69))

        # Quick selectors for batch size (shown on Watch AI Play screen)
        self.batch_small_btn  = Button(80,  80, 160, 40, "Batch: 5,000",   BLUE, (100, 160, 210))
        self.batch_medium_btn = Button(260, 80, 180, 40, "Batch: 25,000",  BLUE, (100, 160, 210))
        self.batch_large_btn  = Button(460, 80, 180, 40, "Batch: 100,000", BLUE, (100, 160, 210))

    def layout_cards_centered(self, cards, center_x, y, spacing=100):
        """Center a hand of cards horizontally around center_x."""
        if not cards:
            return
        total_width = (len(cards) - 1) * spacing
        start_x = center_x - total_width // 2
        for i, card in enumerate(cards):
            card.target_x = start_x + i * spacing
            card.target_y = y

    def finalize_natural(self):
        """Apply outcome and UI when the initial deal ends the round."""
        self.show_dealer_card = True
        reward = self.env.reward
        if reward > 0:
            self.result_text = "BLACKJACK! YOU WIN!"
            self.wins += 1
        elif reward < 0:
            self.result_text = "DEALER BLACKJACK! YOU LOSE!"
            self.losses += 1
        else:
            self.result_text = "PUSH! BLACKJACK"
            self.pushes += 1
        self.ai_suggestion = ""

    # ---------- Game Flow ----------
    def start_new_hand(self):
        self.env.reset()
        self.player_cards = []
        self.dealer_cards = []

        # Create initial card sprites
        for i, card_val in enumerate(self.env.player):
            self.player_cards.append(Card(card_val, WIDTH//2 - 100 + i*100, PLAYER_Y))
        for i, card_val in enumerate(self.env.dealer):
            self.dealer_cards.append(Card(card_val, WIDTH//2 - 100 + i*100, DEALER_Y))

        # Position hands centered at target anchors
        self.layout_cards_centered(self.dealer_cards, HAND_CENTER_X, DEALER_Y)
        self.layout_cards_centered(self.player_cards, HAND_CENTER_X, PLAYER_Y)

        self.show_dealer_card = False
        self.result_text = ""
        self.ai_suggestion = ""

        # Immediate terminal outcome (natural)
        if self.env.done:
            self.finalize_natural()
            return

        # Initial AI hint
        if self.use_ai_hints:
            state = self.env._obs()
            action = self.agent.greedy_action(state)
            self.ai_suggestion = f"AI suggests: {'HIT' if action == 1 else 'STAND'}"

    def handle_action(self, action):
        """Advance the environment by one player action and update UI state."""
        _, reward, done, _ = self.env.step(action)

        # Add card sprite on hit, then re-center
        if action == 1:
            self.player_cards.append(Card(self.env.player[-1], HAND_CENTER_X, PLAYER_Y))
            self.layout_cards_centered(self.player_cards, HAND_CENTER_X, PLAYER_Y)

        # On terminal, reveal dealer and append dealer draws
        if done:
            self.show_dealer_card = True
            while len(self.dealer_cards) < len(self.env.dealer):
                idx = len(self.dealer_cards)
                self.dealer_cards.append(Card(self.env.dealer[idx], HAND_CENTER_X, DEALER_Y))
            self.layout_cards_centered(self.dealer_cards, HAND_CENTER_X, DEALER_Y)

            if reward > 0:
                self.result_text = "YOU WIN!"
                self.wins += 1
            elif reward < 0:
                self.result_text = "YOU LOSE!"
                self.losses += 1
            else:
                self.result_text = "PUSH!"
                self.pushes += 1
        elif self.use_ai_hints:
            state = self.env._obs()
            action = self.agent.greedy_action(state)
            self.ai_suggestion = f"AI suggests: {'HIT' if action == 1 else 'STAND'}"

    def run_training(self, episodes=1_000_000):
        """Train the agent and display simple progress UI."""
        self.is_training = True
        self.training_progress = 0
        self.training_episodes_done = 0
        self.training_target = episodes

        def env_factory():
            return BlackjackEnv()

        batch_size = 500
        update_frequency = 5  # UI refresh cadence

        batch_count = 0
        for i in range(0, episodes, batch_size):
            current_batch = min(batch_size, episodes - i)
            self.agent.train(env_factory, episodes=current_batch)
            self.training_episodes_done += current_batch
            self.training_progress = int((self.training_episodes_done / episodes) * 100)

            batch_count += 1
            if batch_count % update_frequency == 0:
                self.screen.fill(FELT_GREEN)

                # Decorative card backs
                for j in range(4):
                    demo_card = Card(random.randint(1, 10), 100 + j * 250, 150)
                    demo_card.draw(self.screen, self.font_large, self.font_small, hidden=True)

                # Title
                title = self.font_large.render("Training AI...", True, GOLD)
                self.screen.blit(title, (WIDTH//2 - title.get_width()//2, (HEIGHT//2 - 150) + TRAIN_Y_SHIFT))

                # Percent
                pct = self.font_medium.render(f"{self.training_progress}% Complete", True, WHITE)
                self.screen.blit(pct, (WIDTH//2 - pct.get_width()//2, (HEIGHT//2 - 50) + TRAIN_Y_SHIFT))

                # Episodes
                ep_text = self.font_small.render(f"{self.training_episodes_done:,} / {episodes:,} episodes", True, LIGHT_GRAY)
                self.screen.blit(ep_text, (WIDTH//2 - ep_text.get_width()//2, (HEIGHT//2 - 10) + TRAIN_Y_SHIFT))

                # Progress bar
                bar_width = 500
                bar_height = 40
                bar_y = (HEIGHT//2 + 40) + TRAIN_Y_SHIFT
                bar_x = WIDTH//2 - bar_width//2

                pygame.draw.rect(self.screen, DARK_GREEN, (bar_x, bar_y, bar_width, bar_height), border_radius=10)
                pygame.draw.rect(self.screen, GOLD, (bar_x, bar_y, bar_width * self.training_progress // 100, bar_height), border_radius=10)
                pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2, border_radius=10)

                pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

        # Completion screen
        self.screen.fill(FELT_GREEN)
        title = self.font_large.render("Training Complete!", True, GOLD)
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 100))

        complete_text = self.font_medium.render(f"Trained on {episodes:,} episodes", True, WHITE)
        self.screen.blit(complete_text, (WIDTH//2 - complete_text.get_width()//2, HEIGHT//2 - 20))

        continue_text = self.font_small.render("Click anywhere to continue...", True, LIGHT_GRAY)
        self.screen.blit(continue_text, (WIDTH//2 - continue_text.get_width()//2, HEIGHT//2 + 40))

        pygame.display.flip()

        # Await click
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False

        self.is_training = False
        return True

    # ---------- AI Batch Simulation ----------
    def start_batch(self, total_games):
        """Initialize batch-run stats and switch to batch mode."""
        self.batch_total = int(total_games)
        self.batch_done = 0
        self.batch_counts = {"Wins": 0, "Losses": 0, "Pushes": 0}
        self.batch_running = True
        self.mode = "ai_batch"

    def simulate_one_game_greedy(self):
        """Run a single game with greedy policy (no animations). Return reward."""
        env = BlackjackEnv()
        state = env._obs()
        if env.done:
            return env.reward
        while True:
            action = self.agent.greedy_action(state)
            state, reward, done, _ = env.step(action)
            if done:
                return reward

    def update_batch_logic(self):
        """Advance batch by a small chunk so we can animate the bar chart smoothly."""
        per_tick = self.batch_update_every
        remaining = self.batch_total - self.batch_done
        n = min(per_tick, remaining)
        for _ in range(n):
            r = self.simulate_one_game_greedy()
            if r > 0:
                self.batch_counts["Wins"] += 1
            elif r < 0:
                self.batch_counts["Losses"] += 1
            else:
                self.batch_counts["Pushes"] += 1
            self.batch_done += 1
        pygame.time.wait(1)
        if self.batch_done >= self.batch_total:
            self.batch_running = False

    # ---------- Info helpers (newline-aware wrapping) ----------
    def _wrap_text(self, text, font, max_width):
        """
        Wrap text to fit max_width and RESPECT explicit newlines.
        Returns a list of lines; empty strings represent paragraph breaks.
        """
        lines = []
        for para in text.split("\n"):
            if para.strip() == "":
                lines.append("")  # paragraph break
                continue
            words = para.split()
            cur = ""
            for w in words:
                test = f"{cur} {w}".strip()
                if font.size(test)[0] <= max_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
        return lines

    # ---------- Rendering helpers ----------
    def draw_text_with_outline(self, surface, text, font, color, x, y, outline_color=BLACK):
        """Draw text with a 1px outline for readability."""
        txt = font.render(text, True, color)
        out = font.render(text, True, outline_color)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            surface.blit(out, (x+dx, y+dy))
        surface.blit(txt, (x, y))

    def draw_bars(self, origin_x, origin_y, width, height, labels_to_values):
        """
        Draw a relative bar-graph that updates in real-time.
        Bars scale to the max value across series. Labels auto-place either
        above the bar (if room) or inside near the top to avoid overlaps.
        """
        # Frame
        pygame.draw.rect(self.screen, DARK_GREEN, (origin_x, origin_y, width, height), border_radius=12)
        pygame.draw.rect(self.screen, WHITE, (origin_x, origin_y, width, height), 2, border_radius=12)

        labels = list(labels_to_values.keys())
        values = list(labels_to_values.values())
        max_val = max(1, max(values))  # avoid division by zero

        # Chart layout
        chart_top_pad = 70  # extra room for title / tall bars
        side_pad = 40
        gap = 20
        chart_bottom = origin_y + height - 60
        chart_top = origin_y + chart_top_pad
        chart_h = max(1, chart_bottom - chart_top)

        # Title
        title = self.font_small.render("Outcome Distribution (Relative)", True, WHITE)
        self.screen.blit(title, (origin_x + (width - title.get_width()) // 2, origin_y + 10))

        # Bar geometry
        bar_w = int((width - (2 * side_pad) - (gap * (len(labels) - 1))) / len(labels))
        bar_w = max(20, bar_w)

        total = sum(values) if sum(values) > 0 else 1

        for i, label in enumerate(labels):
            v = values[i]
            h = int((v / max_val) * chart_h)
            x = origin_x + side_pad + i * (bar_w + gap)
            y = chart_bottom - h

            # Bar rectangle
            pygame.draw.rect(self.screen, BLUE, (x, y, bar_w, h), border_radius=6)
            pygame.draw.rect(self.screen, GOLD, (x, y, bar_w, h), 2, border_radius=6)

            # Labels (value + percent)
            val_s = f"{v:,}"
            pct_s = f"{(100.0 * v / total):4.1f}%"
            val_surf = self.font_small.render(val_s, True, WHITE)
            pct_surf = self.font_small.render(pct_s, True, LIGHT_GRAY)
            text_stack_h = val_surf.get_height() + 2 + pct_surf.get_height()

            if (y - chart_top) >= (text_stack_h + 6):
                # Place above bar
                val_y = y - text_stack_h - 4
                pct_y = val_y + val_surf.get_height() + 2
            else:
                # Place inside near top of bar
                val_y = y + 6
                pct_y = val_y + val_surf.get_height() + 2

            val_x = x + (bar_w - val_surf.get_width()) // 2
            pct_x = x + (bar_w - pct_surf.get_width()) // 2

            self.draw_text_with_outline(self.screen, val_s, self.font_small, WHITE, val_x, val_y)
            self.draw_text_with_outline(self.screen, pct_s, self.font_small, LIGHT_GRAY, pct_x, pct_y)

            # Category label
            lbl = self.font_small.render(label, True, WHITE)
            self.screen.blit(lbl, (x + (bar_w - lbl.get_width()) // 2, chart_bottom + 10))

        # Footer totals
        footer = self.font_small.render(f"Total games: {sum(values):,}", True, WHITE)
        self.screen.blit(footer, (origin_x + width - footer.get_width() - 12, origin_y + height - 28))

    def draw_info(self):
        # Dim background
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))

        # Card-like panel
        panel_w, panel_h = WIDTH - 200, HEIGHT - 200
        panel_x, panel_y = 100, 100
        pygame.draw.rect(self.screen, DARK_GREEN, (panel_x, panel_y, panel_w, panel_h), border_radius=16)
        pygame.draw.rect(self.screen, WHITE, (panel_x, panel_y, panel_w, panel_h), 2, border_radius=16)

        # Title
        title = self.font_large.render("Info — Monte Carlo RL", True, GOLD)
        self.screen.blit(title, (panel_x + (panel_w - title.get_width()) // 2, panel_y + 20))

        # Body text (newline-aware wrapped)
        body_x = panel_x + 30
        body_y = panel_y + 100
        body_w = panel_w - 60

        line_height = self.font_small.get_height()
        y = body_y
        for line in self._wrap_text(self.info_text, self.font_small, body_w):
            if line == "":
                y += line_height + 6  # extra gap for paragraph breaks
                continue
            line_surf = self.font_small.render(line, True, WHITE)
            self.screen.blit(line_surf, (body_x, y))
            y += line_height + 4

        # Back/Menu button
        self.back_button.draw(self.screen, self.font_small)

    # ---------- Main Loop ----------
    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.mode == "menu":
                    if self.menu_buttons[0].handle_event(event):
                        self.mode = "training"
                        if self.run_training(1_000_000):
                            self.mode = "menu"
                        else:
                            running = False
                    elif self.menu_buttons[1].handle_event(event):
                        self.mode = "playing"
                        self.use_ai_hints = False
                        self.start_new_hand()
                    elif self.menu_buttons[2].handle_event(event):
                        self.mode = "playing"
                        self.use_ai_hints = True
                        self.start_new_hand()
                    elif self.menu_buttons[3].handle_event(event):
                        self.mode = "ai_playing"
                        self.start_new_hand()
                    elif self.menu_buttons[4].handle_event(event):
                        self.prev_mode = "menu"
                        self.mode = "info"

                elif self.mode == "playing":
                    if self.back_button.handle_event(event):
                        self.mode = "menu"
                    elif not self.env.done:
                        if self.hit_button.handle_event(event):
                            self.handle_action(1)
                        elif self.stand_button.handle_event(event):
                            self.handle_action(0)
                    else:
                        if self.new_hand_button.handle_event(event):
                            self.start_new_hand()

                elif self.mode == "ai_playing":
                    if self.back_button.handle_event(event):
                        self.mode = "menu"
                    elif self.new_hand_button.handle_event(event):
                        self.start_new_hand()
                    elif self.run_batch_button.handle_event(event):
                        # default batch size when clicked; user can change with quick selectors
                        self.start_batch(self.batch_total if self.batch_total > 0 else 25000)
                    # quick batch-size selectors
                    elif self.batch_small_btn.handle_event(event):
                        self.batch_total = 5000
                    elif self.batch_medium_btn.handle_event(event):
                        self.batch_total = 25000
                    elif self.batch_large_btn.handle_event(event):
                        self.batch_total = 100000

                elif self.mode == "ai_batch":
                    if self.back_button.handle_event(event):
                        # stop and return to menu
                        self.batch_running = False
                        self.mode = "menu"
                    elif self.batch_stop_button.handle_event(event):
                        self.batch_running = False

                elif self.mode == "info":
                    if self.back_button.handle_event(event):
                        self.mode = self.prev_mode if self.prev_mode else "menu"

            # AI autoplay mode (single-hand visualized)
            if self.mode == "ai_playing" and not self.env.done:
                pygame.time.wait(800)
                state = self.env._obs()
                action = self.agent.greedy_action(state)
                self.handle_action(action)

            # Batch mode progression
            if self.mode == "ai_batch" and self.batch_running:
                self.update_batch_logic()

            # Animate cards
            for card in self.player_cards + self.dealer_cards:
                card.update()

            # Render
            self.draw()
            pygame.display.flip()

        pygame.quit()

    # ---------- Rendering ----------
    def draw(self):
        self.screen.fill(FELT_GREEN)

        # Subtle table texture
        for i in range(0, WIDTH, 40):
            pygame.draw.line(self.screen, DARK_GREEN, (i, 0), (i, HEIGHT), 1)
        for i in range(0, HEIGHT, 40):
            pygame.draw.line(self.screen, DARK_GREEN, (0, i), (WIDTH, i), 1)

        if self.mode == "menu":
            title = self.font_large.render("♠ BLACKJACK MONTE CARLO ♥", True, GOLD)
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 150))
            for button in self.menu_buttons:
                button.draw(self.screen, self.font_medium)
            return

        if self.mode == "info":
            self.draw_info()
            return

        if self.mode == "ai_batch":
            # Headings
            title = self.font_large.render("AI Batch Simulation", True, GOLD)
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))

            progress_ratio = 0 if self.batch_total == 0 else self.batch_done / self.batch_total
            progress_text = self.font_medium.render(f"{self.batch_done:,} / {self.batch_total:,} games", True, WHITE)
            self.screen.blit(progress_text, (WIDTH//2 - progress_text.get_width()//2, 80))

            # Progress bar
            bar_w, bar_h = 800, 30
            bar_x, bar_y = WIDTH//2 - bar_w//2, 130
            pygame.draw.rect(self.screen, DARK_GREEN, (bar_x, bar_y, bar_w, bar_h), border_radius=10)
            pygame.draw.rect(self.screen, GOLD, (bar_x, bar_y, int(bar_w * progress_ratio), bar_h), border_radius=10)
            pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_w, bar_h), 2, border_radius=10)

            # Relative bar chart for Wins/Losses/Pushes
            self.draw_bars(120, 200, WIDTH - 240, 420, self.batch_counts)

            # Controls
            self.back_button.draw(self.screen, self.font_small)
            self.batch_stop_button.draw(self.screen, self.font_small)

            # Status footer
            status = "RUNNING..." if self.batch_running else "FINISHED"
            status_color = WHITE if self.batch_running else (34, 255, 34)
            status_surf = self.font_medium.render(status, True, status_color)
            self.screen.blit(status_surf, (WIDTH//2 - status_surf.get_width()//2, HEIGHT - 120))

            if not self.batch_running:
                hint = self.font_small.render("Click MENU to return or RUN BATCH again from Watch AI Play.", True, LIGHT_GRAY)
                self.screen.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT - 160))
            return

        # Dealer header
        dealer_text = self.font_medium.render("♣ DEALER ♦", True, WHITE)
        self.screen.blit(dealer_text, (WIDTH//2 - dealer_text.get_width()//2, 140))

        # Dealer cards (hide hole card until reveal)
        for i, card in enumerate(self.dealer_cards):
            card.draw(self.screen, self.font_large, self.font_small, hidden=(i == 1 and not self.show_dealer_card))

        # Dealer total
        if self.show_dealer_card:
            dealer_val = hand_value(self.env.dealer)
            val_text = self.font_small.render(f"Value: {dealer_val}", True, WHITE)
            self.screen.blit(val_text, (WIDTH//2 - val_text.get_width()//2, 340))

        # Player header
        player_text = self.font_medium.render("♠ PLAYER ♥", True, WHITE)
        self.screen.blit(player_text, (WIDTH//2 - player_text.get_width()//2, PLAYER_LABEL_Y))

        # Player cards
        for card in self.player_cards:
            card.draw(self.screen, self.font_large, self.font_small)

        # Player total
        player_val = hand_value(self.env.player)
        val_text = self.font_small.render(f"Value: {player_val}", True, WHITE)
        self.screen.blit(val_text, (WIDTH//2 - val_text.get_width()//2, PLAYER_VALUE_Y))

        # Stats
        stats_text = f"W: {self.wins}  L: {self.losses}  P: {self.pushes}"
        stats_surf = self.font_small.render(stats_text, True, GOLD)
        self.screen.blit(stats_surf, (WIDTH - stats_surf.get_width() - 20, 20))

        # Result banner
        if self.result_text:
            result_color = (34, 255, 34) if "WIN" in self.result_text else RED if "LOSE" in self.result_text else WHITE
            result_surf = self.font_large.render(self.result_text, True, result_color)
            self.screen.blit(result_surf, (WIDTH//2 - result_surf.get_width()//2, HEIGHT//2 - 50))

        # AI suggestion
        if self.ai_suggestion and not self.env.done:
            ai_surf = self.font_small.render(self.ai_suggestion, True, GOLD)
            self.screen.blit(ai_surf, (WIDTH//2 - ai_surf.get_width()//2, AI_SUGGEST_Y))

        # Controls
        self.back_button.draw(self.screen, self.font_small)
        if not self.env.done and self.mode == "playing":
            self.hit_button.draw(self.screen, self.font_medium)
            self.stand_button.draw(self.screen, self.font_medium)
        if self.env.done:
            self.new_hand_button.draw(self.screen, self.font_medium)

        # Extra controls in Watch AI Play screen
        if self.mode == "ai_playing":
            self.run_batch_button.draw(self.screen, self.font_medium)

            # Show and allow quick selection of batch sizes
            bs_label = self.font_small.render("Select batch size:", True, WHITE)
            self.screen.blit(bs_label, (80, 50))
            self.batch_small_btn.draw(self.screen, self.font_small)
            self.batch_medium_btn.draw(self.screen, self.font_small)
            self.batch_large_btn.draw(self.screen, self.font_small)

            current = self.batch_total if self.batch_total > 0 else 25000
            cur_txt = self.font_small.render(f"Current selection: {current:,}", True, LIGHT_GRAY)
            self.screen.blit(cur_txt, (660, 88))

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    random.seed(1)
    game = BlackjackGame()
    game.run()
