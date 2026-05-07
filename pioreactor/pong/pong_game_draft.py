"""
pong_game.py
─────────────
Yeast-controlled Pong game.

The ball bounces between a fixed left wall and a right paddle.
The paddle position is controlled entirely by the yeast culture's
biological response — fed in via set_paddle_signal(value) every 5 minutes.

Run standalone (keyboard test):
    python3 pong_game.py --test

Run with yeast controller (no keyboard):
    python3 pong_game.py

Controls in test mode:
    UP/DOWN arrows → manually override paddle (for testing)
    Q              → quit
    R              → reset
"""

import pygame
import math
import time
import threading
import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WIDTH,  HEIGHT   = 800, 600
FPS              = 60
BALL_SPEED_INIT  = 4.0        # pixels per frame
BALL_SPEED_MAX   = 12.0
BALL_SPEED_INC   = 0.3        # speed increase per paddle hit
BALL_RADIUS      = 10

PADDLE_W         = 16
PADDLE_H         = 90
PADDLE_X         = WIDTH - 60  # right side — yeast controls this
WALL_X           = 40          # left wall (ball bounces off)
WALL_W           = 20

# How fast the paddle drifts toward its target (pixels per frame)
# Biological timescale: yeast updates every ~5 min = ~18000 frames
# Paddle drifts 0.15 px/frame = 2700 px over 5 min — enough to cover full screen
PADDLE_DRIFT_SPEED = 0.15

# Biological update interval in seconds
# In test mode this is overridden by keyboard
BIO_UPDATE_INTERVAL = 300   # 5 minutes

# Lookahead for ball position prediction (in seconds)
LOOKAHEAD_SEC    = 600   # 10 minutes

# Score limits
WIN_SCORE        = 7

# Colours
BLACK     = (0,   0,   0  )
WHITE     = (255, 255, 255)
GREEN     = (80,  200, 120)
RED       = (220, 60,  60 )
BLUE      = (80,  140, 220)
YELLOW    = (240, 210, 50 )
GREY      = (120, 120, 120)
DARK_GREY = (40,  40,  40 )
TEAL      = (60,  200, 180)


# ─────────────────────────────────────────────────────────────────────────────
# BALL PHYSICS + LOOKAHEAD PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Ball:
    x: float = WIDTH / 2
    y: float = HEIGHT / 2
    vx: float = BALL_SPEED_INIT
    vy: float = BALL_SPEED_INIT * 0.6
    radius: int = BALL_RADIUS
    speed: float = BALL_SPEED_INIT

    def reset(self):
        self.x  = WIDTH / 2
        self.y  = HEIGHT / 2
        self.vx = BALL_SPEED_INIT
        self.vy = BALL_SPEED_INIT * 0.6
        self.speed = BALL_SPEED_INIT

    def update(self, paddle_rect: pygame.Rect) -> str:
        """
        Move ball one frame. Returns event string:
          'wall_hit'    — ball hit the left wall
          'paddle_hit'  — ball hit the yeast paddle
          'miss'        — ball passed the paddle (score for wall)
          'top_bottom'  — ball hit top/bottom wall
          ''            — normal frame
        """
        self.x += self.vx
        self.y += self.vy
        event = ''

        # Top / bottom bounce
        if self.y - self.radius <= 0:
            self.y  = self.radius
            self.vy = abs(self.vy)
            event = 'top_bottom'
        elif self.y + self.radius >= HEIGHT:
            self.y  = HEIGHT - self.radius
            self.vy = -abs(self.vy)
            event = 'top_bottom'

        # Left wall bounce
        if self.x - self.radius <= WALL_X + WALL_W:
            self.x  = WALL_X + WALL_W + self.radius
            self.vx = abs(self.vx)
            event = 'wall_hit'

        # Right paddle collision
        if (self.vx > 0 and
                self.x + self.radius >= paddle_rect.left and
                self.x - self.radius <= paddle_rect.right and
                self.y + self.radius >= paddle_rect.top and
                self.y - self.radius <= paddle_rect.bottom):
            # Angle depends on where on paddle the ball hits
            relative_y = (self.y - paddle_rect.centery) / (PADDLE_H / 2)
            relative_y = max(-1.0, min(1.0, relative_y))
            bounce_angle = relative_y * math.radians(60)
            self.speed = min(self.speed + BALL_SPEED_INC, BALL_SPEED_MAX)
            self.vx = -abs(self.speed * math.cos(bounce_angle))
            self.vy = self.speed * math.sin(bounce_angle)
            self.x  = paddle_rect.left - self.radius
            event = 'paddle_hit'

        # Ball passed paddle — wall scores
        if self.x - self.radius > WIDTH:
            event = 'miss'

        return event

    def predict_y_after(self, seconds: float, fps: int = FPS,
                        paddle_rect: Optional[pygame.Rect] = None) -> float:
        """
        Simulate ball physics forward `seconds` in time.
        Returns predicted Y position of ball (ignores paddle — ball passes through).
        Used to decide pump direction before yeast responds.
        """
        sx, sy   = self.x,  self.y
        svx, svy = self.vx, self.vy
        frames   = int(seconds * fps)

        for _ in range(frames):
            sx += svx
            sy += svy
            # Top/bottom bounce
            if sy - self.radius <= 0:
                sy  = self.radius
                svy = abs(svy)
            elif sy + self.radius >= HEIGHT:
                sy  = HEIGHT - self.radius
                svy = -abs(svy)
            # Left wall bounce
            if sx - self.radius <= WALL_X + WALL_W:
                sx  = WALL_X + WALL_W + self.radius
                svx = abs(svx)
            # Right wall bounce (treat right edge as wall for prediction)
            if sx + self.radius >= WIDTH:
                sx  = WIDTH - self.radius
                svx = -abs(svx)

        return float(sy)


# ─────────────────────────────────────────────────────────────────────────────
# PADDLE — controlled by yeast biology
# ─────────────────────────────────────────────────────────────────────────────

class YeastPaddle:
    """
    The paddle's Y position is driven entirely by the yeast culture's
    biological response signal (0.0–1.0).

    signal > 0.55  →  paddle target moves up
    signal < 0.45  →  paddle target moves down
    0.45–0.55      →  paddle holds position (neutral zone)

    The paddle drifts toward its target at PADDLE_DRIFT_SPEED px/frame
    to simulate smooth biological response.
    """

    def __init__(self):
        self.y           = HEIGHT / 2 - PADDLE_H / 2   # top-left Y
        self.target_y    = self.y
        self.signal      = 0.5     # current biological signal (0=full down, 1=full up)
        self.last_update = time.time()
        self._lock       = threading.Lock()

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(PADDLE_X, int(self.y), PADDLE_W, PADDLE_H)

    def set_signal(self, value: float):
        """
        Called by the yeast controller every 5 minutes.
        value: float in [0.0, 1.0]
          1.0 = full up response (media pump dominant)
          0.0 = full down response (salt pump dominant)
          0.5 = neutral
        """
        with self._lock:
            self.signal      = max(0.0, min(1.0, float(value)))
            self.last_update = time.time()
            # Map signal to target Y position
            # signal=1.0 → target_y = 0 (top)
            # signal=0.0 → target_y = HEIGHT-PADDLE_H (bottom)
            # signal=0.5 → target_y = HEIGHT/2 - PADDLE_H/2 (centre)
            self.target_y = (1.0 - self.signal) * (HEIGHT - PADDLE_H)

    def update(self):
        """Drift paddle toward target at biological speed."""
        with self._lock:
            diff = self.target_y - self.y
            if abs(diff) > PADDLE_DRIFT_SPEED:
                self.y += PADDLE_DRIFT_SPEED * (1 if diff > 0 else -1)
            else:
                self.y = self.target_y
            # Clamp to screen
            self.y = max(0, min(HEIGHT - PADDLE_H, self.y))

    def get_signal(self) -> float:
        with self._lock:
            return self.signal


# ─────────────────────────────────────────────────────────────────────────────
# GAME STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    ball:             Ball        = field(default_factory=Ball)
    paddle:           YeastPaddle = field(default_factory=YeastPaddle)
    score_yeast:      int         = 0   # yeast (paddle) scored
    score_wall:       int         = 0   # wall scored (ball passed paddle)
    hits:             int         = 0   # total paddle hits
    running:          bool        = True
    paused:           bool        = False
    game_over:        bool        = False
    winner:           str         = ''
    # Biological decision log — for analysis after the game
    bio_log: list = field(default_factory=list)

    def log_bio_event(self, signal: float, pump: str, ball_y: float,
                      pred_y: float, paddle_y: float):
        self.bio_log.append({
            'timestamp':  time.time(),
            'signal':     signal,
            'pump':       pump,
            'ball_y':     ball_y,
            'predicted_y':pred_y,
            'paddle_y':   paddle_y,
        })

    def save_log(self, path: str = 'pong_bio_log.json'):
        with open(path, 'w') as f:
            json.dump(self.bio_log, f, indent=2)
        print(f"Bio log saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# RENDERER
# ─────────────────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, screen: pygame.Surface, font_lg, font_sm, font_xs):
        self.screen   = screen
        self.font_lg  = font_lg
        self.font_sm  = font_sm
        self.font_xs  = font_xs

    def draw(self, state: GameState, test_mode: bool = False,
             predicted_y: Optional[float] = None):
        self.screen.fill(BLACK)

        # ── Court ────────────────────────────────────────────────────────────
        # Left wall
        pygame.draw.rect(self.screen, GREY,
                         (WALL_X, 0, WALL_W, HEIGHT))
        pygame.draw.line(self.screen, DARK_GREY,
                         (WALL_X+WALL_W, 0), (WALL_X+WALL_W, HEIGHT), 1)

        # Centre dashed line
        for y in range(0, HEIGHT, 24):
            pygame.draw.rect(self.screen, DARK_GREY, (WIDTH//2-1, y, 2, 12))

        # ── Paddle ────────────────────────────────────────────────────────────
        # Colour reflects biological signal
        sig   = state.paddle.get_signal()
        r_val = int(220 * (1 - sig))
        g_val = int(200 * sig)
        paddle_color = (r_val, g_val, 60)
        pygame.draw.rect(self.screen, paddle_color, state.paddle.rect, border_radius=6)
        pygame.draw.rect(self.screen, WHITE, state.paddle.rect, 2, border_radius=6)

        # ── Ball ─────────────────────────────────────────────────────────────
        bx, by = int(state.ball.x), int(state.ball.y)
        pygame.draw.circle(self.screen, WHITE, (bx, by), BALL_RADIUS)
        # Trail
        for i, alpha in enumerate([0.3, 0.2, 0.1]):
            trail_x = int(state.ball.x - state.ball.vx*(i+1)*2)
            trail_y = int(state.ball.y - state.ball.vy*(i+1)*2)
            s = pygame.Surface((BALL_RADIUS*2, BALL_RADIUS*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*WHITE, int(255*alpha)),
                               (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
            self.screen.blit(s, (trail_x-BALL_RADIUS, trail_y-BALL_RADIUS))

        # ── Predicted ball Y (lookahead marker) ──────────────────────────────
        if predicted_y is not None:
            py = int(predicted_y)
            pygame.draw.line(self.screen, YELLOW,
                             (PADDLE_X - 20, py), (PADDLE_X + PADDLE_W + 10, py), 2)
            label = self.font_xs.render("pred", True, YELLOW)
            self.screen.blit(label, (PADDLE_X - 40, py - 8))

        # ── Scores ───────────────────────────────────────────────────────────
        score_text = self.font_lg.render(
            f"{state.score_wall}   {state.score_yeast}", True, WHITE)
        self.screen.blit(score_text,
                         (WIDTH//2 - score_text.get_width()//2, 12))

        # Labels
        wall_lbl  = self.font_xs.render("WALL", True, GREY)
        yeast_lbl = self.font_xs.render("YEAST", True, TEAL)
        self.screen.blit(wall_lbl,  (WIDTH//2 - 70, 48))
        self.screen.blit(yeast_lbl, (WIDTH//2 + 30, 48))

        # ── Bio signal bar ────────────────────────────────────────────────────
        bar_h   = 160
        bar_x   = WIDTH - 24
        bar_y   = HEIGHT//2 - bar_h//2
        pygame.draw.rect(self.screen, DARK_GREY, (bar_x, bar_y, 16, bar_h), border_radius=4)
        fill_h  = int(bar_h * sig)
        fill_y  = bar_y + bar_h - fill_h
        fill_c  = (int(220*(1-sig)), int(200*sig), 60)
        if fill_h > 0:
            pygame.draw.rect(self.screen, fill_c,
                             (bar_x, fill_y, 16, fill_h), border_radius=4)
        sig_lbl = self.font_xs.render(f"{sig:.2f}", True, WHITE)
        self.screen.blit(sig_lbl, (bar_x - 4, bar_y + bar_h + 4))
        up_lbl  = self.font_xs.render("▲media", True, GREEN)
        dn_lbl  = self.font_xs.render("▼salt",  True, RED)
        self.screen.blit(up_lbl, (bar_x - 16, bar_y - 16))
        self.screen.blit(dn_lbl, (bar_x - 16, bar_y + bar_h + 18))

        # ── Speed indicator ───────────────────────────────────────────────────
        spd_lbl = self.font_xs.render(
            f"speed: {state.ball.speed:.1f}  hits: {state.hits}", True, GREY)
        self.screen.blit(spd_lbl, (WALL_X + WALL_W + 8, HEIGHT - 22))

        # ── Test mode indicator ───────────────────────────────────────────────
        if test_mode:
            tm = self.font_xs.render("TEST MODE — UP/DOWN to override paddle", True, YELLOW)
            self.screen.blit(tm, (WALL_X + WALL_W + 8, 8))

        # ── Game over ─────────────────────────────────────────────────────────
        if state.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))
            win_text = self.font_lg.render(
                f"{state.winner} WINS!", True,
                TEAL if 'YEAST' in state.winner else GREY)
            sub_text = self.font_sm.render("Press R to restart  |  Q to quit", True, WHITE)
            self.screen.blit(win_text,
                             (WIDTH//2 - win_text.get_width()//2, HEIGHT//2 - 40))
            self.screen.blit(sub_text,
                             (WIDTH//2 - sub_text.get_width()//2, HEIGHT//2 + 20))

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GAME LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_game(test_mode: bool = False):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Yeast Pong 🧫")

    font_lg = pygame.font.SysFont('monospace', 52, bold=True)
    font_sm = pygame.font.SysFont('monospace', 22)
    font_xs = pygame.font.SysFont('monospace', 14)
    clock   = pygame.time.Clock()

    renderer = Renderer(screen, font_lg, font_sm, font_xs)
    state    = GameState()
    state.ball.reset()

    predicted_y: Optional[float] = None
    last_predict_time = 0.0

    print("Pong started.")
    if test_mode:
        print("TEST MODE: UP/DOWN keys control paddle. Q=quit, R=reset.")
    else:
        print("BIOLOGY MODE: Waiting for yeast controller signals...")
        print("  Call state.paddle.set_signal(value) from controller thread.")

    running = True
    while running:
        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    state.ball.reset()
                    state.score_yeast = 0
                    state.score_wall  = 0
                    state.hits        = 0
                    state.game_over   = False
                    state.paddle.set_signal(0.5)

        # Test mode — keyboard override
        if test_mode and not state.game_over:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                state.paddle.set_signal(min(1.0, state.paddle.get_signal() + 0.02))
            elif keys[pygame.K_DOWN]:
                state.paddle.set_signal(max(0.0, state.paddle.get_signal() - 0.02))

        # ── Physics update ────────────────────────────────────────────────────
        if not state.game_over:
            state.paddle.update()
            ball_event = state.ball.update(state.paddle.rect)

            if ball_event == 'paddle_hit':
                state.hits += 1
            elif ball_event == 'miss':
                state.score_wall += 1
                state.ball.reset()
                if state.score_wall >= WIN_SCORE:
                    state.game_over = True
                    state.winner    = "WALL"
            # Yeast scores when ball makes a complete lap (wall→paddle→wall)
            # We count this as a successful return — tracked via hits
            if state.hits > 0:
                if ball_event == 'wall_hit' and state.hits % 5 == 0:
                    state.score_yeast += 1
                    if state.score_yeast >= WIN_SCORE:
                        state.game_over = True
                        state.winner    = "YEAST 🧫"

        # ── Lookahead prediction (every second) ───────────────────────────────
        now = time.time()
        if now - last_predict_time > 1.0:
            predicted_y      = state.ball.predict_y_after(LOOKAHEAD_SEC)
            last_predict_time = now

        # ── Render ────────────────────────────────────────────────────────────
        renderer.draw(state, test_mode=test_mode, predicted_y=predicted_y)
        clock.tick(FPS)

    state.save_log()
    pygame.quit()
    return state


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yeast Pong')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: control paddle with keyboard arrows')
    args = parser.parse_args()
    run_game(test_mode=args.test)