"""
pong_game.py  — Biological Pong
─────────────────────────────────
The game runs at BIOLOGICAL TIMESCALE — one physics update per 5-minute
yeast cycle. The ball moves slowly across the screen; the paddle is driven
entirely by the yeast culture's live sensor readings fed through the trained
ridge regression readout.

Timescale design:
  - Screen = 40 cycle-widths wide
  - Ball speed = 1–2 units per cycle (configurable)
  - One full game = ~3–4 hours of real biology

The game encodes ball Y position as a dosing volume each cycle.
The yeast integrates that chemical signal over 2–3 cycles.
The ridge model reads the biological response and outputs paddle direction.
No software logic decides paddle movement — only the yeast's sensor state.

Run modes:
  python3 pong_game.py --test      # keyboard mode, fast (60fps) for UI testing
  python3 pong_game.py --demo      # auto-simulated biology, fast, for demo
  python3 pong_game.py             # real biology mode (called by controller)
"""

import pygame
import math
import time
import threading
import argparse
import json
import sys
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WIDTH,  HEIGHT   = 900, 600

# BIOLOGICAL GAME UNITS
# The game world is measured in "cycles" not pixels
# SCALE = pixels per game unit
SCALE            = 15          # 1 game unit = 15 pixels
GAME_W           = WIDTH  // SCALE   # 60 game units wide
GAME_H           = HEIGHT // SCALE   # 40 game units tall

BALL_SPEED       = 0.6         # game units per cycle
BALL_RADIUS_GU   = 1           # game units
PADDLE_H_GU      = 8           # game units (tall; biology is imprecise)
PADDLE_W_GU      = 1
PADDLE_X_GU      = GAME_W - 4  # right side
WALL_X_GU        = 2           # left wall

WIN_SCORE        = 5

# Paddle move speed per cycle — how many game units paddle shifts per cycle
# This is what was wrong before: was tiny pixels, now is whole game units
PADDLE_SPEED_GU  = 1.4         # game units per cycle

# Biology timing
BIO_CYCLE_SEC    = 300         # 5 minutes per cycle in real mode
DEMO_CYCLE_SEC   = 2.0         # 2 seconds per cycle in demo mode

# Volume→paddle mapping
# Ridge output 0.0–1.0 maps linearly to paddle position
# UP   = output > 0.55
# DOWN = output < 0.45
# HOLD = 0.45–0.55
UP_THRESHOLD   = 0.55
DOWN_THRESHOLD = 0.45

# Colours
BLACK     = (0,   0,   0  )
WHITE     = (255, 255, 255)
GREEN     = (60,  210, 110)
RED       = (220, 60,  60 )
BLUE      = (80,  150, 230)
YELLOW    = (240, 210, 40 )
GREY      = (100, 100, 100)
DARK_GREY = (30,  30,  30 )
TEAL      = (50,  190, 170)
ORANGE    = (240, 140, 40 )
PURPLE    = (150, 80,  200)


# ─────────────────────────────────────────────────────────────────────────────
# GAME STATE (in game units, not pixels)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BallState:
    x:  float = float(GAME_W // 2)
    y:  float = float(GAME_H // 2)
    vx: float = BALL_SPEED
    vy: float = BALL_SPEED * 0.7

    def reset(self):
        self.x  = float(GAME_W // 2)
        self.y  = float(GAME_H // 2)
        self.vx = BALL_SPEED
        self.vy = BALL_SPEED * 0.7

    def step(self, paddle_y_gu: float) -> str:
        """
        Advance ball one biological cycle.
        Returns: 'wall_hit' | 'paddle_hit' | 'miss' | 'bounce' | ''
        """
        self.x += self.vx
        self.y += self.vy
        event  = ''

        # Top / bottom
        if self.y - BALL_RADIUS_GU <= 0:
            self.y  = float(BALL_RADIUS_GU)
            self.vy = abs(self.vy)
            event   = 'bounce'
        elif self.y + BALL_RADIUS_GU >= GAME_H:
            self.y  = float(GAME_H - BALL_RADIUS_GU)
            self.vy = -abs(self.vy)
            event   = 'bounce'

        # Left wall
        if self.x - BALL_RADIUS_GU <= WALL_X_GU:
            self.x  = float(WALL_X_GU + BALL_RADIUS_GU)
            self.vx = abs(self.vx)
            event   = 'wall_hit'

        # Right paddle
        paddle_top    = paddle_y_gu
        paddle_bottom = paddle_y_gu + PADDLE_H_GU
        paddle_left   = float(PADDLE_X_GU)
        paddle_right  = float(PADDLE_X_GU + PADDLE_W_GU)

        if (self.vx > 0 and
                self.x + BALL_RADIUS_GU >= paddle_left and
                self.x - BALL_RADIUS_GU <= paddle_right and
                self.y + BALL_RADIUS_GU >= paddle_top and
                self.y - BALL_RADIUS_GU <= paddle_bottom):
            rel = (self.y - (paddle_top + PADDLE_H_GU/2)) / (PADDLE_H_GU/2)
            rel = max(-1.0, min(1.0, rel))
            angle  = rel * math.radians(55)
            speed  = math.hypot(self.vx, self.vy)
            self.vx = -abs(speed * math.cos(angle))
            self.vy = speed * math.sin(angle)
            self.x  = paddle_left - BALL_RADIUS_GU
            event   = 'paddle_hit'

        # Miss — ball passed paddle
        if self.x - BALL_RADIUS_GU > GAME_W:
            event = 'miss'

        return event


def set_ball_speed(speed_gu_per_cycle: float):
    global BALL_SPEED
    BALL_SPEED = max(0.1, float(speed_gu_per_cycle))


@dataclass
class PaddleState:
    """
    Paddle position is entirely driven by yeast biology.
    set_bio_output() is called once per cycle with the ridge model output.
    """
    y_gu:       float = float(GAME_H // 2 - PADDLE_H_GU // 2)
    bio_output: float = 0.5       # latest ridge prediction (0–1)
    last_pump:  str   = 'none'    # 'media' | 'salt' | 'none'
    _lock:      object = field(default_factory=threading.Lock, repr=False)

    def set_bio_output(self, value: float, pump: str = 'none'):
        with self._lock:
            self.bio_output = max(0.0, min(1.0, float(value)))
            self.last_pump  = pump

    def update(self):
        """Move paddle based on current bio_output. Called once per cycle."""
        with self._lock:
            val = self.bio_output
        if val > UP_THRESHOLD:
            # Move UP (decrease y in screen coords)
            move = PADDLE_SPEED_GU * (val - UP_THRESHOLD) / (1.0 - UP_THRESHOLD)
            self.y_gu -= move
        elif val < DOWN_THRESHOLD:
            # Move DOWN
            move = PADDLE_SPEED_GU * (DOWN_THRESHOLD - val) / DOWN_THRESHOLD
            self.y_gu += move
        # Clamp
        self.y_gu = max(0.0, min(float(GAME_H - PADDLE_H_GU), self.y_gu))

    def get_output(self) -> float:
        with self._lock:
            return self.bio_output


@dataclass
class GameState:
    ball:          BallState   = field(default_factory=BallState)
    paddle:        PaddleState = field(default_factory=PaddleState)
    score_yeast:   int         = 0
    score_wall:    int         = 0
    cycle:         int         = 0
    game_over:     bool        = False
    winner:        str         = ''
    event_log:     list        = field(default_factory=list)

    # Shared state for controller thread
    current_pump:  str         = 'none'   # what pump fired this cycle
    pump_volume:   float       = 0.0      # volume fired
    waiting_bio:   bool        = False    # True while waiting for yeast response
    controller_status: str      = 'not started'
    missing_features: list      = field(default_factory=list)
    last_prediction: Optional[float] = None
    last_sensor_update: Optional[float] = None

    def reset(self):
        self.ball = BallState()
        self.ball.reset()
        self.paddle = PaddleState()
        self.score_yeast = 0
        self.score_wall = 0
        self.cycle = 0
        self.game_over = False
        self.winner = ''
        self.event_log.clear()
        self.current_pump = 'none'
        self.pump_volume = 0.0
        self.waiting_bio = False

    def encode_ball_to_pump(self) -> Tuple[str, float]:
        """
        Convert current ball Y to a pump action.
        This is the ONLY place ball position affects biology.
        Ball high (y < GAME_H/2) → media pump → yeast grows UP
        Ball low  (y > GAME_H/2) → salt pump  → yeast stressed DOWN

        Volume scales with distance from centre — stronger signal when
        ball is further from centre.
        """
        centre = GAME_H / 2
        dist   = (self.ball.y - centre) / centre   # -1 to +1
        # dist < 0 = ball in upper half = pump media
        # dist > 0 = ball in lower half = pump salt
        vol = 0.3 + abs(dist) * 0.6   # 0.3–0.9 mL, stronger near edges
        vol = round(max(0.1, min(0.9, vol)), 2)
        pump = 'media' if dist < 0 else 'salt'
        return pump, vol

    def step_cycle(self):
        """Advance game state one biological cycle."""
        self.cycle += 1
        self.paddle.update()
        event = self.ball.step(self.paddle.y_gu)

        if event == 'miss':
            self.score_wall += 1
            self.ball.reset()
            self.event_log.append({'cycle':self.cycle,'event':'miss','score_wall':self.score_wall})
            if self.score_wall >= WIN_SCORE:
                self.game_over = True; self.winner = "WALL"
        elif event == 'paddle_hit':
            self.score_yeast += 1
            self.event_log.append({'cycle':self.cycle,'event':'hit','score_yeast':self.score_yeast})
            if self.score_yeast >= WIN_SCORE:
                self.game_over = True; self.winner = "YEAST 🧫"

    def save_log(self, path='pong_game_log.json'):
        with open(path,'w') as f:
            json.dump(self.event_log, f, indent=2)
        print(f"Game log saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def gu_to_px(x_gu, y_gu):
    return int(x_gu * SCALE), int(y_gu * SCALE)

class Renderer:
    def __init__(self, screen):
        self.screen  = screen
        self.font_xl = pygame.font.SysFont('monospace', 56, bold=True)
        self.font_lg = pygame.font.SysFont('monospace', 28, bold=True)
        self.font_sm = pygame.font.SysFont('monospace', 16)
        self.font_xs = pygame.font.SysFont('monospace', 13)
        self._trail  = []   # ball trail positions

    def draw(self, state: GameState, mode_label: str = '',
             cycle_progress: float = 0.0):
        self.screen.fill(BLACK)

        # ── Left wall ─────────────────────────────────────────────────────────
        wx, _ = gu_to_px(WALL_X_GU, 0)
        pygame.draw.rect(self.screen, GREY, (0, 0, wx + SCALE, HEIGHT))
        pygame.draw.line(self.screen, (60,60,60), (wx+SCALE, 0), (wx+SCALE, HEIGHT), 1)

        # ── Centre dashed divider ─────────────────────────────────────────────
        cx = WIDTH // 2
        for y in range(0, HEIGHT, 26):
            pygame.draw.rect(self.screen, DARK_GREY, (cx-1, y, 2, 14))

        # ── Ball trail ────────────────────────────────────────────────────────
        self._trail.append((state.ball.x, state.ball.y))
        if len(self._trail) > 8:
            self._trail.pop(0)
        for i, (tx,ty) in enumerate(self._trail[:-1]):
            alpha = int(180 * (i+1)/len(self._trail))
            bx, by = gu_to_px(tx, ty)
            s = pygame.Surface((BALL_RADIUS_GU*SCALE*2, BALL_RADIUS_GU*SCALE*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255,255,255,alpha),
                               (BALL_RADIUS_GU*SCALE, BALL_RADIUS_GU*SCALE), BALL_RADIUS_GU*SCALE)
            self.screen.blit(s, (bx-BALL_RADIUS_GU*SCALE, by-BALL_RADIUS_GU*SCALE))

        # ── Ball ─────────────────────────────────────────────────────────────
        bx, by = gu_to_px(state.ball.x, state.ball.y)
        pygame.draw.circle(self.screen, WHITE, (bx, by), BALL_RADIUS_GU*SCALE)
        pygame.draw.circle(self.screen, BLUE,  (bx, by), BALL_RADIUS_GU*SCALE, 2)

        # ── Paddle ────────────────────────────────────────────────────────────
        px, py = gu_to_px(PADDLE_X_GU, state.paddle.y_gu)
        pw     = PADDLE_W_GU * SCALE
        ph     = PADDLE_H_GU * SCALE
        sig    = state.paddle.get_output()
        # Colour: green=media/up, red=salt/down, white=neutral
        if sig > UP_THRESHOLD:
            pcol = (int(60+195*(sig-UP_THRESHOLD)/(1-UP_THRESHOLD)),
                    int(180+75*(sig-UP_THRESHOLD)/(1-UP_THRESHOLD)), 80)
        elif sig < DOWN_THRESHOLD:
            s2 = (DOWN_THRESHOLD-sig)/DOWN_THRESHOLD
            pcol = (int(180+75*s2), int(60+60*(1-s2)), 60)
        else:
            pcol = WHITE
        pygame.draw.rect(self.screen, pcol, (px, py, pw, ph), border_radius=5)
        pygame.draw.rect(self.screen, WHITE, (px, py, pw, ph), 2, border_radius=5)

        # ── Scores ───────────────────────────────────────────────────────────
        sc_txt = self.font_xl.render(
            f"{state.score_wall}   {state.score_yeast}", True, WHITE)
        self.screen.blit(sc_txt, (WIDTH//2 - sc_txt.get_width()//2, 8))
        w_lbl = self.font_xs.render("WALL", True, GREY)
        y_lbl = self.font_xs.render("YEAST", True, TEAL)
        self.screen.blit(w_lbl, (WIDTH//2 - 62, 64))
        self.screen.blit(y_lbl, (WIDTH//2 + 26, 64))

        # ── Bio signal bar (right edge) ───────────────────────────────────────
        bx2  = WIDTH - 32
        bh   = 200
        by2  = HEIGHT//2 - bh//2
        pygame.draw.rect(self.screen, DARK_GREY, (bx2, by2, 20, bh), border_radius=4)
        fill = int(bh * sig)
        fy   = by2 + bh - fill
        if fill > 0:
            fc = (int(220*(1-sig)), int(200*sig), 60)
            pygame.draw.rect(self.screen, fc, (bx2, fy, 20, fill), border_radius=4)
        pygame.draw.rect(self.screen, WHITE, (bx2, by2, 20, bh), 1, border_radius=4)
        # Labels
        self.screen.blit(self.font_xs.render(f"{sig:.2f}", True, WHITE),
                         (bx2-4, by2+bh+4))
        self.screen.blit(self.font_xs.render("▲media", True, GREEN), (bx2-22, by2-18))
        self.screen.blit(self.font_xs.render("▼salt",  True, RED),   (bx2-22, by2+bh+20))

        # ── Cycle progress bar (bottom) ───────────────────────────────────────
        pg_w = int((WIDTH - 120) * cycle_progress)
        pygame.draw.rect(self.screen, DARK_GREY, (60, HEIGHT-18, WIDTH-120, 8), border_radius=4)
        if pg_w > 0:
            pygame.draw.rect(self.screen, TEAL, (60, HEIGHT-18, pg_w, 8), border_radius=4)
        cyc_lbl = self.font_xs.render(
            f"cycle {state.cycle}  |  pump: {state.current_pump} {state.pump_volume:.2f}mL  |  {mode_label}",
            True, GREY)
        self.screen.blit(cyc_lbl, (60, HEIGHT-34))
        if mode_label == 'BIOLOGY MODE':
            missing = getattr(state, 'missing_features', [])
            status = getattr(state, 'controller_status', '')
            pred = getattr(state, 'last_prediction', None)
            if pred is not None and 'ridge=' not in status:
                status = f"{status} | ridge={pred:.3f}"
            if missing:
                status = f"{status} | missing: {', '.join(missing[:4])}"
                if len(missing) > 4:
                    status += f" +{len(missing) - 4}"
            st_lbl = self.font_xs.render(status[:110], True, GREY)
            self.screen.blit(st_lbl, (60, HEIGHT-50))

        # ── Waiting indicator ─────────────────────────────────────────────────
        if state.waiting_bio:
            dots = '.' * (int(time.time()*2) % 4)
            wt = self.font_sm.render(f"waiting for yeast{dots}", True, YELLOW)
            self.screen.blit(wt, (WIDTH//2 - wt.get_width()//2, HEIGHT//2 + 80))

        # ── Game over overlay ─────────────────────────────────────────────────
        if state.game_over:
            ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            ov.fill((0,0,0,170))
            self.screen.blit(ov, (0,0))
            wt = self.font_xl.render(f"{state.winner} WINS!", True,
                                     TEAL if 'YEAST' in state.winner else GREY)
            st = self.font_sm.render("Q to quit  |  R to restart", True, WHITE)
            self.screen.blit(wt, (WIDTH//2-wt.get_width()//2, HEIGHT//2-50))
            self.screen.blit(st, (WIDTH//2-st.get_width()//2, HEIGHT//2+20))

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_game(mode: str = 'biology', ball_speed: Optional[float] = None,
             cycle_seconds: Optional[float] = None, dry_run: bool = False,
             start_sensors: bool = False):
    """
    mode: 'test'    — keyboard control, fast 60fps
          'demo'    — simulated biology, fast (2s/cycle)
          'biology' — real yeast, 5 min/cycle
    """
    if ball_speed is not None:
        set_ball_speed(ball_speed)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Yeast Pong 🧫")
    clock  = pygame.time.Clock()

    renderer = Renderer(screen)
    state    = GameState()
    state.ball.reset()

    CYCLE_DUR = {'test': 0, 'demo': DEMO_CYCLE_SEC, 'biology': BIO_CYCLE_SEC}[mode]
    if cycle_seconds is not None and mode != 'test':
        CYCLE_DUR = cycle_seconds
    FPS_DRAW  = 60

    mode_label = {
        'test': 'KEYBOARD MODE',
        'demo': 'DEMO MODE',
        'biology': 'BIOLOGY MODE'
    }[mode]

    print(f"Yeast Pong — {mode_label}")
    print(f"Win condition: {WIN_SCORE} points | cycle duration: {CYCLE_DUR}s")

    cycle_start = time.time()
    last_draw   = time.time()
    cycle_progress = 0.0

    # ✅ START CONTROLLER THREAD ONCE
    if mode == 'biology':
        if dry_run:
            import os
            os.environ["PONG_DRY_RUN"] = "1"

        from controller import run_controller, start_sensor_jobs

        if start_sensors:
            try:
                start_sensor_jobs()
                state.controller_status = 'sensor jobs requested'
            except Exception as exc:
                state.controller_status = f"sensor start failed: {exc}"
                print(state.controller_status)

        threading.Thread(
            target=run_controller,
            args=(state,),
            daemon=True
        ).start()

    running = True
    while running:
        now = time.time()

        # ── EVENTS ─────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    running = False
                elif ev.key == pygame.K_r and state.game_over:
                    state.reset()
                    cycle_start = now

        if state.game_over:
            renderer.draw(state, mode_label, 1.0)
            clock.tick(FPS_DRAW)
            continue

        # ── TEST MODE ─────────────────────────────────────────
        if mode == 'test':
            keys = pygame.key.get_pressed()
            cur  = state.paddle.get_output()

            if keys[pygame.K_UP]:
                state.paddle.set_bio_output(min(1.0, cur + 0.04))
            elif keys[pygame.K_DOWN]:
                state.paddle.set_bio_output(max(0.0, cur - 0.04))

            state.step_cycle()
            cycle_progress = 1.0

        # ── DEMO / BIOLOGY MODE ───────────────────────────────
        else:
            cycle_progress = min(1.0, (now - cycle_start) / CYCLE_DUR) if CYCLE_DUR > 0 else 1.0

            if cycle_progress >= 1.0:

                # Encode ball → pump
                pump, vol = state.encode_ball_to_pump()
                state.current_pump = pump
                state.pump_volume  = vol

                # ── SEND TO PIOREACTOR (BIOLOGY MODE ONLY)
                if mode == 'biology':
                    try:
                        from controller import dose_once
                        dose_once(pump, vol, remove_waste=True)
                    except Exception as exc:
                        state.controller_status = f"dosing failed: {exc}"
                        print(state.controller_status)

                # ── DEMO MODE (SIMULATED BIOLOGY)
                if mode == 'demo':
                    cur = state.paddle.get_output()

                    if pump == 'media':
                        noise = 0.05 * (2*random.random() - 1)
                        new_sig = min(1.0, cur + 0.25 + noise)
                    else:
                        noise = 0.05 * (2*random.random() - 1)
                        new_sig = max(0.0, cur - 0.25 + noise)

                    state.paddle.set_bio_output(new_sig, pump)

                # Step physics
                state.step_cycle()
                cycle_start = now
                state.waiting_bio = (mode == 'biology')

        # ── DRAW (always 60 FPS) ──────────────────────────────
        if now - last_draw >= 1.0 / FPS_DRAW:
            renderer.draw(state, mode_label, cycle_progress)
            last_draw = now

        clock.tick(FPS_DRAW)

    state.save_log()
    pygame.quit()

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yeast Pong')
    parser.add_argument('--test',  action='store_true', help='Keyboard control (fast)')
    parser.add_argument('--demo',  action='store_true', help='Simulated biology (fast)')
    parser.add_argument('--ball-speed', type=float, default=None,
                        help='Ball speed in game units per biological cycle (default: 0.6)')
    parser.add_argument('--cycle-seconds', type=float, default=None,
                        help='Override demo/biology cycle duration for testing')
    parser.add_argument('--dry-run', action='store_true',
                        help='In biology mode, print pump commands without running pumps')
    parser.add_argument('--start-sensors', action='store_true',
                        help='Start Pioreactor stirring, OD, growth-rate, and spectrometer jobs before the game')
    args = parser.parse_args()

    if args.test:
        run_game('test', ball_speed=args.ball_speed)
    elif args.demo:
        run_game('demo', ball_speed=args.ball_speed, cycle_seconds=args.cycle_seconds)
    else:
        run_game('biology', ball_speed=args.ball_speed,
                 cycle_seconds=args.cycle_seconds, dry_run=args.dry_run,
                 start_sensors=args.start_sensors)
