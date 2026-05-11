"""
Two tiny, fully-tabular "hard exploration" environments shared by every
script in this folder.

Why these two?
--------------
1. DeepSeaEnv          -- the canonical chain task used to study *deep*
                          exploration (bsuite, "Deep Exploration via
                          Bootstrapped DQN").  A greedy / epsilon-greedy
                          agent provably almost never finds the reward;
                          systematic exploration solves it instantly.
2. MiniMontezumaEnv    -- a two-room gridworld that mimics the *structure*
                          of Montezuma's Revenge's first screen: walk to a
                          key, pick it up, then the door opens, then walk
                          to the treasure.  Reward is given ONLY at the
                          treasure -- a long chain of correct moves with
                          zero feedback along the way.

Both are deterministic, have a handful of discrete states, and expose a
minimal Gym-like API:

    env = DeepSeaEnv(size=10)
    s = env.reset()                 # s is an int in [0, env.n_states)
    s2, r, done, info = env.step(a) # a is an int in [0, env.n_actions)

Keeping them tabular means you can SEE exactly what exploration is doing
(state-visitation heatmaps, Q-tables) without any neural-network noise.
The lessons -- "epsilon-greedy can't do deep exploration", "an intrinsic
novelty bonus fixes it" -- are exactly the ones you would learn on pixel
Montezuma, just 100,000x faster.
"""

import numpy as np


# ===========================================================================
# 1. DeepSea : the textbook hard-exploration chain
# ===========================================================================
class DeepSeaEnv:
    """
    An N x N grid.  The agent starts in the top-left cell.  On every step it
    descends one row and chooses a column move:

        action 0 = move LEFT  (column -= 1, clamped at 0)      -- "free"
        action 1 = move RIGHT (column += 1, clamped at N-1)    -- tiny cost

    The episode is exactly N steps long (you fall off the bottom).  If the
    final cell is the bottom-RIGHT corner you get reward +1.  Every RIGHT
    action costs -0.01 / N, so the all-right path nets +1 - 0.01 = +0.99
    while the lazy all-left path nets exactly 0.

    The trap: the only positive reward is hidden behind N *consecutive*
    correct RIGHT choices, and each of those choices has a small *immediate*
    penalty.  A myopic agent learns "RIGHT looks bad" and goes left forever.
    The chance a uniform-random policy reaches the corner is 2^-N -- about
    one in a thousand at N=10, one in a million at N=20.  This is the
    cleanest demonstration of why "act randomly sometimes" (epsilon-greedy)
    is NOT real exploration.

    State encoding: state = row * size + col, an int in [0, size*size).
    There is one extra absorbing state with index size*size that the env
    moves to when the N-step fall is over (so every state index -- including
    the terminal one -- is a valid Q-table / count-table row).
    """

    n_actions = 2  # 0 = left, 1 = right

    def __init__(self, size=10, move_cost=0.01):
        self.size = size
        self.terminal_state = size * size
        self.n_states = size * size + 1       # +1 for the absorbing state
        self.move_cost = move_cost / size
        self.row = 0
        self.col = 0

    # --- helpers ---------------------------------------------------------
    def _state(self):
        return self.row * self.size + self.col

    @staticmethod
    def decode(state, size):
        """Turn an int state back into (row, col) -- handy for heatmaps."""
        return divmod(state, size)

    # --- Gym-like API ----------------------------------------------------
    def reset(self):
        self.row, self.col = 0, 0
        return self._state()

    def step(self, action):
        assert action in (0, 1)
        reward = 0.0
        if action == 1:                       # RIGHT: small immediate cost
            self.col = min(self.col + 1, self.size - 1)
            reward -= self.move_cost
        else:                                 # LEFT: free
            self.col = max(self.col - 1, 0)
        self.row += 1

        done = self.row >= self.size
        if done:
            if self.col == self.size - 1:       # reached the bottom-right
                reward += 1.0
            return self.terminal_state, reward, True, {}
        return self._state(), reward, done, {}


# ===========================================================================
# 2. MiniMontezuma : key -> door -> treasure, reward only at the end
# ===========================================================================
#
# Layout (15 columns x 7 rows). Column 7 is a solid wall that splits the map
# into a LEFT room and a RIGHT room, joined only by a door 'D'.  The door is
# passable *only while the agent is carrying the key*.
#
#        col: 0123456789...
#   row 0:   ###############
#   row 1:   #S....#.......#
#   row 2:   #.....#.......#
#   row 3:   #.....#...T...#     T = treasure (the ONLY rewarding tile)
#   row 4:   #.....D.......#     D = door (needs key), in the dividing wall
#   row 5:   #..K..#.......#     K = key
#   row 6:   ###############
#
# Shortest solution: S(1,1) -> K(5,3) [grab key] -> D(4,6) [door opens]
# -> T(3,10).  That is ~15 perfect moves with absolutely no feedback until
# the very last step.  A vanilla epsilon-greedy Q-learner gets reward 0 run
# after run; an intrinsic-novelty bonus walks it through the chain.
#
_MONTEZUMA_MAP = [
    "###############",
    "#S....#.......#",
    "#.....#.......#",
    "#.....#...T...#",
    "#.....D.......#",
    "#..K..#.......#",
    "###############",
]


class MiniMontezumaEnv:
    """
    Tabular gridworld with a key/door/treasure dependency chain.

    Actions: 0=up, 1=down, 2=left, 3=right.  Bumping a wall (or the door
    without the key) is a no-op.  Walking onto the key tile picks it up
    (the key tile then behaves like a normal floor tile).  Walking onto the
    treasure ends the episode with reward +1.  Any other step gives 0.

    State = (row, col, has_key) flattened to a single int in
    [0, rows*cols*2).  With the default 15x7 map that is 210 states -- tiny
    enough for a Q-table, big enough that blind exploration fails.
    """

    n_actions = 4  # up, down, left, right
    _DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, max_steps=400):
        self.grid = [list(r) for r in _MONTEZUMA_MAP]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.max_steps = max_steps
        # locate the special tiles
        for r in range(self.rows):
            for c in range(self.cols):
                ch = self.grid[r][c]
                if ch == "S":
                    self.start = (r, c)
                elif ch == "K":
                    self.key_pos = (r, c)
                elif ch == "D":
                    self.door_pos = (r, c)
                elif ch == "T":
                    self.treasure_pos = (r, c)
        self.n_states = self.rows * self.cols * 2
        self.reset()

    # --- helpers ---------------------------------------------------------
    def _state(self):
        flat_rc = self.r * self.cols + self.c
        return flat_rc * 2 + (1 if self.has_key else 0)

    @classmethod
    def decode(cls, state, cols):
        """state -> (row, col, has_key).  `cols` is env.cols."""
        flat_rc, has_key = divmod(state, 2)
        row, col = divmod(flat_rc, cols)
        return row, col, has_key

    def _passable(self, r, c):
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        ch = self.grid[r][c]
        if ch == "#":
            return False
        if ch == "D":                       # the door
            return self.has_key
        return True

    # --- Gym-like API ----------------------------------------------------
    def reset(self):
        self.r, self.c = self.start
        self.has_key = False
        self.steps = 0
        return self._state()

    def step(self, action):
        assert action in self._DELTAS
        self.steps += 1
        dr, dc = self._DELTAS[action]
        nr, nc = self.r + dr, self.c + dc
        if self._passable(nr, nc):
            self.r, self.c = nr, nc

        # pick up the key
        if (self.r, self.c) == self.key_pos:
            self.has_key = True

        reward = 0.0
        done = False
        if (self.r, self.c) == self.treasure_pos:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
        return self._state(), reward, done, {}

    # --- pretty-printing -------------------------------------------------
    def render_ascii(self, agent_marker="@"):
        out = []
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                if (r, c) == (self.r, self.c):
                    row += agent_marker
                else:
                    row += self.grid[r][c]
            out.append(row)
        return "\n".join(out)


# ===========================================================================
# tiny smoke test
# ===========================================================================
if __name__ == "__main__":
    print("DeepSea(size=6): always-RIGHT rollout")
    env = DeepSeaEnv(size=6)
    s, total = env.reset(), 0.0
    for _ in range(6):
        s, r, done, _ = env.step(1)
        total += r
    print(f"  return = {total:.3f}  (should be ~+0.99)\n")

    print("MiniMontezuma: scripted key->door->treasure rollout")
    env = MiniMontezumaEnv()
    # hand-coded path of action ids (0=up,1=down,2=left,3=right)
    path = ([1, 1, 1, 1]                  # down 4 rows: (1,1) -> (5,1)
            + [3, 3]                       # right to the key:  (5,1) -> (5,3)
            + [0]                          # up one row:        (5,3) -> (4,3)
            + [3, 3, 3]                    # right to the door: (4,3) -> (4,6)
            + [3, 3, 3, 3]                 # right through the right room -> (4,10)
            + [0])                         # up onto the treasure: (4,10) -> (3,10)
    s, total, done = env.reset(), 0.0, False
    for a in path:
        s, r, done, _ = env.step(a)
        total += r
        if done:
            break
    print(env.render_ascii())
    print(f"  has_key={env.has_key}  return={total:.1f}  done={done}  "
          f"steps={env.steps}")
