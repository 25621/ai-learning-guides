"""
Policy Iteration for GridWorld

A 4x4 grid where an agent must navigate from top-left (0,0) to bottom-right (3,3).
- States: 16 grid cells (0-15)
- Actions: Up, Down, Left, Right
- Rewards: -1 per step, 0 at terminal states (top-left and bottom-right corners)
- Discount: gamma = 0.9 (< 1 ensures convergence for any policy)

Policy Iteration alternates between:
  1. Policy Evaluation: Compute V(s) for the current policy
  2. Policy Improvement: Greedily update the policy w.r.t. V(s)
Until the policy stops changing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4  # 0=Up, 1=Down, 2=Left, 3=Right
TERMINAL_STATES = {0, 15}
GAMMA = 0.9
THETA = 1e-6  # convergence threshold

ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
ACTION_SYMBOLS = ['↑', '↓', '←', '→']
DELTA_ROW = [-1, 1, 0, 0]
DELTA_COL = [0, 0, -1, 1]


def state_to_rc(s):
    return s // GRID_SIZE, s % GRID_SIZE


def rc_to_state(r, c):
    return r * GRID_SIZE + c


def step(state, action):
    """Deterministic transition: returns (next_state, reward)."""
    if state in TERMINAL_STATES:
        return state, 0

    r, c = state_to_rc(state)
    nr = np.clip(r + DELTA_ROW[action], 0, GRID_SIZE - 1)
    nc = np.clip(c + DELTA_COL[action], 0, GRID_SIZE - 1)
    next_state = rc_to_state(nr, nc)
    return next_state, -1


def policy_evaluation(policy, V, gamma=GAMMA, theta=THETA):
    """Iteratively compute V(s) for a fixed policy."""
    iterations = 0
    while True:
        delta = 0
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                continue
            v = V[s]
            a = policy[s]
            s_next, r = step(s, a)
            V[s] = r + gamma * V[s_next]
            delta = max(delta, abs(v - V[s]))
        iterations += 1
        if delta < theta:
            break
    return V, iterations


def policy_improvement(policy, V, gamma=GAMMA):
    """Greedily improve policy w.r.t. current V(s). Returns (new_policy, stable)."""
    policy_stable = True
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            continue
        old_action = policy[s]
        action_values = []
        for a in range(N_ACTIONS):
            s_next, r = step(s, a)
            action_values.append(r + gamma * V[s_next])
        policy[s] = np.argmax(action_values)
        if old_action != policy[s]:
            policy_stable = False
    return policy, policy_stable


def policy_iteration():
    """Run full policy iteration until convergence."""
    V = np.zeros(N_STATES)
    policy = np.zeros(N_STATES, dtype=int)  # start with all "Up"

    print("=== Policy Iteration for GridWorld ===\n")
    pi_round = 0
    total_eval_iters = 0

    while True:
        pi_round += 1
        V, eval_iters = policy_evaluation(policy, V)
        total_eval_iters += eval_iters
        policy, stable = policy_improvement(policy, V)
        print(f"Round {pi_round}: policy-eval took {eval_iters} sweeps | policy stable: {stable}")
        if stable:
            break

    print(f"\nConverged in {pi_round} policy-iteration rounds ({total_eval_iters} total eval sweeps).")
    return V, policy


def print_grid(values, label):
    print(f"\n{label}")
    print("-" * (GRID_SIZE * 8 + 1))
    for r in range(GRID_SIZE):
        row = "|"
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            row += f" {values[s]:5.2f} |"
        print(row)
    print("-" * (GRID_SIZE * 8 + 1))


def print_policy(policy):
    print("\nOptimal Policy:")
    print("-" * (GRID_SIZE * 6 + 1))
    for r in range(GRID_SIZE):
        row = "|"
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if s in TERMINAL_STATES:
                row += "  T  |"
            else:
                row += f"  {ACTION_SYMBOLS[policy[s]]}  |"
        print(row)
    print("-" * (GRID_SIZE * 6 + 1))


def visualize(V, policy):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    V_grid = V.reshape(GRID_SIZE, GRID_SIZE)
    im = axes[0].imshow(V_grid, cmap='RdYlGn')
    plt.colorbar(im, ax=axes[0])
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            axes[0].text(c, r, f"{V[s]:.1f}", ha='center', va='center',
                         fontsize=12, fontweight='bold',
                         color='black' if V[s] > V.min() + (V.max() - V.min()) * 0.3 else 'white')
    axes[0].set_title("State Values V(s) — Optimal Policy", fontsize=13)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_xlim(-0.5, GRID_SIZE - 0.5)
    axes[1].set_ylim(-0.5, GRID_SIZE - 0.5)
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    arrow_dx = [0, 0, -0.3, 0.3]
    arrow_dy = [-0.3, 0.3, 0, 0]
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if s in TERMINAL_STATES:
                axes[1].text(c, r, 'T', ha='center', va='center',
                             fontsize=14, fontweight='bold', color='red')
            else:
                a = policy[s]
                axes[1].annotate('', xy=(c + arrow_dx[a], r + arrow_dy[a]),
                                 xytext=(c, r),
                                 arrowprops=dict(arrowstyle='->', color='navy', lw=2))
    for r in range(GRID_SIZE + 1):
        axes[1].axhline(r - 0.5, color='gray', lw=0.5)
    for c in range(GRID_SIZE + 1):
        axes[1].axvline(c - 0.5, color='gray', lw=0.5)
    axes[1].set_title("Optimal Policy π*(s)", fontsize=13)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    out_path = "outputs/policy_iteration_gridworld.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nVisualization saved to {out_path}")


def main():
    V, policy = policy_iteration()
    print_grid(V, "State Values V(s):")
    print_policy(policy)
    visualize(V, policy)


if __name__ == "__main__":
    main()
