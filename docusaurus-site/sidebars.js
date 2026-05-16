/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  guideSidebar: [
    {
      type: 'doc',
      id: 'README',
      label: 'Introduction'
    },
    {
      type: 'category',
      label: 'Foundations',
      collapsed: false,
      items: [
        'foundations/multi_armed_bandit_explained',
        'foundations/frozen_lake_explained',
        'foundations/state_value_visualization_explained',
      ],
    },
    {
      type: 'category',
      label: 'Tabular Methods',
      collapsed: false,
      items: [
        'tabular_methods/policy_iteration_gridworld_explained',
        'tabular_methods/q_learning_frozen_lake_explained',
        'tabular_methods/sarsa_cliff_walking_explained',
        'tabular_methods/sarsa_vs_qlearning_explained',
        'tabular_methods/monte_carlo_blackjack_explained',
      ],
    },
    {
      type: 'category',
      label: 'Function Approximation',
      items: [
        'function_approximation/linear_q_cartpole_explained',
        'function_approximation/dqn_cartpole_explained',
        'function_approximation/dqn_experience_replay_explained',
        'function_approximation/dqn_target_network_explained',
        'function_approximation/dqn_atari_pong_explained',
        'function_approximation/double_dqn_cartpole_explained',
      ],
    },
    {
      type: 'category',
      label: 'Policy Gradients',
      items: [
        'policy_gradients/reinforce_cartpole_explained',
        'policy_gradients/reinforce_baseline_explained',
        'policy_gradients/a2c_lunarlander_explained',
        'policy_gradients/ppo_scratch_explained',
        'policy_gradients/ppo_continuous_explained',
        'policy_gradients/ppo_hyperparams_explained',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Topics',
      items: [
        {
          type: 'category',
          label: 'Model-Based RL',
          items: [
            'advanced_topics/model_based_rl/dyna_q_explained',
            'advanced_topics/model_based_rl/world_model_explained',
            'advanced_topics/model_based_rl/model_based_planning_explained',
          ],
        },
        {
          type: 'category',
          label: 'Multi-Agent RL',
          items: [
            'advanced_topics/multi_agent_rl/matrix_games_explained',
            'advanced_topics/multi_agent_rl/self_play_tic_tac_toe_explained',
            'advanced_topics/multi_agent_rl/pettingzoo_explore_explained',
          ],
        },
        {
          type: 'category',
          label: 'Offline RL',
          items: [
            'advanced_topics/offline_rl/d4rl_dataset_explained',
            'advanced_topics/offline_rl/cql_explained',
            'advanced_topics/offline_rl/behavioral_cloning_explained',
          ],
        },
        {
          type: 'category',
          label: 'Exploration',
          items: [
            'advanced_topics/exploration/curiosity_bonus_explained',
            'advanced_topics/exploration/montezuma_revenge_explained',
            'advanced_topics/exploration/compare_exploration_explained',
          ],
        },
        {
          type: 'category',
          label: 'Hierarchical RL',
          items: [
            'advanced_topics/hierarchical_rl/option_critic_explained',
            'advanced_topics/hierarchical_rl/goal_conditioned_policy_explained',
            'advanced_topics/hierarchical_rl/long_horizon_tasks_explained',
          ],
        },
        {
          type: 'category',
          label: 'RLHF',
          items: [
            'advanced_topics/rlhf/reward_modeling_explained',
            'advanced_topics/rlhf/ppo_fine_tuning_explained',
            'advanced_topics/rlhf/dpo_implementation_explained',
          ],
        },
      ],
    },
  ],
};

module.exports = sidebars;
