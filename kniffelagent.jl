using Flux
import Flux.Losses: huber_loss
using BSON
include("kniffel_env.jl")

# init environment
env = KniffelGym()

# take random legal action
is_terminated(env) ? "total reward: $(env.total_reward)" : env |> legal_action_space |> rand |> env |> show_state

# test on random policy
run(RandomPolicy(), env, StopAfterEpisode(1000), TotalRewardPerEpisode())

# dims for in and output layer
ns, na = length(state(env)), length(action_space(env))


# Baseline Dense Model
architecture = Chain(
    Dense(ns, 128, relu; init = glorot_uniform),
    Dense(128, 64, relu; init = glorot_uniform),
    Dense(64, 32, relu; init = glorot_uniform),
    Dense(32, na; init = glorot_uniform),
) |> cpu

"""
Example DeepRL Agent from documentation (https://juliareinforcementlearning.org/docs/)
"""

policy = Agent(
           policy = QBasedPolicy(
               learner = DQNLearner(
                   approximator = NeuralNetworkApproximator(
                       model = architecture,
                       optimizer = ADAM()
                   ),
                   target_approximator = NeuralNetworkApproximator(
                    model = architecture,
                    optimizer = ADAM()
                   ),
                   loss_func = huber_loss,
                   batch_size = 32,
                   min_replay_history = 100,
               ),
               explorer = EpsilonGreedyExplorer(
                   kind = :exp,
                   ϵ_stable = 0.01,
                   decay_steps = 500,
               ),
           ),
           trajectory = CircularArraySARTTrajectory(
               capacity = 1000,
               state = Vector{Float32} => (ns,),
           ),
)


# create temporary directory
parameters_dir = mktempdir()

run(
    policy,
    env,
    StopAfterEpisode(70000),
    ComposedHook(
        TotalRewardPerEpisode(),
        RewardsPerEpisode(),
        DoEveryNEpisode(n=20000) do t, p, e
            ps = params(p)
            f = joinpath(parameters_dir, "parameters_at_step_$t.bson")
            BSON.@save f ps
            println("parameters at step $t saved to $f")
        end
    )
)