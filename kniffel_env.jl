using ReinforcementLearning
using IntervalSets
using StatsBase
using DataStructures

"""
Define Gym
"""
Base.@kwdef mutable struct KniffelGym <: AbstractEnv
    dices = zeros(5)
    fields = fill(nothing, 13)
    # track total reward
    reward::Float32 = 0
end

# define action space
RLBase.action_space(env::KniffelGym) = vec(1:13)
# define action space
RLBase.ActionStyle(env::KniffelGym) = FULL_ACTION_SET
# set chance style
RLBase.ChanceStyle(env::KniffelGym) = STOCHASTIC
# define legal action space 
RLBase.legal_action_space(env::KniffelGym) = [x for x in 1:13 if check_action(x, env.dices, env.fields)]
# same as above just as BitArray mask
RLBase.legal_action_space_mask(env::KniffelGym) = [check_action(x, env.dices, env.fields) for x in 1:13]
# access state
RLBase.state(env::KniffelGym) = nothing
# access statespace
RLBase.state_space(env::KniffelGym) = nothing
# check if terminated
RLBase.is_terminated(env::KniffelGym) = sum(isnothing.(env.fields)) == 0
# reset gym
function RLBase.reset!(env::KniffelGym)
    env.dices = zeros(5)
    env.fields = fill(nothing, 13)
    env.reward = 0
end

"""
Take an action
"""
function (env::KniffelGym)(action)
    return nothing
end

function check_action(desired::Int8, dices::Vector{Int8}, fields::Vector{Int8})
    agdices = DefaultDict(0, countmap(dices))
    # field is full
    if !isnothing(fields[desired])
        return false
    # kniffel thrown
    elseif length(agdices) == 1 && !isnothing(fields[12])
        return true
    # 1s to 6s
    elseif desired in 1:6
        return true
    # multiples
    elseif desired in [7,8]
        return maximum(values(agdices)) >= desired-4
    # full house
    elseif desired == 9
        return maximum(values(agdices)) == 3 && length(agdices) == 2
    # streets
    elseif desired in [10,11]
        street_candidates = sort(unique([1,3,4,5,6]))
        street_candidates = join([street_candidates[i]-street_candidates[i-1] for i in 2:length(street_candidates)])
        patterns = [r"111", r"1111"]
        return (patterns[desired-9], street_candidates)
    # kniffel chosen
    elseif desired == 12
        return length(agdices) == 1
    # chance
    elseif desired == 13
        return true
    end
    return "Check Kaputt"
end

function eval_action(desired::Int8, dices::Vector{Int8}, fields::Vector{Int8})
    agdices = DefaultDict(0, countmap(dices))
    # kniffel exception
    if length(agdices) == 1
        return 50
    # 1s to 6s
    elseif desired in 1:6
        return agdices[desired]*desired
    # multiples or chance
    elseif desired in [7,8,13]
        return sum(dices)
    # full house
    elseif desired == 9
        return 25
    # streets
    elseif desired in [10,11]
        return 10*(desired-7)
    end
    return "Eval Kaputt"
end

