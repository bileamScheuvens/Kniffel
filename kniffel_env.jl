using ReinforcementLearning
using IntervalSets
using StatsBase
using DataStructures
using Random
using DataFrames

"""
Define Gym
"""
Base.@kwdef mutable struct KniffelGym <: AbstractEnv
    dices::Vector{Int} = rand(1:6, 5)
    fields::Vector{Union{Nothing, Int}} = fill(nothing, 13)
    # track total reward
    total_reward::Int = 0
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
RLBase.state(env::KniffelGym) = [env.dices; env.fields]
# access statespace
RLBase.state_space(env::KniffelGym) = Space(18)
# check if terminated
RLBase.is_terminated(env::KniffelGym) = sum(isnothing.(env.fields)) == 0

function RLBase.reward(env::KniffelGym)
    filled_fields = filter(x->!isnothing(x), env.fields)
    sum(filled_fields) + (sum([x for x in env.fields if !isnothing(x)]) >= 63 ? 35 : 0)
end

# reset gym
function RLBase.reset!(env::KniffelGym)
    env.dices = rand(1:6, 5)
    env.fields = fill(nothing, 13)
    env.total_reward = reward(env)
end

"""
Take an action
"""
function (env::KniffelGym)(action)
    env.fields[action] = check_action(action, env.dices, env.fields)
    filled_fields = filter(x->!isnothing(x), env.fields)
    env.total_reward = length(filled_fields) > 0 ? sum(filled_fields) : 0
    "field $action filled with $(env.fields[action]) points"
    state(env)
    env.dices = rand(1:6, 5)
end

function check_action(desired::Int, dices::Vector{Int}, fields)
    potential_reward = eval_action(desired, dices)
    agdices = DefaultDict(0, countmap(dices))
    # field is full
    if !isnothing(fields[desired])
        @error "unknown or illegal action $action"
    # kniffel thrown
    elseif length(agdices) == 1 && !isnothing(fields[12])
        return eval_action(desired, dices)
    # 1s to 6s
    elseif desired in 1:6
        return eval_action(desired, dices)
    # multiples
    elseif desired in [7,8]
        return (maximum(values(agdices)) >= desired-4) ? potential_reward : 0
    # full house
    elseif desired == 9
        return (maximum(values(agdices)) == 3 && length(agdices) == 2) ? potential_reward : 0
    # streets
    elseif desired in [10,11]
        street_candidates = sort(unique([1,3,4,5,6]))
        street_candidates = join([street_candidates[i]-street_candidates[i-1] for i in 2:length(street_candidates)])
        patterns = [r"111", r"1111"]
        return occursin(patterns[desired-9], street_candidates) ? potential_reward : 0
    elseif desired == 12
        return (length(agdices) == 1) ? potential_reward : 0
    # chance
    elseif desired == 13
        return potential_reward
    end
    return "Check Kaputt"
end

function eval_action(desired::Int, dices::Vector{Int})
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

function show_state(env::KniffelGym)
    dices = env.dices
    fields = env.fields
    round = length(filter(x->!isnothing(x), fields)) + 1 
    label_dice=["Dice 1", "Dice 2", "Dice 3", "Dice 4", "Dice 5"]
    label_field = [
        "Ones", "Twos", "Threes", "Fours", "Fives", "Sixs", "Three of a kind",
        "Four of a kind", "Full House", "Short Street", "Long Street", "Kniffel", "Chance"
    ]
    println("Current Round: $round")
    println(DataFrame(Label=label_dice, Faces=dices), "\n")
    println(DataFrame(Label=label_field, Value=fields))
end


env = KniffelGym()
env(7)
show_state(env)

filter(x->!isnothing(x), env.fields)
