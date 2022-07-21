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
    field_values::Vector{Int} = zeros(13)
    field_isempty::Vector{Bool} = fill(true, 13)
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
RLBase.legal_action_space(env::KniffelGym) = filter(x->x!=0, env.field_isempty .* collect(1:13))
# same as above just as BitArray mask
RLBase.legal_action_space_mask(env::KniffelGym) = env.field_isempty
# access state
RLBase.state(env::KniffelGym) = [env.dices; env.field_values; env.field_isempty]
# access statespace
RLBase.state_space(env::KniffelGym) = Space(31)
# check if terminated
RLBase.is_terminated(env::KniffelGym) = sum(env.field_isempty) == 0

# reward = regular + bonus 
RLBase.reward(env::KniffelGym) = sum(env.field_values) + sum(env.field_values[1:6]) >= 63 ? 35 : 0 

# reset gym
function RLBase.reset!(env::KniffelGym)
    env.dices = rand(1:6, 5)
    env.field_values = zeros(13)
    env.field_isempty = fill(true, 13)
    env.total_reward = 0
end

# for whatever reason reset isn't blocked in namespace, cheeky workaround
reset!(env::KniffelGym) = RLBase.reset!(env)


"""
Take an action
"""
function (env::KniffelGym)(action)
    env.field_values[action] = check_action(action, env.dices, env.field_values, env.field_isempty)
    env.field_isempty[action] = false
    env.total_reward = reward(env)
    env.dices = rand(1:6, 5)
    env
end

function check_action(desired::Int, dices::Vector{Int}, field_values, field_isempty)
    potential_reward = eval_action(desired, dices)
    agdices = DefaultDict(0, countmap(dices))
    # field is full
    if field_isempty[desired] == 0
        @error "unknown or illegal action $action"
    # kniffel thrown and kniffel field not crossed out
    elseif length(agdices) == 1 && field_isempty[desired] && field_values[12] > 0
        return potential_reward
    # 1s to 6s
    elseif desired in 1:6
        return potential_reward
    # multiples
    elseif desired in [7,8]
        return (maximum(values(agdices)) >= desired-4) ? potential_reward : 0
    # full house
    elseif desired == 9
        return (maximum(values(agdices)) == 3 && length(agdices) == 2) ? potential_reward : 0
    # streets
    elseif desired in [10,11]
        # get unique dice faces
        street_candidates = sort(unique(keys(agdices)))
        # calculate difference to next lower face
        street_candidates = join([street_candidates[i]-street_candidates[i-1] for i in 2:length(street_candidates)])
        # match pattern of one increments 3/4 times for short and long street respectively
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
        # 30 if 10, 40 if 11
        return 10*(desired-7)
    end
    return "Eval Kaputt"
end

function show_state(env::KniffelGym)
    dices = env.dices
    fields = env.field_values
    round = sum(.!env.field_isempty) + 1 
    label_dice=["Dice 1", "Dice 2", "Dice 3", "Dice 4", "Dice 5"]
    label_field = [
        "Ones", "Twos", "Threes", "Fours", "Fives", "Sixs", "Three of a kind",
        "Four of a kind", "Full House", "Short Street", "Long Street", "Kniffel", "Chance"
    ]
    println("Current Round: $round")
    println(DataFrame(Label=label_dice, Faces=dices), "\n")
    println(DataFrame(Label=label_field, Value=fields, Is_Full=.!env.field_isempty))
end


env = KniffelGym()