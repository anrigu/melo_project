using Combinatorics: multinomial, with_replacement_combinations as CwR
using StatsBase: counts

using TensorCast
import CUDA
CUDA.allowscalar(false)

const MAXIMUM_PAYOFF = 1e5  # all payoffs are standardized to the (MIN,MAX) range
const MINIMUM_PAYOFF = 1e-5 # to ensure numerical stability and simplify parameter tuning
const F32_EPSILON = eps(1f-20)
const F64_EPSILON = eps(1e-40)

abstract type AbstractSymGame end

struct SymmetricGame <: AbstractSymGame
    num_players::Integer
    num_actions::Integer
    config_table::AbstractMatrix # num_actions X num_configs float array
    payoff_table::AbstractMatrix # num_actions X num_configs float array
    offset::AbstractFloat
    scale::AbstractFloat
    ε::AbstractFloat
    GPU::Bool # flag indicating whether tables are CuArrays
end

function SymmetricGame(num_players, num_actions, payoff_generator;
                    GPU=false, ub=MAXIMUM_PAYOFF, lb=MINIMUM_PAYOFF)
    num_configs = multinomial(num_players-1, num_actions-1)
    config_table = zeros(Float64, num_actions, num_configs)
    payoff_table = Array{Float64}(undef, num_actions, num_configs)
    repeat_table = Array{Float64}(undef, 1, num_configs) # temporary: used to pre-weight payoffs
    # generate each configuration
    for (c,config) in enumerate(CwR(1:num_actions, num_players-1))
        prof = counts(config, 1:num_actions)
        config_table[:,c] = prof
        repeat_table[c] = logmultinomial(prof...)
        payoff_table[:,c] = payoff_generator(prof)
    end
    (offset, scale) = set_scale(minimum(payoff_table), maximum(payoff_table); ub=ub, lb=lb)
    # normalize, log-transform, and repetition-weight the payoff table
    payoff_table = log.(normalize_payoffs(payoff_table, offset, scale)) .+ repeat_table
    if GPU
        config_table = CUDA.CuArray{Float32,2}(config_table)
        payoff_table = CUDA.CuArray{Float32,2}(payoff_table)
        num_players = Int32(num_players)
        num_actions = Int32(num_actions)
        offset = Float32(offset)
        scale = Float32(scale)
        ε = F32_EPSILON
    else
        num_players = Int64(num_players)
        num_actions = Int64(num_actions)
        offset = Float64(offset)
        scale = Float64(scale)
        ε = F64_EPSILON
    end
    SymmetricGame(num_players, num_actions, config_table, payoff_table, offset, scale, ε, GPU)
end

# creates a new game using another game as the 'payoff generator'
function SymmetricGame(game::AbstractSymGame; GPU=false, ub=MAXIMUM_PAYOFF, lb=MINIMUM_PAYOFF)
    payoff_generator = prof -> pure_payoffs(game, prof)
    SymmetricGame(game.num_players, game.num_actions, payoff_generator; GPU=GPU, ub=ub, lb=lb)
end

# finds the affine transformation that scales a game to the standard payoff range
function set_scale(min_payoff, max_payoff; ub=MAXIMUM_PAYOFF, lb=MINIMUM_PAYOFF)
    scale = (ub - lb) / (max_payoff - min_payoff)
    if !isfinite(scale)
        scale = 1
    end
    offset = lb / scale - min_payoff
    if !isfinite(offset)
        offset = 0
    end
    return (offset, scale)
end

# applies the affine transformation that scales a game to the standard payoff range
function normalize_payoffs(payoffs::Union{AbstractVecOrMat,Real}, offset::Real, scale::Real)
    return scale .* (payoffs .+ offset)
end

function normalize_payoffs(payoffs::Union{AbstractVecOrMat,Real}, game::AbstractSymGame)
    return game.scale .* (payoffs .+ game.offset)
end

# undoes the affine transformation to get back payoffs in a game's original scale
function denormalize_payoffs(payoffs::Union{AbstractVecOrMat,Real}, offset::Real, scale::Real)
    return (payoffs ./ scale) .- offset
end

function denormalize_payoffs(payoffs::Union{AbstractVecOrMat,Real}, game::AbstractSymGame)
    return (payoffs ./ game.scale) .- game.offset
end

# look up a profile in the payoff table
function pure_payoffs(game::SymmetricGame, opponent_profile; denormalize=false)
    index = profile_ranking(opponent_profile)
    weighted_normalized_log_payoffs = game.payoff_table[:,index]
    repeats = logmultinomial(opponent_profile...)
    normalized_payoffs = exp.(weighted_normalized_log_payoffs .- repeats)
    if denormalize
        return denormalize_payoffs(normalized_payoffs, game)
    else
        return normalized_payoffs
    end
end

# core method: calculates expected utilities for each action when all opponents play mixture
function deviation_payoffs(game::SymmetricGame, mixture::AbstractVector)
    log_mixture = log.(mixture .+ game.ε)
    @reduce log_config_probs[c] := sum(a) log_mixture[a] * game.config_table[a,c]
    @reduce dev_pays[a] := sum(c) exp(game.payoff_table[a,c] + log_config_probs[c])
    return dev_pays
end

# vectorized over an array of mixtures
function deviation_payoffs(game::SymmetricGame, mixtures::AbstractMatrix)
    log_mixtures = log.(mixtures .+ game.ε)
    @reduce log_config_probs[m,c] := sum(a) log_mixtures[a,m] * game.config_table[a,c]
    @reduce dev_pays[a,m] := sum(c) exp(game.payoff_table[a,c] + log_config_probs[m,c])
    return dev_pays
end

# jacobian matrix of deviation-payoff derivatives
function deviation_derivatives(game::SymmetricGame, mixture::AbstractVector)
    mixture = mixture .+ game.ε
    log_mixture = log.(mixture)
    @reduce log_config_probs[c] := sum(a) log_mixture[a] * game.config_table[a,c]
    @cast deriv_configs[a,c] := game.config_table[a,c] / (mixture[a])
    @reduce dev_jac[a,s] := sum(c) exp(game.payoff_table[a,c] + log_config_probs[c]) * deriv_configs[s,c]
    return dev_jac
end

# vectorized over an array of mixtures
function deviation_derivatives(game::SymmetricGame, mixtures::AbstractMatrix)
    mixtures = mixtures .+ game.ε
    log_mixtures = log.(mixtures)
    @reduce log_config_probs[m,c] := sum(a) log_mixtures[a,m] * game.config_table[a,c]
    @cast deriv_configs[a,m,c] := game.config_table[a,c] / (mixtures[a,m])
    @reduce dev_jac[a,s,m] := sum(c) exp(game.payoff_table[a,c] + log_config_probs[m,c]) * deriv_configs[s,m,c]
    return dev_jac
end

# function whose global minima are Nash equilibria
function deviation_gains(game::AbstractSymGame, mix::AbstractVecOrMat)
    dev_pays = deviation_payoffs(game, mix)
    max.(dev_pays .- sum(dev_pays .* mix, dims=1), 0)
end

# derivatives for minimizing deviation_gains
function gain_gradients(game::AbstractSymGame, mixture::AbstractVector)
    dev_pays = deviation_payoffs(game, mixture)
    mixture_EV = mixture' * dev_pays
    dev_jac = deviation_derivatives(game, mixture)
    util_grads = (mixture' * dev_jac)' .+ dev_pays
    gain_jac = dev_jac .- util_grads'
    gain_jac[dev_pays .< mixture_EV,:] .= 0
    return dropdims(sum(gain_jac, dims=1), dims=1)
end

# vectorized over an array of mixtures
function gain_gradients(game::AbstractSymGame, mixtures::AbstractMatrix)
    dev_pays = deviation_payoffs(game, mixtures)
    @reduce mixture_expectations[m] := sum(a) mixtures[a,m] * dev_pays[a,m]
    dev_jac = deviation_derivatives(game, mixtures)
    @reduce util_grads[s,m] := sum(a) mixtures[a,m] * dev_jac[a,s,m]
    util_grads .+= dev_pays
    @cast gain_jac[s,a,m] := dev_jac[a,s,m] - util_grads[s,m]
    # The findall shouldn't be necessary here; this is a Julia language bug.
    # See discourse.julialang.org/t/slicing-and-boolean-indexing-in-multidimensional-arrays
    gain_jac[:,findall(dev_pays .< mixture_expectations')] .= 0
    return dropdims(sum(gain_jac, dims=2), dims=2)
end

# maximum any player could gain by deviating
function regret(game::AbstractSymGame, mixture::AbstractVector)
    maximum(deviation_gains(game, mixture))
end

# vectorized over an array of mixtures
function regret(game::AbstractSymGame, mixtures::AbstractMatrix)
    dropdims(maximum(deviation_gains(game, mixtures), dims=1), dims=1)
end

# Returns a boolean vector (or matrix if given several mixtures)
# indicating which strategies are best-responses.
function best_responses(game::AbstractSymGame, mix::AbstractVecOrMat; atol=eps(0e0))
    dev_pays = deviation_payoffs(game, mix)
    return isapprox.(dev_pays, maximum(dev_pays, dims=1), atol=atol)
end

# The classic function whose fixed-point is Nash.
# For use in Scarf's simplicial subdivision algrotihm.
function better_response(game::AbstractSymGame, mix::AbstractVecOrMat; scale_factor::Real=1)
    gains = max.(0,deviation_gains(game, mix)) .* scale_factor
    return (mix .+ gains) ./ (1 .+ sum(gains, dims=1))
end

# throw out mixtures with regret greater than threshold
function filter_regrets(game::AbstractSymGame, mixtures::AbstractMatrix; threshold=1e-3, sorted=true)
    mixture_regrets = regret(game, mixtures)
    below_threshold = mixture_regrets .< threshold
    mixtures = mixtures[:,below_threshold]
    mixture_regrets = mixture_regrets[below_threshold]
    if sorted
        mixtures = mixtures[:,sortperm(mixture_regrets)]
    end
    return mixtures
end