module TrueSkill

#using Parameters
#import SpecialFunctions

global const BETA = 1.0::Float64
global const MU = 0.0::Float64
global const SIGMA = (BETA * 6)::Float64
global const GAMMA = (BETA * 0.05)::Float64
global const P_DRAW = 0.0::Float64
global const EPSILON = 1e-6::Float64
global const ITER = 10::Int64
global const sqrt2 = sqrt(2)
global const sqrt2pi = sqrt(2*pi)

struct Environment
    mu::Float64
    sigma::Float64
    beta::Float64
    gamma::Float64
    p_draw::Float64
    epsilon::Float64
    iter::Int64
    function Environment(mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA, gamma::Float64=GAMMA, p_draw::Float64=P_DRAW, epsilon::Float64=EPSILON, iter::Int64=ITER )
        return new(mu, sigma, beta, gamma, p_draw, epsilon, iter)
    end
    function Environment(;mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA, gamma::Float64=GAMMA, p_draw::Float64=P_DRAW, epsilon::Float64=EPSILON, iter::Int64=ITER)
        return new(mu, sigma, beta, gamma, p_draw, epsilon, iter)
    end
end
function erfc(x::Float64)
    #"""Complementary error function (thanks to http://bit.ly/zOLqbc)"""
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    r = begin
        a = -0.82215223 + t * 0.17087277 
        b =  1.48851587 + t * a
        c = -1.13520398 + t * b
        d =  0.27886807 + t * c
        e = -0.18628806 + t * d
        f =  0.09678418 + t * e
        g =  0.37409196 + t * f
        h =  1.00002368 + t * g
        t * exp(-z * z - 1.26551223 + t * h)
        end
    if x < 0
        r = 2.0 - r
    end
    return r
end
function erfcinv(y::Float64)
    """The inverse function of erfc."""
    if y >= 2
        return -Inf
    elseif y < 0
        throw(DomainError(y, "argument must be nonnegative"))
    elseif y == 0
        return Inf
    end
    zero_point = y < 1
    if ! zero_point 
        y = 2 - y
    end
    t = sqrt(-2 * log(y / 2.0))
    x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
    for _ in 0:2
        err = erfc(x) - y
        x += err / (1.12837916709551257 * exp(-(x^2)) - x * err)
    end
    if zero_point
        r = x
    else 
        r = -x
    end
    return r
end
function tau_pi(mu::Float64, sigma::Float64)
    if sigma < 0 
        error("sigma should be greater than 0")
    elseif sigma > 0.
        _pi = sigma^-2
        _tau = _pi * mu
    else
        _pi = Inf
        _tau = Inf
    end
    return _tau, _pi
end
function mu_sigma(_tau::Float64, _pi::Float64)
    if _pi < 0.
        error("Precision should be greater than 0")
    elseif _pi > 0.0
        sigma = sqrt(1/_pi)
        mu = _tau / _pi
    else
        sigma = Inf
        mu = 0
    end
    return mu, sigma
end

struct Gaussian
    # TODO: support Gaussian(mu=0.0,sigma=1.0)
    mu::Float64
    sigma::Float64
    tau::Float64
    pi::Float64
    function Gaussian(a::Float64=MU, b::Float64=SIGMA, inverse::Bool=false)
        if !inverse
            mu, sigma = (a, b)
            _tau, _pi = tau_pi(mu, sigma)
        else
            _tau, _pi = (a, b)
            mu, sigma = mu_sigma(_tau, _pi)
        end
        return new(mu, sigma, _tau, _pi)
    end
end

global const N01 = Gaussian(0.0, 1.0)
global const Ninf = Gaussian(0.0, Inf)
global const N00 = Gaussian(0.0, 0.0)

Base.show(io::IO, g::Gaussian) = print("Gaussian(mu=", round(g.mu,digits=3)," ,sigma=", round(g.sigma,digits=3), ")")
function cdf(N::Gaussian, x::Float64)
    z = -(x - N.mu) / (N.sigma * sqrt2)
    return (0.5 * erfc(z))::Float64
end
function pdf(N::Gaussian, x::Float64)
    normalizer = (sqrt(2 * pi) * N.sigma)^-1
    functional = exp( -((x - N.mu)^2) / (2*N.sigma ^2) ) 
    return (normalizer * functional)::Float64
end
function ppf(N::Gaussian, p::Float64)
    return N.mu - N.sigma * sqrt2  * erfcinv(2 * p)
end 
function trunc(N::Gaussian, margin::Float64, tie::Bool)
    # The range is [alpha, beta]
    if !tie
        _alpha = (margin-N.mu)/N.sigma
        v = pdf(N01,-_alpha) / cdf(N01,-_alpha)
        w = v * (v + (-_alpha))
    else
        _alpha = (-margin-N.mu)/N.sigma
        _beta  = (margin-N.mu)/N.sigma
        v = (pdf(N01,_alpha)-pdf(N01,_beta))/(cdf(N01,_beta)-cdf(N01,_alpha))
        u = (_alpha*pdf(N01,_alpha)-_beta*pdf(N01,_beta))/(cdf(N01,_beta)-cdf(N01,_alpha))
        w =  - ( u - v^2 ) 
    end 
    mu = N.mu + N.sigma * v
    sigma = N.sigma*sqrt(1-w)
    return Gaussian(mu, sigma)
end
function delta(N::Gaussian, M::Gaussian)
    return abs(N.mu - M.mu) , abs(N.sigma - M.sigma) 
end
function exclude(N::Gaussian,M::Gaussian)
    return Gaussian(N.mu - M.mu, sqrt(N.sigma^2 - M.sigma^2) )
end
function Base.:+(N::Gaussian, M::Gaussian)
    mu = N.mu + M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu, sigma )
end
function Base.:-(N::Gaussian, M::Gaussian)
    mu = N.mu - M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu, sigma )
end
function Base.:*(N::Gaussian, M::Gaussian)
    _pi = N.pi + M.pi
    _tau = N.tau + M.tau
    return Gaussian(_tau, _pi, true)        
end
function Base.:/(N::Gaussian, M::Gaussian)
    _pi = N.pi - M.pi
    _tau = N.tau - M.tau
    return Gaussian(_tau, _pi, true)        
end
function Base.isapprox(N::Gaussian, M::Gaussian, atol::Real=0)
    return (abs(N.mu - M.mu) < atol) & (abs(N.sigma - M.sigma) < atol)
end
function compute_margin(p_draw::Float64, sd::Float64)
    _N = Gaussian(0.0, sd )
    res = abs(ppf(_N, 0.5-p_draw/2))
    return res 
end
mutable struct Rating
    N::Gaussian
    beta::Float64
    gamma::Float64
    name::String
    draw::Gaussian
    function Rating(mu::Float64=MU, sigma::Float64=SIGMA,beta::Float64=BETA,gamma::Float64=GAMMA,name::String="",draw::Gaussian=Ninf)
        return new(Gaussian(mu, sigma), beta, gamma, name, draw)
    end
    function Rating(N::Gaussian,beta::Float64=BETA,gamma::Float64=GAMMA,name::String="",draw::Gaussian=Ninf)
        return new(N, beta, gamma, name, draw)
    end
end
Base.show(io::IO, r::Rating) = print("Rating(", round(r.N.mu,digits=3)," ,", round(r.N.sigma,digits=3), ")")
Base.copy(r::Rating) = Rating(r.N,r.beta,r.gamma,r.name)
function forget(R::Rating, t::Int64)
    _sigma = sqrt(R.N.sigma^2 + t*(R.gamma)^2)
    return Rating(Gaussian(R.N.mu, _sigma),R.beta,R.gamma,R.name)
end 
function performance(R::Rating)
    _sigma = sqrt(R.N.sigma^2 + R.beta^2)
    return Gaussian(R.N.mu, _sigma)
end
mutable struct Game
    # Mutable?
    teams::Vector{Vector{Rating}}
    result::Vector{Int64}
    p_draw::Float64
    likelihoods::Vector{Vector{Gaussian}}
    evidence::Float64
    function Game(teams::Vector{Vector{Rating}}, result::Vector{Int64},p_draw::Float64=0.0)
        # Requirements
        (length(teams) != length(result)) && throw(error("length(teams) != length(result)"))
        ((0.0 > p_draw) | (1.0 <= p_draw)) &&  throw(error("0.0 <= Draw probability < 1.0"))
        
        _g = new(teams,result,p_draw,[],0.0)
        likelihoods(_g)
        return _g
    end
end        
Base.length(G::Game) = length(G.result)
#function Base.getindex
function size(G::Game)
    return [length(g.teams[e]) for e in 1:length(g.teams)]
end
function performance(G::Game,i::Int64)
    res = N00
    for r in G.teams[i]
        res += performance(r)
    end
    return res
end 
function draw_performance(G::Game,i::Int64)
    res = N00
    for r in G.teams[i]
        res += r.draw.sigma < Inf ? trunc(r.draw,0.,false) : Ninf
    end
    return res
end 
mutable struct team_messages
    prior::Gaussian
    likelihood_lose::Gaussian
    likelihood_win::Gaussian
    likelihood_draw::Gaussian
end
function p(tm::team_messages)
    return tm.prior*tm.likelihood_lose*tm.likelihood_win*tm.likelihood_draw
end
function posterior_win(tm::team_messages)
    return tm.prior*tm.likelihood_lose*tm.likelihood_draw
end
function posterior_lose(tm::team_messages)
    return tm.prior*tm.likelihood_win*tm.likelihood_draw
end
function posterior_draw(tm::team_messages)
    return tm.prior*tm.likelihood_win*tm.likelihood_lose
end
function likelihood(tm::team_messages)
    return tm.likelihood_win*tm.likelihood_lose*tm.likelihood_draw
end
mutable struct draw_messages
    prior::Gaussian
    prior_team::Gaussian
    likelihood_lose::Gaussian
    likelihood_win::Gaussian
end
function p(um::draw_messages)
    return um.prior_team*um.likelihood_lose*um.likelihood_win
end
function posterior_win(um::draw_messages)
    return um.prior_team*um.likelihood_lose
end
function posterior_lose(um::draw_messages)
    return um.prior_team*um.likelihood_win
end
function likelihood(um::draw_messages)
    return um.likelihood_win*um.likelihood_lose
end
mutable struct diff_messages
    prior::Gaussian
    likelihood::Gaussian
end
function p(dm::diff_messages)
    return dm.prior*dm.likelihood
end
function Base.max(tuple1::Tuple{Float64,Float64}, tuple2::Tuple{Float64,Float64})
    return max(tuple1[1],tuple2[1]), max(tuple1[2],tuple2[2])
end
function Base.:>(tuple::Tuple{Float64,Float64}, threshold::Float64)
    return (tuple[1] > threshold) | (tuple[2] > threshold)
end
function likelihood_teams(g::Game)
    r = g.result
    o = sortperm(r)
    t = [team_messages(performance(g,o[e]), Ninf, Ninf, Ninf) for e in 1:length(g)]
    d = [diff_messages(t[e].prior - t[e+1].prior, Ninf) for e in 1:length(g)-1]
    tie = [r[o[e]]==r[o[e+1]] for e in 1:length(d)]
    margin = [ g.p_draw==0.0 ?  0.0 :
               compute_margin(g.p_draw, sqrt( sum([a.beta^2 for a in g.teams[o[e]]]) + sum([a.beta^2 for a in g.teams[o[e+1]]]) )) 
               for e in 1:length(d)] 
    g.evidence = 1
    for e in 1:length(d)
        g.evidence *= !tie[e] ? 1-cdf(d[e].prior, margin[e]) : cdf(d[e].prior, margin[e])-cdf(d[e].prior, -margin[e])
    end
    step = (Inf, Inf)::Tuple{Float64,Float64}; iter = 0::Int64
    while (step > 1e-6) & (iter < 10)
        step = (0., 0.)
        for e in 1:length(d)-1#e=1
            d[e].prior = posterior_win(t[e]) - posterior_lose(t[e+1])
            d[e].likelihood = trunc(d[e].prior,margin[e],tie[e])/d[e].prior
            likelihood_lose = posterior_win(t[e]) - d[e].likelihood
            step = max(step,delta(t[e+1].likelihood_lose,likelihood_lose))
            t[e+1].likelihood_lose = likelihood_lose
        end
        for e in length(d):-1:2
            d[e].prior = posterior_win(t[e]) - posterior_lose(t[e+1])
            d[e].likelihood = trunc(d[e].prior,margin[e],tie[e])/d[e].prior
            likelihood_win = (posterior_lose(t[e+1]) + d[e].likelihood)
            step = max(step,delta(t[e].likelihood_win,likelihood_win))
            t[e].likelihood_win = likelihood_win
        end
        iter += 1
    end
    if length(d)==1
        d[1].prior = posterior_win(t[1]) - posterior_lose(t[2])
        d[1].likelihood = trunc(d[1].prior,margin[1],tie[1])/d[1].prior
    end
    t[1].likelihood_win = (posterior_lose(t[2]) + d[1].likelihood)
    t[end].likelihood_lose = (posterior_win(t[end-1]) - d[end].likelihood)
    
    return [ likelihood(t[o[e]]) for e in 1:length(t)] 
end
function likelihoods(g::Game)
    m_t_ft = likelihood_teams(g)
    g.likelihoods = [[ m_t_ft[e] - exclude(performance(g,e),g.teams[e][i].N) for i in 1:length(g.teams[e])] for e in 1:length(g)]
    return g.likelihoods
end
function posteriors(g::Game)
    return [[ g.likelihoods[e][i] * g.teams[e][i].N for i in 1:length(g.teams[e])] for e in 1:length(g)]
end
mutable struct Batch
    events::Vector{Vector{Vector{String}}}
    results::Vector{Vector{Int64}}
    time::Int64
    elapsed::Dict{String,Int64}
    prior_forward::Dict{String,Rating}
    prior_backward::Dict{String,Gaussian}
    likelihoods::Vector{Vector{Vector{Gaussian}}}
    likelihood::Dict{String,Gaussian}
    old_within_prior::Vector{Vector{Vector{Gaussian}}}
    evidences::Vector{Float64}
    agents::Set{String}
    max_step::Tuple{Float64, Float64}
    function Batch(events::Vector{Vector{Vector{String}}}, results::Vector{Vector{Int64}} ,time::Int64, last_time::Dict{String,Int64}=Dict{String,Int64}() , priors::Dict{String,Rating}=Dict{String,Rating}(), env::Environment=Environment())
        (length(events)!= length(results)) && throw(error("length(events)!= length(results)"))
        
        likelihoods = [ [ [Ninf for a in team ] for team in teams] for teams in events ]
        agents = Set(vcat((events...)...))
        b = new(events, results, time
                   ,Dict{String,Int64}()
                   ,Dict{String,Rating}()
                   ,Dict{String,Gaussian}()
                   ,likelihoods
                   ,Dict{String,Gaussian}()
                   ,Vector{Vector{Vector{Gaussian}}}()
                   ,[0.0 for _ in 1:length(events)]
                   ,agents 
                   ,(Inf, Inf))
        
        b.prior_backward = Dict([ (a, Ninf) for a in b.agents])
        b.elapsed = Dict([ (a, !haskey(last_time, a) ? 0 : ( last_time[a] == -1 ? 1 : (time - last_time[a]) ) ) for a in b.agents ])
        b.prior_forward  = Dict([ (a, !haskey(priors, a) ? Rating(Gaussian(env.mu,env.sigma), env.beta, env.gamma, a) : forget(priors[a],b.elapsed[a]) ) for a in b.agents])
        b.likelihood = Dict([(a, Ninf) for a in b.agents])
        b.old_within_prior = [ [ [Ninf for a in team ] for team in teams] for teams in events ]
        iteration(b)
        b.max_step = step_within_prior(b)
        return b
    end
    function Batch(;events::Vector{Vector{Vector{String}}}, results::Vector{Vector{Int64}} 
                 ,time::Int64=0, last_time::Dict{String,Int64}=Dict{String,Int64}() , priors::Dict{String,Rating}=Dict{String,Rating}(), env::Environment=Environment())
        Batch(events, results, time, last_time, priors, env)
    end
end

Base.show(io::IO, b::Batch) = print("Batch(time=", b.time, ", events=", b.events, ", results=", b.results,")")
Base.length(b::Batch) = length(b.results)
function posterior(b::Batch, agent::String)
    return b.likelihood[agent]*b.prior_backward[agent]*b.prior_forward[agent].N   
end
function posteriors(b::Batch)
    res = Dict{String,Gaussian}()
    for a in b.agents
        res[a] = posterior(b,a)
    end
    return res
end
function within_priors(b::Batch, event::Int64)
    res = Vector{Vector{Rating}}()
    for (t, team) in enumerate(b.events[event])
        res_team = Vector{Rating}()    
        for (j, a) in enumerate(team)
            r = copy(b.prior_forward[a])
            r.N = posterior(b,a)/b.likelihoods[event][t][j]
            push!(res_team, r)
        end
        push!(res, res_team)
    end
    return res
end
function iteration(b::Batch)
    for e in 1:length(b)#e=1
        teams = b.events[e]
        _priors = within_priors(b, e)
        
        for t in 1:length(teams)
            for j in 1:length(teams[t])
                b.old_within_prior[e][t][j] = _priors[t][j].N
            end
        end
        
        g = Game(_priors, b.results[e])
        
        for t in 1:length(teams)
            for j in 1:length(teams[t])
                b.likelihood[teams[t][j]] = (b.likelihood[teams[t][j]] / b.likelihoods[e][t][j]) * g.likelihoods[t][j]
                b.likelihoods[e][t][j] = g.likelihoods[t][j]
            end
        end
        
        b.evidences[e] = g.evidence
    end
end
function step_within_prior(b::Batch)
    step = (0.,0.)::Tuple{Float64,Float64}
    for (e, teams) in enumerate(b.events)
        for (t, team) in enumerate(teams) 
        for (j, a) in enumerate(team) 
            step = max(step, delta(b.old_within_prior[e][t][j], posterior(b,a)/b.likelihoods[e][t][j]) ) 
        end
        end
    end 
    return step
end
function convergence(b::Batch, epsilon::Float64=1e-6)
    iter = 0::Int64    
    while (b.max_step > epsilon) & (iter < 10)
        iteration(b)
        b.max_step = step_within_prior(b)
        iter += 1
    end
    return iter
end
function forward_prior_out(b::Batch, agent::String)#agent="b"
    res = copy(b.prior_forward[agent])
    res.N *= b.likelihood[agent]
    return res
end
function backward_prior_out(b::Batch, agent::String)
    N = b.likelihood[agent]*b.prior_backward[agent]
    # No usar forget ac\'a
    return N+Gaussian(0., b.prior_forward[agent].gamma * sqrt(b.elapsed[agent]) ) 
end
function new_backward_info(b::Batch, backward_message::Dict{String,Gaussian})
    for a in b.agents#a="c"
        b.prior_backward[a] = haskey(backward_message, a) ? backward_message[a] : Ninf
    end
    b.max_step = (Inf, Inf)
    return iteration(b)
end
function new_forward_info(b::Batch, forward_message::Dict{String,Rating}, env::Environment=Environment())
    for a in b.agents
        # TODO: Cuando haskey(forward_message, a) no es verdadero????????????????????????????????????
        b.prior_forward[a] = haskey(forward_message, a) ? forget(forward_message[a],b.elapsed[a]) : Rating(Gaussian(env.mu, env.sigma), env.beta, env.gamma)
    end
    b.max_step = (Inf, Inf)
    return iteration(b)
end

function history_requirements(events::Vector{Vector{Vector{String}}},results::Vector{Vector{Int64}},times::Vector{Int64})
    if length(events) != length(results)
        error("length(events) != length(results)")
    end
    if (length(times) > 0) & (length(events) != length(times))
        error("length(times) > 0) & (length(events) != length(times))")
    end
end

mutable struct History
    size::Int64
    times::Vector{Int64}
    priors::Dict{String,Rating}
    forward_message::Dict{String,Rating}
    backward_message::Dict{String,Gaussian}
    last_time::Dict{String,Int64}#No hay que guardarlo
    batches::Vector{Batch}
    agents::Set{String}
    env::Environment
    function History(events::Vector{Vector{Vector{String}}},results::Vector{Vector{Int64}},times::Vector{Int64}=Int64[],priors::Dict{String,Rating}=Dict{String,Rating}(), env::Environment=Environment())
        history_requirements(events,results,times)
        agents = Set(vcat((events...)...))
        forward_message = copy(priors)
        _h = new(length(events), times, priors, forward_message ,Dict{String,Gaussian}(), Dict{String,Int64}(), Vector{Batch}(), agents, env)
        trueskill(_h, events, results)
        return _h
    end
    function History(;events::Vector{Vector{Vector{String}}},results::Vector{Vector{Int64}},times::Vector{Int64}=Int64[],priors::Dict{String,Rating}=Dict{String,Rating}(), env::Environment=Environment())
        History(events, results, times, priors, env)
    end
end
Base.length(h::History) = h.size
Base.show(io::IO, h::History) = print("History(Size=", h.size
                                     ,", Batches=", length(h.batches)
                                    ,", Agents=", length(h.agents), ")")
function trueskill(h::History, events::Vector{Vector{Vector{String}}},results::Vector{Vector{Int64}})
    o = length(h.times)>0 ? sortperm(h.times) : [i for i in 1:length(events)]
    i = 1::Int64
    while i <= length(h)
        j, t = i, length(h.times) == 0 ? i : h.times[o[i]]
        while ((length(h.times)>0) & (j < length(h)) && (h.times[o[j+1]] == t)) j += 1 end
        b = Batch(events[o[i:j]],results[o[i:j]], t, h.last_time, h.forward_message, h.env)        
        push!(h.batches,b)
        for a in b.agents
            h.last_time[a] = length(h.times) == 0 ? -1 : t
            h.forward_message[a] = forward_prior_out(b,a)
        end
        i = j + 1
    end
end
function diff(old::Dict{String,Gaussian}, new::Dict{String,Gaussian})
    step = (0., 0.)
    for (a, _) in old
        step = max(step, delta(old[a],new[a]))
    end
    return step
end
function iteration(h::History)
    step = (0., 0.)
    h.backward_message=Dict{String,Gaussian}()
    for j in length(h.batches)-1:-1:1# j=3
        for a in h.batches[j+1].agents# a = "c"
            h.backward_message[a] = backward_prior_out(h.batches[j+1],a)
        end
        old = copy(posteriors(h.batches[j]))
        new_backward_info(h.batches[j], h.backward_message)
        step = max(step, diff(old, posteriors(h.batches[j])))
    end
    
    h.forward_message=copy(h.priors)
    for j in 2:length(h.batches)#j=2
        for a in h.batches[j-1].agents#a = "b"
            h.forward_message[a] = forward_prior_out(h.batches[j-1],a)
        end
        old = copy(posteriors(h.batches[j]))
        new_forward_info(h.batches[j], h.forward_message, h.env)
        step = max(step, diff(old, posteriors(h.batches[j])))
    end
    
    if (length(h.batches) == 1)
        iteration(h.batches[1])
        step = step_within_prior(h.batches[1])
    end
        
    return step
end
function convergence(h::History, verbose = true)
    step = (Inf, Inf)::Tuple{Float64,Float64}
    iter = 1::Int64
    while (step > h.env.epsilon) & (iter <= h.env.iter)
        verbose && print("Iteration = ", iter)
        step = iteration(h)
        iter += 1
        verbose && println(", step = ", step)
    end
    println("End")
    return step, iter
end
# function learning_curves(h::History)
#     res = Dict{String,Array{Tuple{Int64,Gaussian}}}()
#     for a in h.agents
#         res[a] = sort([ (t, posterior(b,a)) for (t, b) in h.partake[a]])
#     end
#     return res
# end
function log_evidence(h::History)
   return sum([log(e) for b in h.batches for e in b.evidences])
end

end # module
