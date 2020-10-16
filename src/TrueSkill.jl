module TrueSkill

global const BETA = 1.0::Float64
global const MU = 0.0::Float64
global const SIGMA = (BETA * 6)::Float64
global const GAMMA = (BETA * 0.05)::Float64
global const P_DRAW = 0.0::Float64
global const EPSILON = 1e-6::Float64
global const ITER = 10::Int64
global const sqrt2 = sqrt(2)
global const sqrt2pi = sqrt(2*pi)
global const PI = 1/(SIGMA^2)
global const TAU = MU*PI
global const minInt64 = (-9223372036854775808)::Int64
global const maxInt64 = ( 9223372036854775807)::Int64

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
    """Complementary error function (thanks to http://bit.ly/zOLqbc)"""
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
    if sigma > 0.
        _pi = sigma^-2
        _tau = _pi * mu
    elseif (sigma + 1e-5) < 0.0 
        error("sigma should be greater than 0")
    else
        _pi = Inf
        _tau = Inf
    end
    return _tau, _pi
end
function mu_sigma(_tau::Float64, _pi::Float64)
    if _pi > 0.0
        sigma = sqrt(1/_pi)
        mu = _tau / _pi
    elseif (_pi + 1e-5) < 0.
        error("Precision should be greater than 0")
    else
        sigma = Inf
        mu = 0.0
    end
    return mu, sigma
end

struct Gaussian
    mu::Float64
    sigma::Float64
    function Gaussian(mu::Float64=MU, sigma::Float64=SIGMA)
        if sigma>=0.0
            return new(mu, sigma)
        else
            throw(error("Require: (sigma >= 0.0)"))
        end
    end
    function Gaussian(;mu::Float64=MU, sigma::Float64=SIGMA)
        return new(mu, sigma)
    end
end

global const N01 = Gaussian(0.0, 1.0)
global const Ninf = Gaussian(0.0, Inf)
global const N00 = Gaussian(0.0, 0.0)

Base.show(io::IO, g::Gaussian) = print("Gaussian(mu=", round(g.mu,digits=3)," ,sigma=", round(g.sigma,digits=3), ")")
function _pi_(N::Gaussian)
    if N.sigma > 0.
        return N.sigma^-2
    else
        return Inf
    end
end
function _tau_(N::Gaussian)
    if N.sigma > 0.
        return N.mu * (N.sigma^-2)
    else
        return Inf
    end
end
function cdf(N::Gaussian, x::Float64)
    z = -(x - N.mu) / (N.sigma * sqrt2)
    return (0.5 * erfc(z))::Float64
end
function pdf(N::Gaussian, x::Float64)
    normalizer = (sqrt2pi * N.sigma)^-1
    functional = exp( -((x - N.mu)^2) / (2*N.sigma ^2) ) 
    return (normalizer * functional)::Float64
end
function ppf(N::Gaussian, p::Float64)
    return N.mu - N.sigma * sqrt2  * erfcinv(2 * p)
end 
function trunc(N::Gaussian, margin::Float64, tie::Bool)
    #The range is [alpha, beta]
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
    return Gaussian(mu, sigma)
end
function Base.:-(N::Gaussian, M::Gaussian)
    mu = N.mu - M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu, sigma)
end
function Base.:*(N::Gaussian, M::Gaussian)
    _pi = _pi_(N) + _pi_(M)
    _tau = _tau_(N) + _tau_(M)
    mu, sigma = mu_sigma(_tau, _pi)
    return Gaussian(mu, sigma)        
end
function Base.:/(N::Gaussian, M::Gaussian)
    _pi = _pi_(N) - _pi_(M)
    _tau = _tau_(N) - _tau_(M)
    mu, sigma = mu_sigma(_tau, _pi)
    return Gaussian(mu, sigma)        
end
function Base.isapprox(N::Gaussian, M::Gaussian, atol::Real=0)
    return (abs(N.mu - M.mu) < atol) & (abs(N.sigma - M.sigma) < atol)
end
function forget(N::Gaussian, gamma::Float64, t::Int64)
    return Gaussian(N.mu, sqrt(N.sigma^2 + t*gamma^2))
end 
function compute_margin(p_draw::Float64, sd::Float64)
    _N = Gaussian(0.0, sd )
    res = abs(ppf(_N, 0.5-p_draw/2))
    return res 
end
struct Rating
    N::Gaussian
    beta::Float64
    gamma::Float64
    draw::Gaussian
    function Rating(mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA, gamma::Float64=GAMMA, draw::Gaussian=Ninf)
        Rating(Gaussian(mu, sigma), beta, gamma, draw)
    end
    function Rating(N::Gaussian,beta::Float64=BETA,gamma::Float64=GAMMA,draw::Gaussian=Ninf)
        (N.sigma == 0.0) && throw(error("Rating require: (N.sigma > 0.0)"))
        return new(N, beta, gamma, draw)
    end
    function Rating(;mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA, gamma::Float64=GAMMA, draw::Gaussian=Ninf)
        Rating(Gaussian(mu, sigma), beta, gamma, draw)
    end
end
Base.show(io::IO, r::Rating) = print("Rating(", round(r.N.mu,digits=3)," ,", round(r.N.sigma,digits=3), ")")
function performance(R::Rating)
    return Gaussian(R.N.mu, sqrt(R.N.sigma^2 + R.beta^2))
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


mutable struct Game
    teams::Vector{Vector{Rating}}
    result::Vector{Int64}
    p_draw::Float64
    likelihoods::Vector{Vector{Gaussian}}
    evidence::Float64
    function Game(teams::Vector{Vector{Rating}}, result::Vector{Int64},p_draw::Float64=0.0)
        (length(teams) != length(result)) && throw(error("length(teams) != length(result)"))
        ((0.0 > p_draw) | (1.0 <= p_draw)) &&  throw(error("0.0 <= Draw probability < 1.0"))
        
        _g = new(teams,result,p_draw,[],0.0)
        likelihoods(_g)
        return _g
    end
end        
Base.length(G::Game) = length(G.result)
function size(G::Game)
    return [length(team) for team in g.teams]
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
mutable struct Skill
    forward::Gaussian
    backward::Gaussian
    likelihood::Gaussian
    elapsed::Int64
    function Skill(forward::Gaussian=Ninf, backward::Gaussian=Ninf, likelihood::Gaussian=Ninf, elapsed::Int64=0)
        return new(forward, backward, likelihood, elapsed)
    end
    function Skill(;forward::Gaussian=Ninf, backward::Gaussian=Ninf, likelihood::Gaussian=Ninf, elapsed::Int64=0)
        return new(forward, backward, likelihood, elapsed)
    end
end
mutable struct Agent
    prior::Rating
    message::Gaussian
    last_time::Int64
end
function receive(agent::Agent, elapsed::Int64)
    if agent.message != Ninf
        res = forget(agent.message, agent.prior.gamma, elapsed) 
    else
       res = agent.prior.N
    end
    return res
end
function clean(agents::Dict{String,Agent}, last_time::Bool=false)
    for a in keys(agents)
        agents[a].message = Ninf
        if last_time agents[a].last_time = minInt64 end
    end
end
mutable struct Item
    agent::String
    likelihood::Gaussian
end
mutable struct Team
    items::Vector{Item}
    output::Int64
end
mutable struct Event
    teams::Vector{Team}
    evidence::Float64
end
function outputs(event::Event)
    return [ team.output for team in event.teams]
end
function get_composition(events::Vector{Event})
    return [[[i.agent for i in t.items] for t in e.teams] for e in events]
end
function get_results(events::Vector{Event})
    return [ [t.output for t in e.teams ] for e in events]
end
function compute_elapsed(last_time::Int64, actual_time::Int64)
    return last_time == minInt64 ? 0 : ( last_time == maxInt64 ? 1 : (actual_time - last_time))
end
mutable struct Batch
    time::Int64
    events::Vector{Event}
    skills::Dict{String,Skill}
    agents::Dict{String,Agent}
    function Batch(composition::Vector{Vector{Vector{String}}}, results::Vector{Vector{Int64}} ,time::Int64, agents::Dict{String,Agent}=Dict{String,Agent}(), env::Environment=Environment())
        (length(composition)!= length(results)) && throw(error("length(events)!= length(results)"))
        
        this_agents = Set(vcat((composition...)...))
        elapsed = Dict([ (a, compute_elapsed(agents[a].last_time, time) ) for a in this_agents  ])
        skills = Dict([ (a, Skill(receive(agents[a],elapsed[a]) ,Ninf ,Ninf , elapsed[a])) for a in this_agents  ])
        events = [Event([Team([Item(composition[e][t][a], Ninf) for a in 1:length(composition[e][t]) ] 
                              ,results[e][t]  ) for t in 1:length(composition[e]) ]
                        ,0.0) for e in 1:length(composition) ]
        
        b = new(time, events , skills, agents)

        iteration(b)
        return b
    end
    function Batch(;events::Vector{Vector{Vector{String}}}, results::Vector{Vector{Int64}} 
                 ,time::Int64=0 , agents::Dict{String,Agent}=Dict{String,Agent}(), env::Environment=Environment())
        Batch(events, results, time, agents, env)
    end
end

Base.show(io::IO, b::Batch) = print("Batch(time=", b.time, ", events=", b.events, ")")
Base.length(b::Batch) = length(b.events)

function add_events(b::Batch, composition::Vector{Vector{Vector{String}}}, results::Vector{Vector{Int64}})
    this_agents = Set(vcat((composition...)...))
    for a in this_agents#a="c"
        elapsed = compute_elapsed(b.agents[a].last_time , b.time )  
        if !haskey(b.skills,a)
            b.skills[a] = Skill(receive(b.agents[a],elapsed) ,Ninf ,Ninf , elapsed)
        else
            b.skills[a].elapsed = elapsed
            b.skills[a].forward = receive(b.agents[a],elapsed)
        end
    end
    from = length(b)+1
    for e in 1:length(composition)
        event = Event([Team([Item(composition[e][t][a], Ninf) for a in 1:length(composition[e][t]) ] 
                              ,results[e][t]  ) for t in 1:length(composition[e]) ] , 0.0)
        push!(b.events, event)
    end
    iteration(b, from)
end
function posterior(b::Batch, agent::String)#agent="a_b"
    return b.skills[agent].likelihood*b.skills[agent].backward*b.skills[agent].forward 
end
function posteriors(b::Batch)
    res = Dict{String,Gaussian}()
    for (a, s) in b.skills
        res[a] = posterior(b,a)
    end
    return res
end
function within_prior(b::Batch, item::Item)
    prior = b.agents[item.agent].prior
    return Rating(posterior(b,item.agent)/item.likelihood,prior.beta,prior.gamma)
end
function within_priors(b::Batch, event::Int64)#event=1
    return [ [within_prior(b,item) for item in team.items ] for team in b.events[event].teams ]
end
function iteration(b::Batch, from::Int64 = 1)
    for e in from:length(b)#e=1
        
        g = Game(within_priors(b, e), outputs(b.events[e]))
        
        for (t, team) in enumerate(b.events[e].teams)#(t,team) = (2, b.events[e].teams[2])
            for (i, item) in enumerate(team.items)#(i, item) = (2, team.items[2])
                b.skills[item.agent].likelihood = (b.skills[item.agent].likelihood / item.likelihood) * g.likelihoods[t][i]
                item.likelihood = g.likelihoods[t][i]
            end
        end
        
        b.events[e].evidence = g.evidence
    end
end
function convergence(b::Batch, epsilon::Float64=1e-6, iterations::Int64 = 20)
    iter = 0::Int64    
    step = (Inf, Inf)
    while (step > epsilon) & (iter < iterations)
        old = copy(posteriors(b))
        iteration(b)
        step = diff(old, posteriors(b))
        iter += 1
    end
    return iter
end
function forward_prior_out(b::Batch, agent::String)
    return b.skills[agent].forward * b.skills[agent].likelihood
end
function backward_prior_out(b::Batch, agent::String)
    N = b.skills[agent].likelihood*b.skills[agent].backward
    return forget(N, b.agents[agent].prior.gamma, b.skills[agent].elapsed) 
end
function new_backward_info(b::Batch)
    for a in keys(b.skills)
        b.skills[a].backward = b.agents[a].message
    end
    return iteration(b)
end
function new_forward_info(b::Batch)
    for a in keys(b.skills)
        b.skills[a].forward = receive(b.agents[a], b.skills[a].elapsed) 
    end
    return iteration(b)
end
mutable struct History
    size::Int64
    batches::Vector{Batch}
    agents::Dict{String,Agent}
    env::Environment
    time::Bool
    function History(events::Vector{Vector{Vector{String}}},results::Vector{Vector{Int64}},times::Vector{Int64}=Int64[],priors::Dict{String,Rating}=Dict{String,Rating}(), env::Environment=Environment())
        (length(events) != length(results)) && throw(error("length(events) != length(results)"))
        (length(times) > 0) & (length(events) != length(times)) && throw(error("length(times) > 0) & (length(events) != length(times))"))
        
        agents = Dict([ (a, Agent(haskey(priors, a) ? priors[a] : Rating(env.mu, env.sigma, env.beta, env.gamma), Ninf, minInt64)) for a in Set(vcat((events...)...)) ])
        h = new(length(events), Vector{Batch}(), agents, env, length(times)>0)
        trueskill(h, events, results, times)
        return h
    end
    function History(;events::Vector{Vector{Vector{String}}},results::Vector{Vector{Int64}},times::Vector{Int64}=Int64[],priors::Dict{String,Rating}=Dict{String,Rating}(), env::Environment=Environment())
        History(events, results, times, priors, env)
    end
end

Base.length(h::History) = h.size
Base.show(io::IO, h::History) = print("History(Events=", h.size
                                     ,", Batches=", length(h.batches)
                                    ,", Agents=", length(h.agents), ")")
function trueskill(h::History, composition::Vector{Vector{Vector{String}}},results::Vector{Vector{Int64}}, times::Vector{Int64})
    o = length(times)>0 ? sortperm(times) : [i for i in 1:length(composition)]
    i = 1::Int64
    while i <= length(h)
        j, t = i, length(times) == 0 ? i : times[o[i]]
        while ((length(times)>0) & (j < length(h)) && (times[o[j+1]] == t)) j += 1 end
        b = Batch(composition[o[i:j]],results[o[i:j]], t, h.agents, h.env)        
        push!(h.batches,b)
        for a in keys(b.skills)
            h.agents[a].last_time = length(times) == 0 ? maxInt64 : t
            h.agents[a].message = forward_prior_out(b,a)
        end
        i = j + 1
    end
end
function diff(old::Dict{String,Gaussian}, new::Dict{String,Gaussian})
    step = (0., 0.)
    for a in keys(old)
        step = max(step, delta(old[a],new[a]))
    end
    return step
end
function iteration(h::History)
    step = (0., 0.)
    
    clean(h.agents)
    for j in length(h.batches)-1:-1:1# j = 2 
        for a in keys(h.batches[j+1].skills)
            h.agents[a].message = backward_prior_out(h.batches[j+1],a)
        end
        old = copy(posteriors(h.batches[j]))
        new_backward_info(h.batches[j])
        step = max(step, diff(old, posteriors(h.batches[j])))
    end
    
    clean(h.agents)
    for j in 2:length(h.batches)# j = 3
        for a in keys(h.batches[j-1].skills)
            h.agents[a].message = forward_prior_out(h.batches[j-1],a)
        end
        old = copy(posteriors(h.batches[j]))
        new_forward_info(h.batches[j])
        step = max(step, diff(old, posteriors(h.batches[j])))
    end
    
    if (length(h.batches) == 1)
        old = copy(posteriors(h.batches[1]))
        iteration(h.batches[1])
        step = max(step, diff(old, posteriors(h.batches[1])))
    end
    
    return step
end
function convergence(h::History, verbose = false)
    step = (Inf, Inf)::Tuple{Float64,Float64}
    iter = 1::Int64
    while (step > h.env.epsilon) & (iter <= h.env.iter)
        verbose && print("Iteration = ", iter)
        step = iteration(h)
        iter += 1
        verbose && println(", step = ", step)
    end
    verbose && println("End")
    return step, iter
end
function learning_curves(h::History)
    res = Dict{String,Array{Tuple{Int64,Gaussian}}}()
    for b in h.batches
        for a in keys(b.skills)
            t_p = (b.time, posterior(b,a))
            res[a] = haskey(res, a) ? push!(res[a],t_p) : [t_p]
        end
    end
    return res
end
function log_evidence(h::History)
   return sum([log(event.evidence) for b in h.batches for event in b.events])
end
function add_events(h::History,composition::Vector{Vector{Vector{String}}},results::Vector{Vector{Int64}},times::Vector{Int64}=Int64[],priors::Dict{String,Rating}=Dict{String,Rating}())
    
    (length(times)>0) & !h.time && throw(error("length(times)>0 but !h.time"))
    (length(times)==0) & h.time && throw(error("length(times)==0 but h.time"))
    (length(composition) != length(results)) && throw(error("length(composition) != length(results)"))
    (length(times) > 0) & (length(composition) != length(times)) && throw(error("length(times) > 0) & (length(composition) != length(times))"))
        
        
    this_agents = Set(vcat((composition...)...))
    for a in this_agents
        if !haskey(h.agents,a)
            h.agents[a] = Agent(haskey(priors, a) ? priors[a] : Rating(h.env.mu, h.env.sigma, h.env.beta, h.env.gamma), Ninf, minInt64)
        end
    end
    
    clean(h.agents,true)
    n = length(composition)
    o = length(times)>0 ? sortperm(times) : [i for i in 1:length(composition)]
    i = 1::Int64; k = 1::Int64
    while i <= n
        j, t = i, length(times) == 0 ? i : times[o[i]]
        while ((length(times)>0) & (j < n) && (times[o[j+1]] == t)) j += 1 end
        while (!h.time & (h.size > k) ) || (h.time && (length(h.batches) >= k) && (h.batches[k].time < t))
            b = h.batches[k]
            if (k>1) new_forward_info(b) end
            for a in intersect(keys(b.skills), this_agents)#a ="a"
                b.skills[a].elapsed = compute_elapsed(b.agents[a].last_time , b.time )
                h.agents[a].last_time = length(times) == 0 ? maxInt64 : b.time
                h.agents[a].message = forward_prior_out(b,a)
            end
            k += 1
        end
        if (h.time && (length(h.batches) >= k) && (h.batches[k].time == t))
            b = h.batches[k]
            add_events(b, composition[o[i:j]],results[o[i:j]])
        else
            if !h.time k = k + 1 end
            b =  Batch(composition[o[i:j]],results[o[i:j]], t, h.agents, h.env)        
            insert!(h.batches,  k , b)
            if h.time k = k + 1 end
        end
        for a in keys(b.skills)#a="a"
            h.agents[a].last_time = length(times) == 0 ? maxInt64 : t
            h.agents[a].message = forward_prior_out(b,a)
        end
        i = j + 1
    end
    while h.time && (length(h.batches) >= k) 
        b = h.batches[k]
        new_forward_info(b)
        for a in intersect(keys(b.skills), this_agents)#a ="a"
            b.skills[a].elapsed = compute_elapsed(b.agents[a].last_time , b.time )
            h.agents[a].last_time = length(times) == 0 ? maxInt64 : b.time
            h.agents[a].message = forward_prior_out(b,a)
        end
        k += 1
    end
    h.size = h.size + n
    iteration(h)
end

# 
# ta = [Rating(0.0,1.0),Rating(0.0,1.0),Rating(0.0,1.0)]
# tb = [Rating(0.0,1.0),Rating(0.0,1.0),Rating(0.0,1.0)]
# posteriors(Game([ta,tb],[1,0]))




end # module
