module TrueSkillThroughTime

export BETA, MU, SIGMA, GAMMA, P_DRAW
export Gaussian, +, -, /, *, isapprox, forget
export Player, performance
export Game, posteriors, performance
export History, convergence, learning_curves, log_evidence

"""
The default standar deviation of the performances is

    global const BETA = 1.0
    
This parameter acts as the scale of the estimates.
A real difference of one `beta` between two skills is equivalent to 76% probability of winning.
"""
global const BETA = 1.0::Float64
"""
The default mean of the priors is

    global const MU = 0.0
    
used by the [`Gaussian` class](@ref gaussian)
"""
global const MU = 0.0::Float64
"""
The default standar deviation of the priors is

    global const SIGMA = (BETA * 6)

used by the [`Gaussian` class](@ref gaussian)
"""
global const SIGMA = (BETA * 6)::Float64
"""
The default amount of uncertainty (standar deviation) added to the estimates as time progresses

    global const GAMMA = (BETA * 0.03)
"""
global const GAMMA = (BETA * 0.03)::Float64
"""
The default probability of a draw is 
    
    global const P_DRAW = 0.0
"""
global const P_DRAW = 0.0::Float64
global const EPSILON = 1e-6::Float64
global const ITERATIONS = 30::Int64
global const sqrt2 = sqrt(2)
global const sqrt2pi = sqrt(2*pi)
global const PI = 1/(SIGMA^2)
global const TAU = MU*PI
global const minInt64 = (-9223372036854775808)::Int64
global const maxInt64 = ( 9223372036854775807)::Int64



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

"""
The `Gaussian` class is used to define the prior beliefs of the agents' skills (and for internal computations).

We can create objects by passing the parameters in order or by mentioning the names. 

    Gaussian(mu::Float64=MU, sigma::Float64=SIGMA)
    Gaussian(;mu::Float64=MU, sigma::Float64=SIGMA)

- `mu` is the mean of the Gaussian distribution
- `sigma` is the standar deviation of the Gaussian distribution
"""
struct Gaussian
    mu::Float64
    sigma::Float64
    function Gaussian(mu::Float64=MU, sigma::Float64=SIGMA)
        if isnan(mu) throw(error("Require: mu must be a number, was NaN")) end
        if isnan(sigma) throw(error("Require: sigma must be a number, was NaN")) end
        if isinf(mu) throw(error("Require: (-Inf < mu < Inf)")) end
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

Base.show(io::IO, g::Gaussian) = print(io, "Gaussian(mu=", round(g.mu,digits=6), ", sigma=", round(g.sigma,digits=6), ")")
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
"""
    cdf(N::Gaussian, x::Float64)

The cumulative density function of the Gaussian distribution
"""
function cdf(N::Gaussian, x::Float64)
    z = -(x - N.mu) / (N.sigma * sqrt2)
    return (0.5 * erfc(z))::Float64
end
function pdf(N::Gaussian, x::Float64)
    normalizer = (sqrt(2*pi) * N.sigma)^-1
    functional = exp( -((x - N.mu)^2) / (2*N.sigma ^2) ) 
    return (normalizer * functional)::Float64
end
function ppf(N::Gaussian, p::Float64)
    return N.mu - N.sigma * sqrt2  * erfcinv(2 * p)
end 
function approx(;N::Gaussian, margin::Float64, tie::Bool)
    approx(N, margin, tie)
end
function approx(N::Gaussian, margin::Float64, tie::Bool)
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
"""
    +(N::Gaussian, M::Gaussian)
"""
function Base.:+(N::Gaussian, M::Gaussian)
    mu = N.mu + M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu, sigma)
end
"""
    -(N::Gaussian, M::Gaussian)
"""
function Base.:-(N::Gaussian, M::Gaussian)
    mu = N.mu - M.mu
    sigma = sqrt(N.sigma^2 + M.sigma^2)
    return Gaussian(mu, sigma)
end
"""
    *(N::Gaussian, M::Gaussian)
"""
function Base.:*(N::Gaussian, M::Gaussian)
    if N.sigma == 0.0|| M.sigma == 0.0
        mu = N.mu/(N.sigma^2/M.sigma^2 + 1) + M.mu/(M.sigma^2/N.sigma^2 + 1)
        sigma = sqrt(1/((1/N.sigma^2) + (1/M.sigma^2)))
    else
        _pi = _pi_(N) + _pi_(M)
        _tau = _tau_(N) + _tau_(M)
        mu, sigma = mu_sigma(_tau, _pi)
    end
    return Gaussian(mu, sigma)        
end
"""
    +(k::Float64, M::Gaussian)
"""
function Base.:*(k::Float64, M::Gaussian)
    if isinf(k) return Ninf end
    return Gaussian(k*M.mu, abs(k)*M.sigma)
end
function Base.:*(M::Gaussian, k::Float64)
    return k*M
end
"""
    /(N::Gaussian, M::Gaussian)
"""
function Base.:/(N::Gaussian, M::Gaussian)
    if N.sigma == 0.0|| M.sigma == 0.0
        mu = N.mu/(1 - N.sigma^2/M.sigma^2) - M.mu/(M.sigma^2/N.sigma^2 - 1)
        sigma =  sqrt(1/((1/N.sigma^2) - (1/M.sigma^2)))
    else
        _pi = _pi_(N) - _pi_(M)
        _tau = _tau_(N) - _tau_(M)
        mu, sigma = mu_sigma(_tau, _pi)
    end
    return Gaussian(mu, sigma)        
end
"""
    forget(N::Gaussian, gamma::Float64, t::Int64=1)
"""
function forget(N::Gaussian, gamma::Float64, t::Int64=1)
    return Gaussian(N.mu, sqrt(N.sigma^2 + t*gamma^2))
end
"""
    isapprox(N::Gaussian, M::Gaussian, atol::Real=0)
"""
function Base.isapprox(N::Gaussian, M::Gaussian, atol::Real=0)
    return (abs(N.mu - M.mu) < atol) & (abs(N.sigma - M.sigma) < atol)
end
function compute_margin(p_draw::Float64, sd::Float64)
    _N = Gaussian(0.0, sd )
    res = abs(ppf(_N, 0.5-p_draw/2))
    return res 
end
"""
The `Player` class is used to define the features of the agents. We can create objects by indicating the parameters in order or by mentioning their names. 

    Player(prior::Gaussian=Gaussian(MU,SIGMA), beta::Float64=BETA, gamma::Float64=GAMMA)
    Player(;prior::Gaussian=Gaussian(MU,SIGMA), beta::Float64=BETA, gamma::Float64=GAMMA)
    
- `prior` is the prior belief distribution of skill hypotheses
- `beta` is the standar deviation of the agent performance
- `gamma` is the uncertainty (standar deviation) added to the estimates as time progresses
"""
struct Player
    prior::Gaussian
    beta::Float64
    gamma::Float64
    draw::Gaussian
    function Player(prior::Gaussian=Gaussian(MU,SIGMA), beta::Float64=BETA, gamma::Float64=GAMMA, draw::Gaussian=Ninf)
        new(prior, beta, gamma, draw)
    end
    function Player(;prior::Gaussian=Gaussian(MU,SIGMA), beta::Float64=BETA, gamma::Float64=GAMMA, draw::Gaussian=Ninf)
        Player(prior, beta, gamma, draw)
    end
end
Base.show(io::IO, r::Player) = print(io, "Player(Gaussian(mu=", round(r.prior.mu,digits=3),", sigma=", round(r.prior.sigma,digits=3), "), beta=", round(r.beta, digits=3), ", gamma=" , round(r.gamma, digits=3),")")
"""
    performance(R::Player)
"""
function performance(R::Player)
    return forget(R.prior, R.beta)
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

function evidence(d::Vector{diff_messages}, margin::Vector{Float64}, tie::Vector{Bool}, e::Int64)
    return !tie[e] ? 1-cdf(d[e].prior, margin[e]) : cdf(d[e].prior, margin[e])-cdf(d[e].prior, -margin[e])
end
"""
The `Game` class

    Game(teams::Vector{Vector{Player}}, result::Vector{Float64}, p_draw::Float64, weights::Vector{Vector{Float64}})

Properties:
- `teams::Vector{Vector{Player}}`
- `result::Vector{Float64}`
- `p_draw::Float64`
- `weights::Vector{Vector{Float64}}`
- `likelihoods::Vector{Vector{Gaussian}}`
- `evidence::Float64`
"""
mutable struct Game
    teams::Vector{Vector{Player}}
    result::Vector{Float64}
    weights::Vector{Vector{Float64}}
    p_draw::Float64
    likelihoods::Vector{Vector{Gaussian}}
    evidence::Float64
    function Game(teams::Vector{Vector{Player}}, result::Vector{Float64}=Float64[],p_draw::Float64=0.0,weights::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
        ((0.0 > p_draw) | (1.0 <= p_draw)) &&  throw(error("0.0 <= Draw probability < 1.0"))
        (length(result)>0) && (length(teams)!= length(result)) && throw(error("(length(result)>0) & (length(teams)!= length(result))"))
        (length(weights)>0) && (length(teams)!= length(weights)) && throw(error("(length(weights)>0) & (length(teams)!= length(weights))"))
        (length(weights)>0) && (any([length(team) != length(weight) for (team, weight) in zip(teams, weights)])) && throw(error("(length(weights)>0) & exists i (length(teams[i]) != length(weights[i])"))
        (p_draw == 0.0) && (length(result)>0) && (length(unique(result)) != length(result)) && throw(error("(p_draw == 0.0) && (length(result)>0) && (length(unique(result)) != length(result))"))
        if isempty(weights)
            weights = [[1.0 for p in t] for t in teams]
        end
        _g = new(teams,result,weights,p_draw,[],0.0)
        likelihoods(_g)
        return _g
    end
    function Game(teams::Vector{Vector{Player}}, result::Vector{Float64}=Float64[];p_draw::Float64=0.0,weights::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
        Game(teams, result, p_draw, weights)
    end
    function Game(teams::Vector{Vector{Player}}; result::Vector{Float64}=Float64[],p_draw::Float64=0.0,weights::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
        Game(teams, result, p_draw, weights)
    end
end        
Base.length(G::Game) = length(G.teams)
function size(G::Game)
    return [length(team) for team in g.teams]
end
function performance(team::Vector{Player}, weights::Vector{Float64})
    p = N00
    for (r, w) in zip(team, weights)
        p += w * performance(r)
    end
    return p
end
"""
    performance(G::Game,i::Int64)
"""
function performance(G::Game,i::Int64)
    return performance(G.teams[i], G.weights[i])
end 
function draw_performance(G::Game,i::Int64)
    res = N00
    for r in G.teams[i]
        res += r.draw.sigma < Inf ? approx(r.draw,0.,false) : Ninf
    end
    return res
end 
function likelihood_teams(g::Game)
    r = g.result == Float64[] ? [Float64(i) for i in length(g.teams):-1:1] : g.result
    o = sortperm(r, rev=true)
    t = [team_messages(performance(g,o[e]), Ninf, Ninf, Ninf) for e in 1:length(g)]
    d = [diff_messages(t[e].prior - t[e+1].prior, Ninf) for e in 1:length(g)-1]
    tie = [r[o[e]]==r[o[e+1]] for e in 1:length(d)]
    margin = [ g.p_draw==0.0 ?  0.0 :
               compute_margin(g.p_draw, sqrt( sum([a.beta^2 for a in g.teams[o[e]]]) + sum([a.beta^2 for a in g.teams[o[e+1]]]) )) 
               for e in 1:length(d)] 
    g.evidence = 1.0
    step = (Inf, Inf)::Tuple{Float64,Float64}; iter = 0::Int64
    while (step > 1e-6) & (iter < 10)
        step = (0., 0.)
        for e in 1:length(d)-1#e=1
            d[e].prior = posterior_win(t[e]) - posterior_lose(t[e+1])
            if iter == 0 g.evidence *= evidence(d, margin, tie, e) end
            d[e].likelihood = approx(d[e].prior,margin[e],tie[e])/d[e].prior
            likelihood_lose = posterior_win(t[e]) - d[e].likelihood
            step = max(step,delta(t[e+1].likelihood_lose,likelihood_lose))
            t[e+1].likelihood_lose = likelihood_lose
        end
        for e in length(d):-1:2
            d[e].prior = posterior_win(t[e]) - posterior_lose(t[e+1])
            if (iter == 0) & (e == length(d)) g.evidence *= evidence(d, margin, tie, e) end
            d[e].likelihood = approx(d[e].prior,margin[e],tie[e])/d[e].prior
            likelihood_win = (posterior_lose(t[e+1]) + d[e].likelihood)
            step = max(step,delta(t[e].likelihood_win,likelihood_win))
            t[e].likelihood_win = likelihood_win
        end
        iter += 1
    end
    if length(d)==1
        g.evidence = evidence(d, margin, tie, 1)
        d[1].prior = posterior_win(t[1]) - posterior_lose(t[2])
        d[1].likelihood = approx(d[1].prior,margin[1],tie[1])/d[1].prior
    end
    t[1].likelihood_win = (posterior_lose(t[2]) + d[1].likelihood)
    t[end].likelihood_lose = (posterior_win(t[end-1]) - d[end].likelihood)
    
    return [ likelihood(t[o[e]]) for e in 1:length(t)] 
end
function likelihoods(team::Vector{Player}, weights::Vector{Float64}, message::Gaussian)
    team_perf = performance(team, weights)
    return [
        forget(
            (1/w) * (message - exclude(team_perf, w*performance(p))),
            p.beta)
        for (p,w) in zip(team, weights)
    ]
end
function likelihoods(g::Game)
    m_t_ft = likelihood_teams(g)
    g.likelihoods = [likelihoods(t, w, m) for (t,w,m) in zip(g.teams, g.weights, m_t_ft)]
    return g.likelihoods
end
"""
    posteriors(g::Game)
"""
function posteriors(g::Game)
    return [[ g.likelihoods[e][i] * g.teams[e][i].prior for i in 1:length(g.teams[e])] for e in 1:length(g)]
end
mutable struct Skill
    forward::Gaussian
    backward::Gaussian
    likelihood::Gaussian
    elapsed::Int64
    online::Gaussian
    function Skill(forward::Gaussian=Ninf, backward::Gaussian=Ninf, likelihood::Gaussian=Ninf, elapsed::Int64=0)
        return new(forward, backward, likelihood, elapsed)
    end
    function Skill(;forward::Gaussian=Ninf, backward::Gaussian=Ninf, likelihood::Gaussian=Ninf, elapsed::Int64=0)
        return new(forward, backward, likelihood, elapsed)
    end
end
mutable struct Agent
    player::Player
    message::Gaussian
    last_time::Int64
end
function receive(agent::Agent, elapsed::Int64)
    if agent.message != Ninf
        res = forget(agent.message, agent.player.gamma, elapsed) 
    else
       res = agent.player.prior
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
    output::Float64
end
mutable struct Event
    teams::Vector{Team}
    evidence::Float64
    weights::Vector{Vector{Float64}}
    function Event(teams, evidence, weights=Vector{Vector{Float64}}())
       new(teams, evidence, weights) 
    end
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
    p_draw::Float64
    function Batch(composition::Vector{Vector{Vector{String}}}, results::Vector{Vector{Float64}}=Vector{Vector{Float64}}() ,time::Int64=0, agents::Dict{String,Agent}=Dict{String,Agent}(), p_draw::Float64 = 0.0, weights::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}())
        (length(results)>0) & (length(composition)!= length(results)) && throw(error("(length(results)>0) & (length(composition)!= length(results))"))
        (length(weights)>0) & (length(composition)!= length(weights)) && throw(error("(length(weights)>0) & (length(composition)!= length(weights))"))
        
        this_agents = Set(vcat((composition...)...))
        elapsed = Dict([ (a, compute_elapsed(agents[a].last_time, time) ) for a in this_agents  ])
        skills = Dict([ (a, Skill(receive(agents[a],elapsed[a]) ,Ninf ,Ninf , elapsed[a])) for a in this_agents  ])
        events = [Event([Team([Item(composition[e][t][a], Ninf) for a in 1:length(composition[e][t]) ] 
                              , length(results) > 0 ? results[e][t] : Float64(length(composition[e])-t)  ) for t in 1:length(composition[e]) ]
                        ,0.0, isempty(weights) ? weights : weights[e]) for e in 1:length(composition) ]
        
        b = new(time, events , skills, agents, p_draw)

        iteration(b)
        return b
    end
    function Batch(;composition::Vector{Vector{Vector{String}}}, results::Vector{Vector{Float64}} 
                 ,time::Int64=0 , agents::Dict{String,Agent}=Dict{String,Agent}(), p_draw::Float64 = 0.0, weights::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}())
        Batch(composition, results, time, agents, p_draw, weights)
    end
end

Base.show(io::IO, b::Batch) = print(io, "Batch(time=", b.time, ", events=", b.events, ")")
Base.length(b::Batch) = length(b.events)

function add_events(b::Batch, composition::Vector{Vector{Vector{String}}}, results::Vector{Vector{Float64}}=Vector{Vector{Float64}}(), weights::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}())
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
                              , length(results) > 0 ? results[e][t] : Float64(length(composition[e])-t) ) for t in 1:length(composition[e]) ]
                      , 0.0, isempty(weights) ? weights : weights[e])
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
function within_prior(b::Batch, item::Item, online = false, forward = false)
    r = b.agents[item.agent].player
    if online
        return Player(b.skills[item.agent].online,r.beta,r.gamma)
    elseif forward  
        return Player(b.skills[item.agent].forward,r.beta,r.gamma)
    else
        wp = posterior(b,item.agent)/item.likelihood
        return Player(wp,r.beta,r.gamma)
    end
end
function within_priors(b::Batch, event::Int64; online = false, forward = false)#event=1
    return [ [within_prior(b,item,online,forward) for item in team.items ] for team in b.events[event].teams ]
end
function iteration(b::Batch, from::Int64 = 1)
    for e in from:length(b)#e=1
        
        g = Game(within_priors(b, e), outputs(b.events[e]), b.p_draw, b.events[e].weights)
        
        for (t, team) in enumerate(b.events[e].teams)#(t,team) = (2, b.events[e].teams[2])
            for (i, item) in enumerate(team.items)#(i, item) = (2, team.items[2])
                b.skills[item.agent].likelihood = (b.skills[item.agent].likelihood / item.likelihood) * g.likelihoods[t][i]
                item.likelihood = g.likelihoods[t][i]
            end
        end
        
        b.events[e].evidence = g.evidence
    end
end
function log_evidence2(b::Batch, online::Bool; agents::Vector{String} = Vector{String}(), forward::Bool = false)
    if isempty(agents)
        if online | forward
            return sum([log(Game(within_priors(b, e, online=online, forward=forward), outputs(b.events[e]), b.p_draw, b.events[e].weights).evidence) for e in 1:length(b)])
        else
            return sum([log(event.evidence) for event in b.events])
        end
    else
        filter = [!isdisjoint(vcat((comp...)...),agents) for comp in get_composition(b.events)]
        if online | forward
            return sum([log(Game(within_priors(b, e, online=online, forward=forward), outputs(b.events[e]), b.p_draw, b.events[e].weights).evidence) for e in 1:length(b) if filter[e] ])
        else
            return sum([log(b.events[e].evidence) for e in 1:length(b.events) if filter[e]])
        end
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
    return forget(N, b.agents[agent].player.gamma, b.skills[agent].elapsed) 
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
"""
The `History` class
    
    History(composition::Vector{Vector{Vector{String}}},
    results::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
    times::Vector{Int64}=Int64[], priors::Dict{String,Player}=Dict{String,Player}()
    ; mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA,
    gamma::Float64=GAMMA, p_draw::Float64=P_DRAW, online::Bool=false,
    weights::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}())

Properties:

    size::Int64
    batches::Vector{Batch}
    agents::Dict{String,Agent}
    time::Bool
    mu::Float64
    sigma::Float64
    beta::Float64
    gamma::Float64
    p_draw::Float64
    online::Bool

"""
mutable struct History
    size::Int64
    batches::Vector{Batch}
    agents::Dict{String,Agent}
    time::Bool
    mu::Float64
    sigma::Float64
    beta::Float64
    gamma::Float64
    p_draw::Float64
    online::Bool
    weights::Vector{Vector{Vector{Float64}}}
    epsilon::Float64
    iterations::Int64
    function History(composition::Vector{Vector{Vector{String}}}, results::Vector{Vector{Float64}}=Vector{Vector{Float64}}(), times::Vector{Int64}=Int64[], priors::Dict{String,Player}=Dict{String,Player}(); mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA, gamma::Float64=GAMMA, p_draw::Float64=P_DRAW, online::Bool=false, weights::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}(), epsilon::Float64=EPSILON, iterations::Int64=ITERATIONS)
        (length(results) > 0) & (length(composition) != length(results)) && throw(error("(length(times) > 0) & (length(composition) != length(results))"))
        (length(weights) > 0) & (length(composition) != length(weights)) && throw(error("(length(weights) > 0) & (length(composition) != length(weights))"))
        (length(times) > 0) & (length(composition) != length(times)) && throw(error("length(times) > 0) & (length(composition) != length(times))"))
        
        agents = Dict([ (a, Agent(haskey(priors, a) ? priors[a] : Player(Gaussian(mu, sigma), beta, gamma), Ninf, minInt64)) for a in Set(vcat((composition...)...)) ])
        h = new(length(composition), Vector{Batch}(), agents, length(times)>0, mu, sigma, beta, gamma, p_draw, online, weights, epsilon, iterations)
        trueskill(h, composition, results, times, online, weights, epsilon, iterations)
        return h
    end
    function History(;composition::Vector{Vector{Vector{String}}},results::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),times::Vector{Int64}=Int64[],priors::Dict{String,Player}=Dict{String,Player}(), mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA, gamma::Float64=GAMMA, p_draw::Float64=P_DRAW, online::Bool=false, weights::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}(), epsilon::Float64=EPSILON, iterations::Int64=ITERATIONS)
        History(composition, results, times, priors, mu=mu, sigma=sigma, beta=beta, gamma=gamma, p_draw=p_draw, online=online, weights=weights, epsilon=epsilon, iterations=iterations)
    end
end

# History(composition::Vector{Vector{Vector{String}}}, results::Vector{Vector{Int64}}=Vector{Vector{Int64}}(), times::Vector{Int64}=Int64[], priors::Dict{String,Player}=Dict{String,Player}(); mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA, gamma::Float64=GAMMA, p_draw::Float64=P_DRAW, online::Bool=false) = History(composition, convert(Vector{Vector{Float64}},results), times, priors, mu=mu, sigma=sigma, beta=beta, gamma=gamma, p_draw=p_draw, online=online)
#     
# History(;composition::Vector{Vector{Vector{String}}}, results::Vector{Vector{Int64}}=Vector{Vector{Int64}}(), times::Vector{Int64}=Int64[], priors::Dict{String,Player}=Dict{String,Player}(), mu::Float64=MU, sigma::Float64=SIGMA, beta::Float64=BETA, gamma::Float64=GAMMA, p_draw::Float64=P_DRAW, online::Bool=false) = History(composition, convert(Vector{Vector{Float64}},results), times, priors, mu=mu, sigma=sigma, beta=beta, gamma=gamma, p_draw=p_draw, online=online)
    
Base.length(h::History) = h.size
Base.show(io::IO, h::History) = print(io, "History(Events=", h.size
                                     ,", Batches=", length(h.batches)
                                    ,", Agents=", length(h.agents), ")")
function trueskill(h::History, composition::Vector{Vector{Vector{String}}},results::Vector{Vector{Float64}}, times::Vector{Int64}, online::Bool, weights::Vector{Vector{Vector{Float64}}}, epsilon::Float64, iterations::Int64)
    o = length(times)>0 ? sortperm(times) : [i for i in 1:length(composition)]
    i = 1::Int64
    last = 0.0
    while i <= length(h)
        j, t = i, length(times) == 0 ? i : times[o[i]]
        while ((length(times)>0) & (j < length(h)) && (times[o[j+1]] == t)) j += 1 end
        if length(results)>0
            b = Batch(composition[o[i:j]],results[o[i:j]], t, h.agents, h.p_draw, isempty(weights) ? weights : weights[o[i:j]])
        else
            b = Batch(composition[o[i:j]], results , t, h.agents, h.p_draw, isempty(weights) ? weights : weights[o[i:j]])
        end
        push!(h.batches,b)
        if online
            new = round(100*(i/length(h)))
            if new != last
                print("\r",new,"%")
                last = new
            end
            for a in keys(b.skills)
                b.skills[a].online = b.skills[a].forward
            end
            convergence(h, iterations=h.iterations, epsilon=h.epsilon)
        end
        for a in keys(b.skills)
            h.agents[a].last_time = length(times) == 0 ? maxInt64 : t
            h.agents[a].message = forward_prior_out(b,a)
        end
        i = j + 1
    end
    if online println("\r100.0%") end
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
"""
    convergence(h::History; epsilon::Float64=EPSILON,
    iterations::Int64=ITERATIONS; epsilon::Float64=EPSILON, iterations::Int64=ITERATIONS, verbose = true)
"""
function convergence(h::History; epsilon::Float64=h.epsilon, iterations::Int64=h.iterations, verbose = true)
    step = (Inf, Inf)::Tuple{Float64,Float64}
    iter = 1::Int64
    while (step > epsilon) & (iter <= iterations)
        verbose && print("Iteration = ", iter)
        step = iteration(h)
        iter += 1
        verbose && println(", step = ", step)
    end
    verbose && println("End")
    return step, iter
end
# function posteriors(h::History)
#     res = Vector{Vector{Vector{Gaussian}}}()
#     for b in h.batches
#         for e in b.events
#             ps = Vector{Vector{Gaussian}}()
#             for a in get_composition(e)
#             
#                 t_p = (b.time, posterior(b,a))
#                 res[a] = haskey(res, a) ? push!(res[a],t_p) : [t_p]
#         end
#     end
# end
"""
    learning_curves(h::History)
"""
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
"""
    log_evidence(h::History; agents::Vector{String} = Vector{String}(), forward::Bool = false)
"""
function log_evidence(h::History; forward::Bool = false, agents::Vector{String} = Vector{String}() )
    return sum([log_evidence2(b, h.online, agents = agents, forward=forward) for b in h.batches])
end

function add_events(h::History,composition::Vector{Vector{Vector{String}}};results::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),times::Vector{Int64}=Int64[],priors::Dict{String,Player}=Dict{String,Player}(), weights::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}())
    add_events(h, composition, results, times, priors, weights)
end

function add_events(h::History,composition::Vector{Vector{Vector{String}}},results::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),times::Vector{Int64}=Int64[],priors::Dict{String,Player}=Dict{String,Player}(), weights::Vector{Vector{Vector{Float64}}}=Vector{Vector{Vector{Float64}}}())
    
    (length(times)>0) & !h.time && throw(error("length(times)>0 but !h.time"))
    (length(times)==0) & h.time && throw(error("length(times)==0 but h.time"))
    (length(results) > 0) & (length(composition) != length(results)) && throw(error("(length(results) > 0) & (length(composition) != length(results))"))
    (length(weights) > 0) & (length(composition) != length(weights)) && throw(error("(length(weights) > 0) & (length(composition) != length(weights))"))
    (length(times) > 0) & (length(composition) != length(times)) && throw(error("length(times) > 0) & (length(composition) != length(times))"))
        
        
    this_agents = Set(vcat((composition...)...))
    for a in this_agents
        if !haskey(h.agents,a)
            h.agents[a] = Agent(haskey(priors, a) ? priors[a] : Player(Gaussian(h.mu, h.sigma), h.beta, h.gamma), Ninf, minInt64)
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
            if length(results)>0
                add_events(b, composition[o[i:j]], results[o[i:j]], isempty(weights) ? weights : weights[o[i:j]])
            else
                add_events(b, composition[o[i:j]], results, isempty(weights) ? weights : weights[o[i:j]])
            end
        else
            if !h.time k = k + 1 end
            if length(results)>0
                b =  Batch(composition[o[i:j]],results[o[i:j]], t, h.agents, h.p_draw, isempty(weights) ? weights : weights[o[i:j]])
            else
                b =  Batch(composition[o[i:j]], results , t, h.agents, h.p_draw, isempty(weights) ? weights : weights[o[i:j]])
            end
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




end # module
