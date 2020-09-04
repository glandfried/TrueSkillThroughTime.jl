    #
    # Sin TESTEAR. TTT-D
    #
    
    function likelihood_teams_draw(g::Game)
        r = g.result
        o = sortperm(r)
        t = [team_messages(performance(g,o[e]), Ninf, Ninf, Ninf) for e in 1:length(g)]
        u = [draw_messages(draw_performance(g,o[e]), draw_performance(g,o[e]) + t[e].prior, Ninf, Ninf) for e in 1:length(g)]
        tie = [r[o[e]]==r[o[e+1]] for e in 1:length(g)-1]
        d = [(diff_messages(Ninf, Ninf), diff_messages(Ninf, Ninf),) for e in 1:length(tie) ]
        step = (Inf, Inf)::Tuple{Float64,Float64}; iter = 0::Int64
        
        while (step > 1e-6) & (iter < 20)
            step = (0., 0.)
            for e in 1:length(d)#e=2
                if !tie[e]
                    #TODO: crear par\'ametros por defecto para trunc()
                    d[e][1].prior = posterior_win(t[e]) - posterior_lose(u[e+1])
                    d[e][1].likelihood = trunc(d[e][1].prior,0.,false)/d[e][1].prior
                    u[e+1].likelihood_lose =  posterior_win(t[e]) - d[e][1].likelihood
                else
                    d[e][1].prior = posterior_win(u[e]) - posterior_lose(t[e+1])
                    d[e][1].likelihood = trunc(d[e][1].prior,0.,false)/d[e][1].prior
                    t[e+1].likelihood_lose = posterior_win(u[e]) - d[e][1].likelihood
                    d[e][2].prior = posterior_win(u[e+1]) - posterior_lose(t[e])
                    d[e][2].likelihood = trunc(d[e][2].prior,0.,false)/d[e][2].prior
                    u[e+1].likelihood_win = posterior_lose(t[e]) + d[e][2].likelihood
                end
                t[e+1].likelihood_draw = likelihood(u[e+1]) - u[e+1].prior
            end
            d21_likelihood = d[2][1].likelihood
            for e in length(d):-1:1
                if !tie[e]
                    d[e][1].prior = posterior_win(t[e]) - posterior_lose(u[e+1])
                    d[e][1].likelihood = trunc(d[e][1].prior,0.,false)/d[e][1].prior
                    t[e].likelihood_win = posterior_lose(u[e+1]) + d[e][1].likelihood
                else
                    d[e][1].prior = posterior_win(u[e]) - posterior_lose(t[e+1])
                    d[e][1].likelihood = trunc(d[e][1].prior,0.,false)/d[e][1].prior
                    u[e].likelihood_win = posterior_lose(t[e+1]) + d[e][1].likelihood
                    d[e][2].prior = posterior_win(u[e+1]) - posterior_lose(t[e])
                    d[e][2].likelihood = trunc(d[e][2].prior,0.,false)/d[e][2].prior
                    t[e].likelihood_lose = posterior_win(u[e+1]) - d[e][2].likelihood
                end
                u[e].prior_team = posterior_draw(t[e]) + u[e].prior
            end
            step = max(step,delta(d[2][1].likelihood,d21_likelihood))
            iter += 1
        end
        if length(d)==1
            e=1
            if !tie[e]
                d[e][1].prior = posterior_win(t[e]) - posterior_lose(u[e+1])
                d[e][1].likelihood = trunc(d[e][1].prior,0.,false)/d[e][1].prior
                u[e+1].likelihood_lose =  posterior_win(t[e]) - d[e][1].likelihood
                t[e+1].likelihood_draw = likelihood(u[e+1]) - u[e+1].prior
                t[e].likelihood_win = posterior_lose(u[e+1]) + d[e][1].likelihood
            else
                while (step > 1e-6) & (iter < 10)
                    d11_likelihood = d[e][1].likelihood
                    
                    u[e].prior_team = posterior_draw(t[e]) + u[e].prior
                    
                    d[e][1].prior = posterior_win(u[e]) - posterior_lose(t[e+1])
                    d[e][1].likelihood = trunc(d[e][1].prior,0.,false)/d[e][1].prior
                    u[e].likelihood_win = posterior_lose(t[e+1]) + d[e][1].likelihood
                    t[e+1].likelihood_lose = posterior_win(u[e]) - d[e][1].likelihood
                    
                    d[e][2].prior = posterior_win(u[e+1]) - posterior_lose(t[e])
                    d[e][2].likelihood = trunc(d[e][2].prior,0.,false)/d[e][2].prior
                    u[e+1].likelihood_win = posterior_lose(t[e]) + d[e][2].likelihood
                    t[e].likelihood_lose = posterior_win(u[e+1]) - d[e][2].likelihood
                    
                    t[e+1].likelihood_draw = likelihood(u[e+1]) - u[e+1].prior
                    t[e].likelihood_draw = likelihood(u[e]) - u[e].prior
                    
                    step = delta(d[1][1].likelihood_win,d11_likelihood)
                end
            end
        end
        return [ likelihood(t[o[e]]) for e in 1:length(t)]
    end
