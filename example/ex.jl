using Flux
using Flux.Losses: mse

include("../src/RecurrentT2V.jl")
using .RecurrentT2V
include("data.jl")

function t2v_loss(m, x, y)
    Flux.reset!(m)
    x_rec = RecurrentT2V.predict(m, x)
    sum(mse.(x_rec,y)) / length(x_rec)
end

function train_t2v(m, ps, opt, dat, n)
    x_dat, y = dat
    for i = 1:n
        train_loss, back = Flux.pullback( ()-> t2v_loss(m, x_dat, y), ps)
        grad = back(one(train_loss))
        Flux.update!(opt, ps, grad)
        @show i
        @show train_loss
    end
end

M = Chain(LSTMt2v(3, 32, 16), Dense(32, 16, swish), Dense(16,3))
ps = Flux.params(M)
opt = ADAM(1e-2)