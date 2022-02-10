using StatsBase

function make_time_data(x)
    [(Float32(i), x[i]) for i=1:length(x)]
end

gen1(t) = sin.(2*π*10 .* t)
gen2(t) = cos.(2*π*10 .* t)
gen3(t) = sin.(2*π*10 .* t) + cos.(2*π*30 .* t)

t = 0:.001:1
xs = Float32.(hcat(gen1(t), gen2(t), gen3(t)))
xs[200:400, :] .= xs[200:400, :] ./ 2
xs[600:800, :] .= xs[600:800, :] ./ 4

T = StatsBase.fit(ZScoreTransform, xs, dims=1)
StatsBase.transform!(T, xs)

data = Flux.unstack(xs, 1)
dat = [(Float32(i), data[i]) for i=1:length(data)]

inds = sample(1:1001, 300)
sort!(inds)
dat2 = dat[inds]

datanom = hcat(fill(0.2f0, 300, 3))
datanom2 = Flux.unstack(datanom,1)
datanom2 = make_time_data(datanom2)
