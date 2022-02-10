# following the paper https://arxiv.org/pdf/1907.05321.pdf

# ω = weights (freqiency), ϕ = bias (phase), ℱ = activation function (should be periodic)
struct T2V{V,F}
    ω::V
    ϕ::V
    ℱ::F
end

function T2V(k::Integer, ℱ = sin, init = glorot_uniform, initb = zeros32)
    T2V(init(k), initb(k), ℱ)
end

function(l::T2V)(t::T) where T
    τ = @. l.ω * t + l.ϕ
    vcat(τ[1], l.ℱ.(τ[2:end]))
end

@functor T2V

# LSTM+Time2Vec
struct LSTMt2vCell{A,T2,V,S}
    Wi::A
    Wh::A
    t2v::T2
    b::V
    state0::S
end

function LSTMt2vCell(in::Integer, out::Integer, k::Integer, ℱ = sin;
                init = glorot_uniform,
                initb = zeros32,
                init_state = zeros32)
    cell = LSTMt2vCell(init(out * 4, in + k), init(out * 4, out), T2V(k, ℱ, init, initb), initb(out * 4), (init_state(out,1), init_state(out,1)))
    cell.b[gate(out, 2)] .= 1
    return cell
end

# input datum is a tuple of scalar time and feature vector
function (m::LSTMt2vCell{A,T2,V,S})((h, c), data::Tuple{T, VecOrMat{T}}) where {A,V,T2,S,T}
    t, x = data
    τ = m.t2v(t)
    x′ = vcat(τ, x)
    b, o = m.b, size(h, 1)
    g = m.Wi*x′ .+ m.Wh*h .+ b
    input, forget, cell, output = multigate(g, o, Val(4))
    c′ = @. sigmoid_fast(forget) * c + sigmoid_fast(input) * tanh_fast(cell)
    h′ = @. sigmoid_fast(output) * tanh_fast(c′)
    return (h′, c′), reshape_cell_output(h′, x)
end

@functor LSTMt2vCell

Base.show(io::IO, l::LSTMt2vCell) =
    print(io, "LSTMt2v(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4,  ", ", size(l.ω, 1), ")")

LSTMt2v(a...; ka...) = Flux.Recur(LSTMt2vCell(a...; ka...))
Flux.Recur(m::LSTMt2vCell) = Flux.Recur(m, m.state0)