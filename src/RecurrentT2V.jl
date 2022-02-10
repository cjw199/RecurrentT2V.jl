module RecurrentT2V

using Flux
using Flux: OneHotArray
using Flux: @functor
using Flux: @adjoint
using Flux: glorot_uniform
using Flux: zeros32
using Flux: gate 
using Flux: sigmoid_fast
using Flux: tanh_fast
using Zygote

include("layer.jl")
include("utils.jl")

export LSTMt2v, predict

end # module
