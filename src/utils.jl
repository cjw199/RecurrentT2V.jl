

# AD-friendly helper for dividing monolithic RNN params into equally sized gates
multigate(x::AbstractArray, h, ::Val{N}) where N = ntuple(n -> gate(x,h,n), N)

@adjoint function multigate(x::AbstractArray, h, c)
  function multigate_pullback(dy)
    dx = Zygote._zero(x, eltype(x))
    map(multigate(dx, h, c), dy) do dxᵢ, dyᵢ
      dyᵢ !== nothing && (dxᵢ.= Zygote.accum.(dxᵢ, dyᵢ));
    end
    return (dx, nothing, nothing)
  end
  return multigate(x, h, c), multigate_pullback
end

reshape_cell_output(h, x) = reshape(h, :, size(x)[2:end]...)

function predict(m, x)
  [m(xi) for xi in x]
end