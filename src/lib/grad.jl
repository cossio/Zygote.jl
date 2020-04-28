using MacroTools: @q

"""
    @nograd(f...)

The output of the argument functions will be considered as constant
and will not contribute gradients.
"""
macro nograd(ex)
  isexpr(ex, :tuple) || (ex = Expr(:tuple, ex))
  blk = @q begin end
  for f in ex.args
    back = MacroTools.@q _ -> ($__source__; nothing)
    push!(blk.args, :(@inline Zygote._pullback(::Context, ::Core.Typeof($(esc(f))), args...) = $(esc(f))(args...), $back))
  end
  return blk
end

"""
    nograd(f)

Calls `f()`, without accumulating gradients. For example:

```julia-repl
julia> using Zygote; using Zygote: nograd
julia> f(x) = (A = zero(x); nograd(_ -> (A .= x)); sum(A + x))
julia> x = randn(3); f(x) == sum(2x)
true
julia> f'(x)
[1.0, 1.0, 1.0]
```

In the example above, the gradient is an array of ones even though
`f(x)` is computing `sum(2x)`.
"""
nograd(f) = f()
@nograd nograd

macro which(ex)
  @capture(ex, f_(args__)) || error("Zygote.@which f(args...)")
  :(InteractiveUtils.@which adjoint(Context(), $(esc(f)), $(esc.(args)...)))
end
