
# original problem
mutable struct Problem
   H
   Hout
   dims::Vector{Int64}
   dimH::Int
   nsubs::Int
   BST::Vector{TreeSys}
   Treeids::Vector{Int64}
   Zdims::Vector{Int64}
   Zleafids::Vector{Int64}
   Zrootid::Int64
   cutoffbound::Float64
   proximal
   globalZBs
   fixvars::Dict{Tuple{Int64, Symbol, Int64, Int64}, Float64}

   function Problem(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64})
      dimH = reduce(*, dims)
      nsubs = length(dims)
      H = Dict(:RE=>HR, :IM=>HI)
      Hout = Dict(:RE=>HR, :IM=>HI)
      treeidct = Ref(0)
      BST = []
      rootid = buildBST(1:nsubs, treeidct, BST, -1)
      @assert rootid == 1
      Treeids = []
      Zdims = []
      Zleafids = []
      Zrootid = 1
      function createAuxSys(sys, leftsys, rightsys, retleft, retright)
         dim = 0
         if sys.parent == -1
            @assert !isnothing(leftsys)
            @assert !isnothing(rightsys)
            @assert !isnothing(retleft)
            @assert !isnothing(retright)
            push!(Treeids, sys.treeid)
            sys.Zind = length(Treeids)
            @assert retleft * retright == dimH
            dim = retleft * retright
            push!(Zdims, dim)
            Zrootid = length(Zdims)
         elseif sys.sysid != -1
            push!(Treeids, sys.treeid)
            sys.Zind = length(Treeids)
            dim = dims[sys.sysid]
            push!(Zdims, dim)
            push!(Zleafids, length(Zdims))
         else
            @assert !isnothing(leftsys)
            @assert !isnothing(rightsys)
            @assert !isnothing(retleft)
            @assert !isnothing(retright)
            dim = retleft * retright
            push!(Treeids, sys.treeid)
            sys.Zind = length(Treeids)
            push!(Zdims, dim)
         end
         return dim
      end
      traverseDPSBST(1, BST, createAuxSys)
      cutoffbound = 0.0
      proximal = nothing
      globalZBs = []
      fixvars = Dict{Tuple{Int64, Symbol, Int64, Int64}, Float64}()
      new(H, Hout, dims, dimH, nsubs, BST, Treeids, Zdims,
         Zleafids, Zrootid, cutoffbound, proximal, globalZBs, fixvars)
   end
end

# Model
mutable struct OptModel
   model
   Xs
   Y
   Zs
   Zvars

   function OptModel(model, Xs, Y, Zs, Zvars)
      new(model, Xs, Y, Zs, Zvars)
   end
end
