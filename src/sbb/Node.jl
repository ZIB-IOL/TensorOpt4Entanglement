# Node
mutable struct Node
    nodeid::Int
    parentid::Int
    childs::Vector{Int}
    sibling::Int
    depth::Int
    isleave::Bool
    ZBs
    localdualbd::Float64
    pruned::Bool
    fixvars::Dict{Tuple{Int64, Symbol, Int64, Int64}, Float64}
    sol
    heursol

    function Node(nodeid::Int, parentid::Int, sibling::Int, depth::Int, isleave::Bool, ZBs, fixvars)
        new(nodeid, parentid, [], sibling, depth, isleave, ZBs, Inf, false, fixvars)
    end

end

function nodeHasChilds(node::Node)
    @assert length(node.childs) != 1
    return length(node.childs) == 2
end

"Intersect one entry's interval with new bounds and its Hermitian counterpart."
function tightenEntryBounds!(node::Node, Zind, part, i, j, lower, upper)
    B = node.ZBs[Zind]
    L,U = B[part,:L],B[part,:U]
    if part === :RE
        lower = max(lower,L[i,j],L[j,i])
        upper = min(upper,U[i,j],U[j,i])
        L[i,j] = L[j,i] = lower
        U[i,j] = U[j,i] = upper
    else
        lower = max(lower,L[i,j],-U[j,i])
        upper = min(upper,U[i,j],-L[j,i])
        L[i,j],U[i,j] = lower,upper
        L[j,i],U[j,i] = -upper,-lower
    end
    return lower,upper
end

# create root node
function createRootNode(dims, BST, Zdims, tol, globalZBs, globalfixvars)
    # PSD and trace one imply |Z[i,j]|² <= Z[i,i]Z[j,j] <= 1/4 for i != j.
    # These bounds also tighten the McCormick envelopes without an OBBT solve.
    fixvars = deepcopy(globalfixvars)
    ZBs = []
    if !isempty(globalZBs)
        ZBs = deepcopy(globalZBs)
    else
        function createBoundSys(sys, leftsys, rightsys, retleft, retright)
            Zind = sys.Zind
            dim = Zdims[Zind]
            radius = 0.5 + tol / 2
            ZB = Dict(
                (:RE,:L) => fill(-radius, (dim, dim)) + Diagonal(fill(0.5,dim)),
                (:RE,:U) => fill(radius, (dim, dim)) + Diagonal(fill(0.5,dim)),
                (:IM,:L) => fill(-radius, (dim, dim)) + Diagonal(fill(radius,dim)),
                (:IM,:U) => fill(radius, (dim, dim)) - Diagonal(fill(radius,dim)))
            push!(ZBs, ZB)
            # Imaginary part is anti-symmetric fix the vars
            for tj in 1:dim
                fixvars[(Zind, :IM, tj, tj)] = 0.0
            end
            return nothing
        end

        traverseDPSBST(1, BST, createBoundSys)
    end

    return Node(1, -1, -1, 0, true, ZBs, fixvars)
end
