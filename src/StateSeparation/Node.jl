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
    cuts
    sol
    heursol

    function Node(nodeid::Int, parentid::Int, sibling::Int, depth::Int, isleave::Bool, ZBs, fixvars)
        new(nodeid, parentid, [], sibling, depth, isleave, ZBs, 0.0, false, fixvars, [])
    end

end

function nodeHasChilds(node::Node)
    @assert length(node.childs) != 1
    return length(node.childs) == 2
end

# create root node
function creatRootNode(dims, BST, Zdims, tol, globalZBs, globalfixvars)
    # diagonal real [0,1], imag 0. nondiagonal [-1,1]
    fixvars = deepcopy(globalfixvars)
    ZBs = []
    if !isempty(globalZBs)
        ZBs = deepcopy(globalZBs)
    else
        function createBoundSys(sys, leftsys, rightsys, retleft, retright)
            Zind = sys.Zind
            dim = Zdims[Zind]
            ZB = Dict(
                (:RE,:L) => fill(-1.0 - tol / 2 , (dim, dim)) + Diagonal(ones(dim)),
                (:RE,:U) => fill(1.0 + tol / 2, (dim, dim)),
                (:IM,:L) => fill(-1.0 - tol / 2, (dim, dim)) + Diagonal(ones(dim)),
                (:IM,:U) => fill(1.0 +  tol / 2, (dim, dim)) - Diagonal(ones(dim)))
            push!(ZBs, ZB)
            # Imaginary part is anti-symmetric fix the vars
            for tj in 1:dim
                fixvars[(Zind, IM, tj, tj)] = 0.0
            end
            return nothing
        end

        traverseDPSBST(1, BST, createBoundSys)
    end

    return Node(1, -1, -1, 0, true, ZBs, fixvars)
end


