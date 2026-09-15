# Define the sys structure
mutable struct TreeSys
    sysid::Int   # index in the dims array
    sysids::Array{Int}  # index in the dims array of sub nodes
    Zind::Int    # Z matrix index
    treeid::Int  # index in the TreeSys array
    left::Int
    right::Int
    parent::Int
    depth::Int
end

# Function to build the balanced binary search tree from an array
function buildBST(arr, treeidct, BST, parent=-1)
    if isempty(arr)
        return -1
    end
    # Find the middle element to ensure balanced tree
    if length(arr) == 1
        treeidct[] += 1
        treeid = treeidct[]
        @assert treeid > 0
        depth = parent == -1 ? 0 : BST[parent].depth + 1
        sys = TreeSys(arr[1], [arr[1]], -1, treeid, -1, -1, parent, depth)
        push!(BST, sys)
        return treeid
    else
        mid = div(length(arr) + 1, 2)
        treeidct[] += 1
        treeid = treeidct[]
        @assert treeid > 0
        # Recursively build the left and right subtrees
        depth = parent == -1 ? 0 : BST[parent].depth + 1
        sys = TreeSys(-1, [], -1, treeid, -1, -1, parent, depth)
        push!(BST, sys)
        BST[treeid].left = buildBST(arr[1:mid], treeidct, BST, treeid)
        @assert BST[treeid].treeid == treeid
        BST[treeid].right = buildBST(arr[mid+1:end], treeidct, BST, treeid)
        @assert BST[treeid].treeid == treeid
        @assert sys.left != - 1
        @assert sys.right != - 1
        sys.sysids = vcat(BST[sys.left].sysids, BST[sys.right].sysids)
        return treeid
    end
end

function traverseDPSBST(treeid, BST, funcsys, params...)
    sys = BST[treeid]
    leftsys = nothing
    rightsys = nothing
    retleft = nothing
    retright = nothing
    if sys.left != -1
        leftsys = BST[sys.left]
        retleft = traverseDPSBST(leftsys.treeid, BST, funcsys, params...)
    end
    if sys.right != -1
        rightsys = BST[sys.right]
        retright = traverseDPSBST(rightsys.treeid, BST, funcsys, params...)
    end
    return funcsys(sys, leftsys, rightsys, retleft, retright, params...)
end

