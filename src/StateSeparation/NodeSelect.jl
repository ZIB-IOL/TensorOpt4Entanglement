

# get the best leave node by dual bound
function stateseparatorBestNode(stateseparator::StateSeparator)
    bestnodeid = -1
    bestbd = -Inf
    for leave in stateseparator.leaves
        node = stateseparator.nodes[leave]
        @assert node.isleave
        if !node.pruned
            if node.localdualbd > bestbd
                bestnodeid = leave
                bestbd = node.localdualbd
            end
        end
    end
    if bestnodeid == -1
        return nothing
    else
        return stateseparator.nodes[bestnodeid]
    end
end

# get the best child nodes by lower bound
function stateseparatorBestChild(stateseparator::StateSeparator, node::Node)
    bestnodeid = -1
    bestbd = -Inf
    for child in node.childs
        childnode = stateseparator.nodes[child]
        if !childnode.pruned && childnode.isleave
            if childnode.localdualbd > bestbd
                bestnodeid = child
                bestbd = childnode.localdualbd
            end
        end
    end
    if bestnodeid == -1
        return nothing
    else
        return stateseparator.nodes[bestnodeid]
    end
end

# get the best sibling by lower bound
function stateseparatorBestSibling(stateseparator::StateSeparator, node::Node)
    if node.sibling == -1
        return nothing
    end
    siblnode = stateseparator.nodes[node.sibling]
    if siblnode.pruned || siblnode.isleave
        return nothing
    else
        return siblnode
    end
end


function stateseparatorSelectNode(stateseparator::StateSeparator)
    if stateseparator.nodes[stateseparator.selectnode].isleave
        return stateseparator.nodes[stateseparator.selectnode]
    end
    # Dynamically adjust depths
    maxplungequot = 0.25
    minplungedepth = 0
    maxplungedepth = 2

    #minplungedepth = div(stateseparator.maxdepth, 10)  # Integer division
    #maxplungedepth = div(stateseparator.maxdepth, 2)
    #maxplungedepth = max(maxplungedepth, minplungedepth)
    return stateseparatorBestNode(stateseparator)
    # Evaluate if plunging should continue
    if stateseparator.plungedepth >= maxplungedepth
       # we don't want to plunge again: select best node from the tree */
       stateseparator.plungedepth = 0
       #print("replunge\n")
       return stateseparatorBestNode(stateseparator)
    else
        maxbound = 2
        # Compute bounds for node evaluation
        if stateseparator.plungedepth > minplungedepth
            lowerbound = stateseparator.dualbd
            cutoffbound = stateseparator.primalbd
            maxbound = lowerbound + maxplungequot * (cutoffbound - lowerbound);
        end

        selectnode = stateseparator.nodes[stateseparator.selectnode];
        childnode = stateseparatorBestChild(stateseparator, selectnode)
        #print(childnode)
        if !isnothing(childnode) && childnode.localdualbd < maxbound
            #print("child\n")
            stateseparator.plungedepth += 1
            return childnode
        else
            siblnode = stateseparatorBestSibling(stateseparator, selectnode)
            #print("sible\n")
            if !isnothing(siblnode) && siblnode.localdualbd < maxbound
                stateseparator.plungedepth = 0
                return siblnode
            else
                #print("no sible\n")
                stateseparator.plungedepth = 0
                return stateseparatorBestNode(stateseparator)
            end
        end
    end

end