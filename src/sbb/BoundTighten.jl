"""
    tightenOBBTEntryBounds!(node, Zind, part, i, j, lower, upper, feas_tol)

Intersect OBBT bounds without replacing a positive-width interval by its
midpoint. Reject a small inconsistent intersection as numerical uncertainty;
return `false` only when the contradiction exceeds the feasibility tolerance.
"""
function tightenOBBTEntryBounds!(node::Node, Zind, part, i, j, lower, upper, feas_tol)
    B = node.ZBs[Zind]
    if part === :RE
        lower = max(lower, B[part, :L][i, j], B[part, :L][j, i])
        upper = min(upper, B[part, :U][i, j], B[part, :U][j, i])
    else
        lower = max(lower, B[part, :L][i, j], -B[part, :U][j, i])
        upper = min(upper, B[part, :U][i, j], -B[part, :L][j, i])
    end
    tol = feas_tol * max(1.0, abs(lower), abs(upper))
    lower > upper && return lower - upper < tol
    tightenEntryBounds!(node, Zind, part, i, j, lower, upper)
    # A finite interval contains feasible points away from its midpoint, even
    # if its width is below the numerical feasibility tolerance.
    lower == upper && (node.fixvars[(Zind, part, i, j)] = lower)
    return true
end

function BoundTighten(stateseparator::StateSeparator, focusnode::Node, globalobbt)
    BST = stateseparator.problem.BST
    Zdims = stateseparator.problem.Zdims
    primalbd = stateseparator.primalbd
    isfeasible = true
    function tighten(sys, leftsys, rightsys, retleft, retright)
        if sys.parent != -1
            Zind = sys.Zind
            dim = Zdims[Zind]
            if sys.left != -1 && sys.right !=-1
                return
            end
            # tighten each entry
            for tj in 1:dim
                for tk in 1:dim
                    for part in (:RE, :IM)
                        if !isfeasible
                            return
                        end
                        if !haskey( focusnode.fixvars, (Zind, part, tj, tk) )
                            for direction in (:L, :U)
                                remainingTime(stateseparator.param) <= 0 && return
                                optmodel = initRelaxationBound(stateseparator, focusnode, primalbd, Zind, part, tj, tk, direction, globalobbt)
                                status, solverstatus, sol = solveModel(optmodel, BST, stateseparator.param, true)
                                if !isnothing(status) &&  (status == RelaxFeasible || status == RelaxOptimal)
                                    lower = focusnode.ZBs[Zind][part,:L][tj,tk]
                                    upper = focusnode.ZBs[Zind][part,:U][tj,tk]
                                    direction == :L ? (lower = sol.dualobj - stateseparator.param.tol) :
                                        (upper = -sol.dualobj + stateseparator.param.tol)
                                    isfeasible = tightenOBBTEntryBounds!(focusnode, Zind, part, tj, tk,
                                        lower, upper, stateseparator.param.feas_tol)
                                    isfeasible || return
                                end
                            end
                        end
                    end
                end
            end
            if stateseparator.param.log_level > 1
                print("finished bound tightenning for Z$(Zind)\n")
            end
        end
    end
    for i in 1:stateseparator.param.max_obbt
        print("obbt round: $(i) \n")
        traverseDPSBST(1, BST, tighten)
    end
    if stateseparator.param.log_level > 1 && !isfeasible
        print("bound tightenning implies no better solution\n")
    end
    return isfeasible
end
