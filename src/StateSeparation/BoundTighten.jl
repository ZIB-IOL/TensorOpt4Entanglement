

function BoundTighten(stateseparator::StateSeparator, focusnode::Node, globalobbt)
    BST = stateseparator.problem.BST
    Zdims = stateseparator.problem.Zdims
    primalbd = stateseparator.primalbd
    isfeasible = true
    #copynode = deepcopy(focusnode)
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
                                optmodel = initRelaxationBound(stateseparator, focusnode, primalbd, Zind, part, tj, tk, direction, globalobbt)
                                status, solverstatus, sol = solveModel(optmodel, BST, stateseparator.param, true)
                                if !isnothing(status) &&  (status == RelaxFeasible || status == RelaxOptimal)
                                    focusnode.ZBs[Zind][part, direction][tj, tk] = direction == :L ? sol.dualobj - stateseparator.param.tol : - sol.dualobj + stateseparator.param.tol
                                end
                                #print(sol.dualobj, "\n")
                            end
                            tol = stateseparator.param.feas_tol * maximum((1,0,abs( focusnode.ZBs[Zind][part, :U][tj, tk]), abs(focusnode.ZBs[Zind][part, :L][tj, tk]) ))
                            if abs( focusnode.ZBs[Zind][part, :U][tj, tk] - focusnode.ZBs[Zind][part, :L][tj, tk] ) < tol
                                print("fix\n")
                                focusnode.fixvars[(Zind, part, tj, tk)] = (focusnode.ZBs[Zind][part, :U][tj, tk] + focusnode.ZBs[Zind][part, :L][tj, tk]) / 2
                            elseif focusnode.ZBs[Zind][part, :U][tj, tk] - focusnode.ZBs[Zind][part, :L][tj, tk] <= -tol
                                isfeasible = false
                            end
                        end
                    end
                end
            end
            #print(focusnode.ZBs[Zind])
            if stateseparator.param.log_level > 1
                print("finished bound tightenning for Z$(Zind)\n")
            end
        end
    end
    for i in 1:stateseparator.param.max_obbt
        print("obbt round: $(i) \n")
        traverseDPSBST(1, BST, tighten)
        #copynode = deepcopy(focusnode)
    end
    if stateseparator.param.log_level > 1 && !isfeasible
        print("bound tightenning implies no better solution\n")
    end
    return isfeasible
end