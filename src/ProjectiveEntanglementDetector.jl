

mutable struct ProjectiveEntanglementDetector <: AbstractEntanglementDetector
    H
    dims::Vector{Int64}
    dimH::Int
    nsubs::Int
    M
    b
    model
    purestates
    substates

    function ProjectiveEntanglementDetector(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, purestates)
        dimH = reduce(*, dims)
        nsubs = length(dims)
        H = Dict(:RE=>HR, :IM=>HI)
        M = Dict(:RE=>  zeros(AffExpr, 0, 0), :IM=>  zeros(AffExpr, 0, 0))
        entanglementdetector = new(H, dims, dimH, nsubs, M)
        entanglementdetector.purestates = copy(purestates)
        entanglementdetector.substates = []
        return entanglementdetector
    end
end

function normalizationCondition(detector::ProjectiveEntanglementDetector, param)
    if param.norm == 1
        l1normsize = 2 * detector.dimH ^ 2 + 1
        # normalization condition
        @constraint(detector.model, [1; vcat(vec(detector.M[:RE] ), vec(detector.M[:IM] )) ] in MOI.NormOneCone(l1normsize) )
    elseif param.norm == 2
        #l2normsize = 2 * detector.dimH ^ 2 + 1
        # normalization condition
        @constraint(detector.model, [1; vcat(vec(detector.M[:RE] ), vec(detector.M[:IM])) ] in SecondOrderCone() )
    elseif param.norm == -1
        linfnormsize = 2 * detector.dimH ^ 2 + 1
        @constraint(detector.model, [1; vcat(vec(detector.M[:RE] ), vec(detector.M[:IM]))]  in MOI.NormInfinityCone( linfnormsize))
    elseif param.norm == 3
        @constraint(detector.model, [1; vec([(detector.M[:RE]) (detector.M[:IM]); (-detector.M[:IM]) (detector.M[:RE])]) ] in MOI.NormSpectralCone(2*detector.dimH, 2 * detector.dimH))
    elseif param.norm == 4
        @constraint(detector.model, [1; vec([(detector.M[:RE]) (detector.M[:IM]); (-detector.M[:IM]) (detector.M[:RE])]) ] in MOI.NormNuclearCone(2*detector.dimH, 2 * detector.dimH))
    end
end


function earlyStopping(detector::ProjectiveEntanglementDetector, primalobj, param)
    isEarlyStopping = primalobj < param.master_obj_tol
    if isEarlyStopping
        println("early stopping: the state is not entangled: $(primalobj)\n")
    end
    return isEarlyStopping
end


function detectEntanglementProjective(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    purestates = getPureStates(HR, HI, dims, param)
    detector = ProjectiveEntanglementDetector(HR, HI, dims, purestates)
    separateproblem = Problem(HR, HI, dims)
    cuttingPlane(detector, separateproblem, param, 1)
end