using Test, JuMP, MosekTools, LinearAlgebra, Random
import Mosek
const E = ExactEntanglement

# The solver-backed algorithms need a Mosek licence. Skip cleanly without one
# so the rest of the suite still runs in CI.
function mosek_available()
    try
        m = Model(Mosek.Optimizer); set_silent(m)
        @variable(m, x >= 1.0); @objective(m, Min, x)
        optimize!(m)
        return termination_status(m) == OPTIMAL
    catch
        return false
    end
end

if !mosek_available()
    @info "Mosek licence not available - skipping integration tests. " *
          "Set MOSEKLM_LICENSE_FILE to enable them."
else
    @testset "integration (Mosek)" begin
        outdir = mktempdir()
        ENV["EXACTENT_RESULTS_DIR"] = outdir
        try
            args(algo) = Dict{String,Any}(
                "state" => "state_133.jl", "algo" => algo, "time-limit" => 60.0,
                "log-level" => 0, "maxnnodes" => 100, "minnnodes" => 1, "maxrounds" => 100,
                "heur-ladmm1-maxiter" => 16, "heur-ladmm-maxiter" => 8,
                "heur-manopt-maxiter" => 150, "loop" => -3)

            readbound(file, key) = parse(Float64,
                match(Regex("$key: (\\S+)"), read(joinpath(outdir, file), String)).captures[1])

            @testset "DDPS+ gives a valid lower bound" begin
                @test runEntangle(args("DDPS+")) == 0
                @test 0 < readbound("state_133.jl_DDPS+", "lb_relx") < 1
            end

            @testset "sBB honours even node budgets and preserves the oracle bound" begin
                # A generic witness leaves a root gap. The symmetric W-state
                # projector can instead close the gap at the root, making it
                # unsuitable for checking how many child nodes are solved.
                rng = MersenneTwister(2468)
                A = randn(rng,ComplexF64,16,16)
                H = Matrix(Hermitian(A+A')); H /= norm(H)
                bounds = Float64[]
                for limit in (1,2,3)
                    problem = E.Problem(real(H),imag(H),[2,2,2,2])
                    p = Param(time_limit=120.0,maxnnodes=limit,maxeffortnnodes=limit,log_level=1)
                    result,log = mktemp() do path,io
                        result = redirect_stdout(io) do
                            E.separate!(problem,p,2)
                        end
                        flush(io)
                        return result,read(path,String)
                    end
                    primal,dual = result[1:2]
                    @test occursin("node solve limit reached: $limit",log)
                    @test isfinite(dual) && dual >= primal - 1e-6
                    push!(bounds,dual)
                end
                @test all(diff(bounds) .<= 1e-5)
            end

            @testset "Alt-SDP gives a valid upper bound" begin
                @test runEntangle(args("Alt-SDP")) == 0
                @test 0 < readbound("state_133.jl_Alt-SDP", "ub_relx") < 1
            end

            @testset "GHZ bounds respect the analytic threshold" begin
                # the m-party GHZ white-noise threshold is known in closed form
                ghz(m) = 1 - 1 / (1 + 2.0^(m - 1))
                @test ghz(3) ≈ 0.8
                ghzargs = merge(args("DDPS+"), Dict{String,Any}("state" => "state_033.jl"))
                @test runEntangle(ghzargs) == 0
                lb = readbound("state_033.jl_DDPS+", "lb_relx")
                # DDPS+ is a relaxation, so its lower bound cannot exceed the truth
                @test lb <= ghz(3) + 1e-6
                @test lb > 0.5                      # and it is not vacuous
            end

            @testset "memory is attributed to each level" begin
                controls = merge(args("CP"),Dict{String,Any}("cp-certify-every"=>1,"maxrounds"=>2))
                @test runEntangle(controls) == 0
                txt = read(joinpath(outdir, "state_133.jl_CP"), String)
                num(k) = parse(Float64, match(Regex("$k: (\\S+)"), txt).captures[1])
                # CP calls the oracle, so :cp must contain :lmo, and :total both
                @test num("mem_lmo_calls") > 0
                @test num("mem_lmo_alloc_gib") <= num("mem_cp_alloc_gib") + 1e-9
                @test num("mem_cp_alloc_gib") <= num("mem_total_alloc_gib") + 1e-9
                @test num("mem_total_peak_rss_mib") > 0
                @test num("mem_lmo_model_nnz") > 0
            end

            @testset "a legacy code writes the canonical filename" begin
                legacy = merge(args("DDPS+"), Dict{String,Any}("algo" => "RLT"))
                @test runEntangle(legacy) == 0
                @test isfile(joinpath(outdir, "state_133.jl_DDPS+"))
                @test !isfile(joinpath(outdir, "state_133.jl_RLT"))
            end

            @testset "the bounds bracket each other" begin
                # DDPS+ lower bound must not exceed the Alt-SDP upper bound
                @test readbound("state_133.jl_DDPS+", "lb_relx") <=
                      readbound("state_133.jl_Alt-SDP", "ub_relx") + 1e-6
            end

            @testset "IR enters its CP tail without an empty LADMM call" begin
                p = Param(time_limit=60.0,tratio=0.15,log_level=0)
                boundary = (1-p.tratio)*p.time_limit
                @test E.irShouldLift(p,false;elapsed=prevfloat(boundary))
                @test !E.irShouldLift(p,false;elapsed=boundary)
                @test !E.irShouldLift(p,false;elapsed=boundary+1)
                @test E.irShouldLift(p,true;elapsed=boundary)
                p.is_last = true
                @test E.irShouldLift(p,false;elapsed=boundary)

                # Reserve the whole budget for CP to force the driver branch
                # without relying on machine speed or a nearly expired clock.
                # The computational basis already represents this target.
                H = Matrix{Float64}(Diagonal([0.4,0.3,0.2,0.1]))
                p = Param(time_limit=60.0,tratio=1.0,loop=1,log_level=0,
                    pointsize_bound=4,rank_bound=1,maxnnodes=1,maxeffortnnodes=1)
                E.resetPhases!()
                p.start_time = time()
                ub,lb,approxub,residual,weights_sum = E.solveIR(H,zeros(4,4),[2,2],p)
                @test E.phaseStat(:ladmm).calls == 0
                @test E.phaseStat(:cp).calls == 1
                @test p.is_last
                @test ub == lb == 0.0
                @test approxub == 1.0
                @test residual == 0.0
                @test weights_sum ≈ 1
            end

            @testset "IR warm starts across two passes on four and five qubits" begin
                for state in ("state_0.jl", "state_10.jl")
                    data = E.loadBenchmark(state)
                    H = Matrix(data.ρ); dims = collect(data.dims)
                    p = Param(time_limit=3600.0, loop=2, log_level=0,
                        heur_LADMM_penalty_update=:balance, heur_LADMM_conjugates=true,
                        cp_real_master=true, ir_refit_scalar=true, cp_certify_every=2,
                        cp_rounds_per_ir=1)
                    E.applySizePreset!(p, length(dims), "IR")
                    p.cp_rounds_per_ir = 1
                    p.cp_certify_every = 2
                    p.heur_LADMM_maxiter = p.heur_LADMM1_maxiter = 2
                    p.heur_MANOPT_maxiter = p.heur_MANOPT1_maxiter = 3
                    p.maxrounds = -1
                    p.maxnnodes = 1
                    Random.seed!(p.seed)
                    E.resetPhases!()
                    p.start_time = time()
                    ub, lb, _, residual, weights_sum = E.ALGORITHMS["IR"](real(H), imag(H), dims, p)
                    @test E.phaseStat(:ladmm).calls == 2
                    @test E.phaseStat(:cp).calls == 2
                    @test p.maxrounds == -1
                    @test 1 - 1 / (1 + 2.0^(length(dims)-1)) - 1e-6 <= ub <= 1 + 1e-6
                    @test lb <= ub + 1e-6
                    @test lb <= 1 - 1 / (1 + 2.0^(length(dims)-1)) + 1e-5
                    @test isfinite(residual) && residual >= 0
                    @test weights_sum ≈ 1 atol=1e-5
                end
            end

            @testset "Real master matches conjugate columns and reconstructs factors" begin
                for m in (3,4)
                    dims=fill(2,m); D=prod(dims)
                    data=E.loadBenchmark(m==3 ? "state_033.jl" : "state_0.jl")
                    H=Matrix(data.ρ)
                    p=Param(time_limit=3600.0,pointsize_bound=2D^2,rank_bound=2D^2,log_level=0)
                    Random.seed!(75)
                    detector,_,_,_=E.initialActiveSet(real(H),imag(H),dims,p)
                    E.addConjugateStates!(detector)
                    E.initialLPRelaxation(detector,p)
                    _,_,_,u=E.solveMSK(detector.model,p)
                    full=E.masterSnapshot(detector,u,p)
                    @test !isnothing(full)
                    variables=num_variables(detector.model)
                    p.cp_real_master=true
                    E.initialLPRelaxation(detector,p)
                    _,_,_,u=E.solveMSK(detector.model,p)
                    real_solution=E.masterSnapshot(detector,u,p)
                    @test !isnothing(real_solution)
                    @test abs(tr(real_solution.witness[:RE])) < 1e-6
                    @test real_solution.upper≈full.upper atol=1e-7
                    @test num_variables(detector.model)<variables
                    @test length(detector.cuts)<length(detector.purestates)
                    Y=sum(real_solution.weights[a]*real_solution.purestates[a] for a in eachindex(real_solution.weights))
                    @test norm(Y-((1-u)*H+u*Matrix{ComplexF64}(I,D,D)/D))<1e-6
                    @test all(real_solution.purestates[a]≈foldl(kron,[v*v' for v in real_solution.substates[a]]) for a in eachindex(real_solution.weights))
                    lpobjective = objective_function(detector.model)
                    nconstraints = num_constraints(detector.model; count_variable_in_set_constraints=true)
                    stable = E.stableMasterWitness(detector, real_solution, p)
                    @test stable.upper == real_solution.upper
                    @test stable.weights == real_solution.weights
                    @test stable.purestates === real_solution.purestates
                    @test stable.residual == real_solution.residual
                    @test stable.objective >= real_solution.objective - 100p.master_obj_tol - 1e-6
                    W = stable.witness[:RE] + im*stable.witness[:IM]
                    @test real(dot(W,H-Matrix{ComplexF64}(I,D,D)/D)) ≈ 1 atol=1e-6
                    @test all(real(dot(W,P)) <= stable.offset+1e-6 for P in stable.purestates)
                    @test norm(W) <= norm(real_solution.witness[:RE]+im*real_solution.witness[:IM])+1e-6
                    @test E.JuMP.isequal_canonical(objective_function(detector.model), lpobjective)
                    @test num_constraints(detector.model; count_variable_in_set_constraints=true) == nconstraints
                    oldlength=length(real_solution.purestates)
                    E.addBatchStates(detector,[detector.purestates[end]],[detector.substates[end]])
                    @test length(real_solution.purestates)==oldlength
                end
            end

            @testset "CP selects a sparse validated solver result" begin
                psi = ComplexF64[1,0,0,1]/sqrt(2); H = psi*psi'
                p = Param(time_limit=60.0,log_level=0,cp_real_master=true,
                    pointsize_bound=64,rank_bound=64)
                Random.seed!(81)
                detector,_,_,_ = E.initialActiveSet(real(H),imag(H),[2,2],p)
                E.initialLPRelaxation(detector,p)
                _,_,_,u = E.solveMSK(detector.model,p)
                snapshots = [E.masterSnapshot(detector,u,p;result=r)
                    for r in 1:result_count(detector.model)]
                valid = filter(!isnothing,snapshots)
                @test !isempty(valid)
                best_upper = minimum(snapshot.upper for snapshot in valid)
                eligible = filter(snapshot -> snapshot.upper <= best_upper+p.master_obj_tol,valid)
                selected = E.masterSnapshot(detector,u,p)
                @test !isnothing(selected)
                @test selected.upper <= best_upper+p.master_obj_tol
                @test length(selected.weights) == minimum(length(snapshot.weights) for snapshot in eligible)
                @test all(>(0),selected.weights)
                @test sum(selected.weights) ≈ 1
                Y = sum(w*P for (w,P) in zip(selected.weights,selected.purestates))
                @test Y ≈ (1-selected.upper)*H+selected.upper*Matrix{ComplexF64}(I,4,4)/4 atol=1e-7
                @test E.masterSnapshot(detector,u,p;result=result_count(detector.model)+1) === nothing
            end

            @testset "IR fallback does not freeze a valid new CP warm start" begin
                v = ComplexF64[1,0,0,1]/sqrt(2); H = v*v'
                basis = [ComplexF64[1,0],ComplexF64[0,1]]
                S = [[a,b] for a in basis for b in basis]
                P = [kron(s[1]*s[1]',s[2]*s[2]') for s in S]
                detector = E.ThresholdEntanglementDetector(real(H),imag(H),[2,2],P,S)
                plus = ComplexF64[1,1]/sqrt(2); minus = ComplexF64[1,-1]/sqrt(2)
                yp = ComplexF64[1,im]/sqrt(2); ym = conj.(yp)
                oldS = [[basis[1],basis[1]],[basis[2],basis[2]],
                    [plus,plus],[minus,minus],[yp,ym],[ym,yp]]
                oldP = [kron(s[1]*s[1]',s[2]*s[2]') for s in oldS]
                old = E.MasterSnapshot(2/3,2/3,0.0,oldP,oldS,fill(1/6,6),
                    Dict(:RE=>zeros(4,4),:IM=>zeros(4,4)),1.0,0.0)
                @test sum(old.weights[a]*oldP[a] for a in eachindex(oldP)) ≈
                    H/3+Matrix{ComplexF64}(I,4,4)/6
                saved = Ref{Union{Nothing,E.MasterSnapshot}}(old)
                p = Param(time_limit=60.0,log_level=0,cp_real_master=true)
                ub,_,_,_,_,_,_,_ = E.cuttingPlane(detector,
                    E.Problem(real(H),imag(H),[2,2]),p,0,true;snapshot_state=saved)
                @test ub ≈ 1 atol=1e-6
                @test saved[].upper ≈ 1 atol=1e-6
                @test old.upper == 2/3
            end

            @testset "CP respects the physical threshold for separable targets" begin
                dims = [2,2]; mixed = Matrix{Float64}(I,4,4)/4
                p = Param(time_limit=60.0,log_level=0,pointsize_bound=4,rank_bound=4)
                @test E.solveCP(mixed,zeros(4,4),dims,p) == (0.0,0.0,0.0,0.0)
                @test E.solveIR(mixed,zeros(4,4),dims,p) == (0.0,0.0,0.0,0.0,1.0)

                H = Matrix{ComplexF64}(Diagonal([0.4,0.3,0.2,0.1]))
                detector,_,_,_ = E.initialActiveSet(real(H),imag(H),dims,p)
                saved = Ref{Union{Nothing,E.MasterSnapshot}}(nothing)
                result = E.cuttingPlane(detector,E.Problem(real(H),imag(H),dims),p;
                    snapshot_state=saved)
                @test result[1] == result[2] == 0.0
                snapshot = saved[]
                @test snapshot.objective < 0
                @test all(>=(0),snapshot.weights)
                @test sum(snapshot.weights) ≈ 1
                Y = sum(w*P for (w,P) in zip(snapshot.weights,snapshot.purestates))
                @test Y ≈ H atol=1e-8
            end

            @testset "A small point budget retains the full mixed-state basis" begin
                psi = ComplexF64[1,0,0,1]/sqrt(2); H = psi*psi'
                p = Param(time_limit=60.0,log_level=0,pointsize_bound=2,rank_bound=1)
                detector,w,_,_ = E.initialActiveSet(real(H),imag(H),[2,2],p)
                @test length(detector.purestates) == length(detector.persistentInds) == 4
                @test sum(w[i]*detector.purestates[i] for i in eachindex(w)) ≈
                    Matrix{ComplexF64}(I,4,4)/4
                result = E.cuttingPlane(detector,E.Problem(real(H),imag(H),[2,2]),p,0,true)
                @test result[1] ≈ 1 atol=1e-6
            end

            @testset "Tiny positive CP support can be essential" begin
                # The four equatorial columns provide the only off-diagonal
                # entries. Discarding their tiny weights leaves a diagonal
                # polytope whose intersection with this target ray is I/4.
                delta = 1e-8
                a,b = inv(sqrt(1+delta^2)),delta/sqrt(1+delta^2)
                psi = ComplexF64[a,0,0,b]; H = psi*psi'
                t = 4a*b/(1+4a*b)
                plus = ComplexF64[1,1]/sqrt(2); minus = ComplexF64[1,-1]/sqrt(2)
                yp = ComplexF64[1,im]/sqrt(2); ym = conj.(yp)
                zero = ComplexF64[1,0]; one = ComplexF64[0,1]
                S = [[zero,zero],[one,one],[plus,plus],[minus,minus],[yp,ym],[ym,yp]]
                P = [foldl(kron,[v*v' for v in s]) for s in S]
                weights = [(1-t)*a^2,(1-t)*b^2,t/4,t/4,t/4,t/4]
                complete,_,w = E.activeFactors(P,S,weights;tol=0.0)
                @test length(complete) == 6
                @test sum(w[i]*complete[i] for i in eachindex(w)) ≈
                    (1-t)*H+t*Matrix{ComplexF64}(I,4,4)/4 atol=1e-15
                filtered,_,_ = E.activeFactors(P,S,weights)
                @test all(isdiag,filtered)
                @test H[1,4] != 0
            end

            @testset "IR keeps the final priced column of a capped CP pass" begin
                psi = ComplexF64[1,0,0,1]/sqrt(2); H = psi*psi'
                p = Param(time_limit=60.0,log_level=0,pointsize_bound=16,rank_bound=16,
                    cp_real_master=true,maxrounds=1)
                Random.seed!(73)
                detector,_,_,_ = E.initialActiveSet(real(H),imag(H),[2,2],p)
                pool = (pure=Any[],sub=Any[])
                first = E.cuttingPlane(detector,E.Problem(real(H),imag(H),[2,2]),p;
                    pricing_pool=pool)
                @test length(pool.pure) == length(pool.sub) == 1
                @test pool.pure[1] ≈ foldl(kron,[v*v' for v in pool.sub[1]])
                witness = first[7]
                offset = real(dot(witness[:RE]+im*witness[:IM],H))-first[1]
                @test dot(witness[:RE],real(pool.pure[1]))+
                    dot(witness[:IM],imag(pool.pure[1])) > offset
                E.addBatchStates(detector,pool.pure,pool.sub)
                second = E.cuttingPlane(detector,E.Problem(real(H),imag(H),[2,2]),p,0,true)
                @test second[1] <= first[1]+p.master_obj_tol
            end

            @testset "Final CP stabilization preserves gap closing" begin
                psi = ComplexF64[1,0,0,1]/sqrt(2); H = psi*psi'
                zero = ComplexF64[1,0]; one = ComplexF64[0,1]
                plus = ComplexF64[1,1]/sqrt(2); minus = ComplexF64[1,-1]/sqrt(2)
                yp = ComplexF64[1,im]/sqrt(2); ym = conj.(yp)
                S = [[zero,zero],[one,one],[plus,plus],[minus,minus],[yp,ym],[ym,yp]]
                P = [foldl(kron,[v*v' for v in s]) for s in S]
                detector = E.ThresholdEntanglementDetector(real(H),imag(H),[2,2],P,S)
                p = Param(time_limit=60.0,log_level=0,cp_real_master=true,
                    cp_certify_every=4,is_last=true,maxeffortnnodes=1,obj_tol=1e-7)
                E.initialLPRelaxation(detector,p)
                _,_,_,u = E.solveMSK(detector.model,p)
                raw = E.masterSnapshot(detector,u,p)
                result = E.cuttingPlane(detector,E.Problem(real(H),imag(H),[2,2]),p)
                @test result[1] ≈ 2/3 atol=1e-6
                @test result[2] > 0
                @test result[1]-result[2] <= p.master_obj_tol
                @test result[3]
                @test norm(result[7][:RE]) <= norm(raw.witness[:RE])+1e-5
            end

        finally
            delete!(ENV, "EXACTENT_RESULTS_DIR")
        end
    end
end
