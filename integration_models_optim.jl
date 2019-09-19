using Optim

# Parameter names are defined here, in LaTeX code and Unicode characters
param_names_latex = ["L", "R", "P", raw"\tau L", raw"\tau R"]; # For plots
param_names_unicode = ["L", "R", "P", "τL", "τR"];


"A regulizer for the features of the double integrator model,
additionaly penalizes powers different form 1 and different time constants"
function l1_regularizer(L::T, R::T, P::T, τL::T, τR::T) where {T}
    return abs(L) + abs(R) + abs(log(P)) + abs(τL) + abs(τR) + abs(τR-τL)
end

"A highly optimized way of calculating the model error in one loop pass

# Arguments
- `L::T`: input weight to the left integrator
- `R::T`: same for right
- `P::T`: input nonlinearity, equal for both integrators
- `τL::T`: time constant of the left leaky integrator
- `τR::T`: same for right
- `sL::Array{Uint8}`: left stimulus array, where coherence is in levels 0-10, where 10 is fully coherent stimulus
- `sR::Array{Uint8}`: same for right
- `dt`: timestep of the intgrator
- `decay_ca`: factor by which the calcium signal decays in one timepoint
- `coh_pows::Array{T}`:  a temporary array of length 11 that will hold precomupted stimulus strengths passed
- `exp_trace`: optional, if present will calculate error between stimulation and an experimental trace,
    otherwise the function returns the simulation
- `mask`: optional, if calculating the error with respect to the experimental trace,
    this is a binary mask that selects which timepoints to compare, for cross-validation
"
function independent_integrator_model(L::T, R::T, P::T, τL::T, τR::T, sL, sR,
                             dt::T, decay_ca::T, coh_pows,
                             exp_trace::OptT=nothing, mask::OptM=nothing) where {T, OptT, OptM}
    coh_pows[2:end] .=  (0.1:0.1:1) .^ P
    decay_L = exp(-dt/τL)
    decay_R = exp(-dt/τR)
    aL = zero(T)
    aR = zero(T)
    res = zero(T)
    if OptT === Nothing
        output = Array{T,1}(undef, length(sL))
        output[1] = zero(T)
    else
        err = zero(T)
    end
    @inbounds for i in 1:(length(sL)-1)
        inputL = L * coh_pows[sL[i]+1]
        inputR = R * coh_pows[sR[i]+1]
        aL = decay_L*(aL-inputL) + inputL
        aR = decay_R*(aR-inputR) + inputR
        a = aL+aR
        res = decay_ca*(res-a)+a
        if OptT === Nothing
            output[i+1] = res
        else
            if OptM == Nothing
                err += (res - exp_trace[i+1])^2
            else
                if mask[i+1]
                    err += (res - exp_trace[i+1])^2
                end
            end
        end
    end
    if OptT === Nothing
        return output
    else
        return err
    end
end


"Shift the trace so the baseline is at no stimulus"
function normalize_trace(trace, coh_L, coh_R)
    trace_nostim = Float32.(trace[(coh_L .== 0) .&
                              (coh_R .== 0)])
    return Float32.(trace).-median(trace_nostim)
end

"For a cell, optimize the paramters of the model, without cross validation"
function optimize_model(cell, stimuli, t::Timing, stim_map;
                        τCa = 1.77f0,
                        initial_params = [0.0f0, 0.0f0, 1.0f0, 0.01f0, 0.01f0],
                        lower_bounds   = [-20.0f0, -20.0f0, 0.0f0, 0.0f0, 0.0f0],
                        upper_bounds   = [20.0f0, 20.0f0, 10.0f0, 50.0f0, 50.0f0],
                        max_attempts = 100
                        )

    if length(cell.planes) < 2
        println(cell.original_id, " skipped")
        return nothing
    end
    println(cell.original_id)
    if any(isnan.(cell.trace))
        return nothing
    end

    initial_params[1:2] .= randn(Float32, 2).*2 # the function will restart if fitting
    # has failed, therefor it is good to start always with random initial params
    # Upsampling is not actually necessary for the fitting
    t2 = Timing(t.dt_imaging, t.dt_stim, 1, t.n_frames_trial);

    coh_L, coh_R = fill_stim(cell, stimuli, t2, stim_map)

    trace = normalize_trace(cell.trace, coh_L, coh_R)

    decay_ca = Float32.(exp(-t2.dt_sim/τCa))
    cohs = Float32.(collect(0:0.1:1));
    dt_sim = Float32.(t2.dt_sim)

    to_optim = params -> independent_integrator_model(params[1], params[2], params[3], params[4], params[5],
                                                coh_L, coh_R, dt_sim, decay_ca, cohs, trace)
    od  = OnceDifferentiable(to_optim, initial_params)

    res = nothing
    for i in 1:max_attempts
        try
            res = optimize(od,
                            initial_params,
                            lower_bounds,
                            upper_bounds,
                Fminbox{BFGS}())
            break
        catch
            initial_params[1:2] .= randn(Float32, 2).*2
        end
    end
    return res
end

function max_within_std_err(cv_errs)
    mean_errs = mean(cv_errs,1)[:]
    minval, minind = findmin(mean_errs)
    sterr_min = std(cv_errs[:, minind])/sqrt(size(cv_errs,1))
    i_reg = minind
    while i_reg < size(cv_errs,2) && mean_errs[i_reg+1]<minval+sterr_min
        i_reg +=1
    end
    return i_reg
end

struct RegularizedFitResult{T, N}
    params::NTuple{N, T}
    λ::T
    error_variance::T
end

"For a cell, optimize the paramters of the model, with cross validation"
function optimize_regularized(cell, stimuli, t::Timing, stim_map;
                              τCa = 1.77f0,
                              initial_params = [0.0f0, 0.0f0, 1.0f0, 0.01f0, 0.01f0],
                              lower_bounds   = [-20.0f0, -20.0f0, 0.0f0, 0.0f0, 0.0f0],
                              upper_bounds   = [20.0f0, 20.0f0, 10.0f0, 50.0f0, 50.0f0],
                              K=3,
                              max_iterations = 100,
                              λs = logspace(-4.0f0,2f0,9))

    if length(cell.planes) < 2
        println(cell.original_id, " skipped, too few planes")
        return nothing
    elseif any(isnan.(cell.trace))
        println(cell.original_id, " skipped, NaNs")
        return nothing
    end
    # Upsampling is not actually necessary for the fitting
    t2 = Timing(t.dt_imaging, t.dt_stim, 1, t.n_frames_trial);

    coh_L, coh_R = fill_stim(cell, stimuli, t2, stim_map)

    cell_trace = normalize_trace(cell.trace, coh_L, coh_R)

    decay_ca = Float32.(exp(-t2.dt_sim/τCa))
    cohs = Float32.(collect(0:0.1:1));
    dt_sim = Float32.(t2.dt_sim)

    n_samples = length(cell_trace)
    cv_errs = zeros(Float32, (K, length(λs)))
    cv_order = randperm(n_samples)
    n_items_segment = n_samples ÷ K

    parameters = Array{Float32}((length(initial_params),
                                K,
                                 length(λs)))

    # create test masks for cross-validation
    masks_test = zeros(Bool, (n_samples, K))

    for k in 1:K
        masks_test[cv_order[(k-1)*n_items_segment+1:(k)*n_items_segment], k] = true
    end
    masks_train = .~masks_test

    for (iλ, λ) in enumerate(λs)
        for k in 1:K
            to_optim(params) = (
                independent_integrator_model(params[1], params[2], params[3], params[4], params[5],
                                             coh_L, coh_R, dt_sim, decay_ca, cohs, cell_trace,
                                             masks_train[:, k]) + λ*l1_regularizer(params...))
            od  = OnceDifferentiable(to_optim, initial_params)
            res = nothing
            for i_iter in 1:max_iterations
                try
                    res = optimize(od, initial_params, lower_bounds,
                               upper_bounds, Fminbox{BFGS}())
                    break
                catch
                    initial_params[1:2] = randn(Float32, 2)*2
                    initial_params[3] = 1+randn(Float32)*0.2
                end
            end
            if res == nothing
                println("sadly, cell $(cell.original_id) optimization failed")
                return nothing
            end
            cv_errs[k, iλ] = independent_integrator_model(res.minimizer...,
                coh_L, coh_R, dt_sim, decay_ca, cohs, cell_trace, masks_test[:, k])
            parameters[:, k, iλ] .= res.minimizer
        end
    end

    i_reg = max_within_std_err(cv_errs)
    to_optim = params -> (independent_integrator_model(params[1], params[2], params[3], params[4], params[5],
                                            coh_L, coh_R, dt_sim, decay_ca, cohs, cell_trace) +
                                            λs[i_reg]*l1_regularizer(params...))
    od  = OnceDifferentiable(to_optim, initial_params)
    res = nothing

    for i_iter in 1:max_iterations
        try
            res = optimize(od, initial_params, lower_bounds,
                   upper_bounds, Fminbox{BFGS}())
            break
        catch
            initial_params[1:2] = randn(Float32, 2)*2
            initial_params[3] = 1+randn(Float32)*0.2
        end
    end

    println(cell.original_id, " optimized!")

    if res != nothing
        sim = independent_integrator_model(res.minimizer...,
                                            coh_L, coh_R, dt_sim, decay_ca, cohs)
        error_var = var(sim-cell_trace)/var(cell_trace)

        return RegularizedFitResult(tuple(res.minimizer...), λs[i_reg],
         error_var)
    else
        return res
    end
end

"Given a cell and model parameters, simulates the trace with the stimulus
sequence shown during recording the cell"
function simulate_trace(cell, stimuli, t, params, stim_map;
     τCa = 1.77f0, model=independent_integrator_model)
    t2 = Timing(t.dt_imaging, t.dt_stim, 1, t.n_frames_trial);

    coh_L, coh_R = fill_stim(cell, stimuli, t2, stim_map)

    trace = normalize_trace(cell.trace, coh_L, coh_R)

    decay_ca = Float32.(exp(-t2.dt_sim/τCa))
    cohs = Float32.(collect(0:0.1:1));
    dt_sim = Float32.(t2.dt_sim)

    return trace, model(params...,coh_L, coh_R, dt_sim, decay_ca, cohs)
end
