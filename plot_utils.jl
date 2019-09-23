using Colors
using Interpolations
using AxisArrays
using Formatting
using Statistics

include("data_preparation.jl")
include("integration_models_optim.jl")


median_filter(v, ws) =  [median(v[i:(i+ws-1)]) for i=1:(length(v)-ws+1)]

"Return the interval when active and the coherence for each stimulus"
function coherence_intervals(time, coh)
    intervals = Array{NTuple{2, Float64},1}()
    coherences = Array{Float64,1}()
    cstart = -1
    for i_t in 1:length(coh)-1
        if (coh[i_t+1] != 0.0 && coh[i_t]==0)
            cstart=i_t+1

        elseif (coh[i_t+1] == 0.0 && coh[i_t]!=0)
            push!(intervals,(time[cstart], time[i_t]))
            push!(coherences, coh[(cstart+i_t)÷2])
        end
    end
    return intervals, coherences
end

function coherence_intervals_fixed_transition(time, coh)
    intervals = Array{NTuple{2, Float64},1}()
    coherences = Array{Float64,1}()
    cstart = 1
    coh = median_filter(coh, 5)
    for i_t in 2:length(coh)-1
        if (coh[i_t+1] != coh[i_t])
            push!(intervals,(time[cstart], time[i_t]))
            push!(coherences, coh[(cstart+i_t)÷2])
            cstart=i_t
        end
    end
    return intervals, coherences
end

# We duplicate the outer colors to account for the 0.8 coherence
# in the fixed-transition experiments
coherence_colors = reverse(parse.(Colorant, [
"#B07382",
"#B07382",
"#C89EA8",
"#E1CAD0",
"#F5F5F5",
"#CBD6D4",
"#9EB0AE",
"#738C89",
"#738C89" ] ))
all_colors = [RGB(0.7,0.7,0.7); coherence_colors[[1, 4,5, 6, end]]]

# Mapping from coherences to colors
coherences = [-1.0,-0.8, -0.6, -0.3, 0, 0.3, 0.6, 0.8, 1.0]
color_dict = Dict(coh => col for (coh, col) in zip(coherences, coherence_colors))

# The color which is far from all the others
# (found using distinguishable_colors from Color.jl)
sim_color = RGB{Float64}(0,0.2,0.8)

rect(x1,x2,y1, y2) = Shape([x1,x2,x2,x1],[y1,y1, y2,y2])


"Plots the cell trace across all the planes with the model fit superimposed"
function plot_cell_trace(cell::Cell, stimdata, t::Timing, stim_map,
                         params, overlay_planes=false; max_n_planes=0,
                          offset = 0, limits=:nothing,model=independent_integrator_model,
                          τCa=1.77)

    t2 = Timing(t.dt_imaging, t.dt_stim, 1, t.n_frames_trial);

    coh_L, coh_R = fill_stim(cell, stimdata, t2, stim_map)

    trace_normalized = normalize_trace(cell.trace, coh_L, coh_R)

    decay_ca = Float32.(exp(-t2.dt_sim/τCa))
    cohs = Float32.(collect(0:0.1:1));
    dt_sim = Float32.(t2.dt_sim)

    n_frames_sim = overlay_planes ? t.n_frames_trial : length(trace_normalized)
    coh_L = coh_L[1:n_frames_sim]
    coh_R = coh_R[1:n_frames_sim]

    sim = params -> model(params[1], params[2], params[3], params[4], params[5],
                          coh_L, coh_R,
                          dt_sim, decay_ca, cohs)

    result_conv = sim(params)
    t_imaging = 0:t2.dt_imaging:(n_frames_sim-1)*t2.dt_imaging

    if limits == :nothing
        limits = extrema(trace_normalized)
    end
    limits = limits .+ offset

    # Plot the backgrounds indicating the coherence
    intvals_cohs = coherence_intervals_fixed_transition
    for (interval, coherence) in zip(intvals_cohs(t_imaging, (coh_L.-Float32.(coh_R))./10.0)...)
        plot!(rect(interval..., limits...),
              color=color_dict[coherence], linewidth=0, linecolor=nothing)
    end

    # Plot the cell trace
    if overlay_planes
        for i_plane in 1:length(cell.planes)
            plot!(t_imaging, trace_normalized[1+(i_plane-1)*t.n_frames_trial:i_plane*t.n_frames_trial]+offset,
             label="flourescence", color=RGBA(0.2,0.2,0.2,0.9))
        end
    else
        plot!(t_imaging, trace_normalized+offset, label="flourescence", color=RGB(0.2,0.2,0.2))
    end

    plot!(t_imaging, result_conv[1:length(t_imaging)]+offset,  label="model", color=sim_color,
        lw=2, tick_direction=:out, grid=:none, legend=:none, size=(800,300))

    if max_n_planes > 0
        xlims!((0, t2.n_frames_trial*max_n_planes*t2.dt_imaging))
    end
    if offset == 0
        ylims!(limits)
    end
end

### Combining traces for fixed transition experiments

"Get the times when there is a transition from one stimulus to the other"
function transition_times(stimdata)
    stim_plane = stimdata[1][:]
    dt = 0.01
    t_start = 10
    dt_stim = 20

    i_start = round(Int64, t_start/dt)
    di = round(Int64, dt_stim/dt)
    stim_order = round.(Int64,stim_plane[i_start:di:end]);

    transition_times = Dict()

    for (i_t, (s1, s2)) in enumerate(zip(stim_order[1:end-1], stim_order[2:end]))
        transition_times[(s1, s2)] = i_t*20
    end

    return transition_times
end

function split_trace_per_stim(trace, n_planes, transition_times, dt_plane)
    tr_responses = Dict{Tuple{Int8, Int8}, Array{AxisArray{Float64,1},1}}()
    plane_duration = (length(trace)/n_planes)*dt_plane
    itp =  interpolate(trace, BSpline(Linear()), OnGrid())
    for i_plane in 0:n_planes-1
        offset = plane_duration*i_plane
        for (key, time) in transition_times
            if key ∉ keys(tr_responses)
                tr_responses[key] = Array{AxisArray{Float32,1},1}()
            end
            push!(tr_responses[key], AxisArray([itp[(time+offset)/dt_plane+os] for os in -30:30],
                Axis{:time}((-30:30)*dt_plane)))
        end
    end
    return tr_responses
end

function plot_split_traces(split_trs; ylims=(-3,4), normalise=false)
    prev_next_plots = []
    for i_a in [1, 2, 4, 5]
        for rev in [true, false]
            cplot = plot()
            idxs_b = cat(1, 0:i_a-1, i_a+1:5)
            for i_b in idxs_b

                idx = rev ? (i_b, i_a) : (i_a, i_b)
                traces = cat(2, split_trs[idx]...)
                to_plot = mean(traces, 2)[:,1]
                mid_val = normalise ? to_plot[atvalue(0)] : 0
                plot!(cplot, axisvalues(to_plot), to_plot-mid_val,
                ribbon=std(traces, 2)[:,1], c=all_colors[i_b+1],
                legend=:none, yticks=(i_a==1 && rev) || (i_a==4 && rev) ? :auto : [],
                    xticks=i_a<4 ? [] : :auto, xlim=(-10,10), ylim=ylims,
                    foreground_color_axis=:transparent,
                    title=format("Coherence {} {}",
                        i_a >= 4 ? "right" : "left",
                        i_a % 2 == 0 ? 0.3 : 0.8),
                    titlefont=Plots.Font("sans-serif", 8, :hcenter, :vcenter, 0.0, RGB(0,0,0)))
                vline!(cplot, [0], c=RGB(0,0,0), lw=0.5)
            end
            push!(prev_next_plots, cplot)
        end
    end
    plot(prev_next_plots..., layout=(2,4), grid=:none)
end

function plot_split_traces_subset(split_trs; selected = [(4, false), (5, false)], ylims=(-3,1))
    prev_next_plots = []
    for (i_a, rev) in selected
        cplot = plot()
        idxs_b = cat(1, 0:i_a-1, i_a+1:5)
        for i_b in idxs_b
            idx = rev ? (i_b, i_a) : (i_a, i_b)
            traces = cat(2, split_trs[idx]...)
            to_plot = mean(traces, 2)[:,1]
            mid = to_plot[atvalue(0)]
            to_plot = to_plot[0..10]
            # just plot the second part, with the first one a signle line
            plot!([-5,0],[0,0], c=all_colors_r[i_a+1], lw=5)
            plot!(cplot, axisvalues(to_plot), to_plot-mid,
            ribbon=std(traces,2)[:,1], c=all_colors_r[i_b+1],
            legend=:none, yticks=(i_a==1 && rev) || (i_a==4 && rev) ? :auto : [],
                xticks=i_a<4 ? [] : :auto, xlim=(-5,10), ylim=ylims,
                foreground_color_axis=:transparent,
                title=format("Coherence {} {}",
                    i_a >= 4 ? "right" : "left",
                    i_a % 2 == 0 ? 0.3 : 0.8),
                titlefont=Plots.Font("sans-serif", 8, :hcenter, :vcenter, 0.0, RGB(0,0,0)))
            vline!(cplot, [0], c=RGB(0,0,0), lw=0.5)
        end
        push!(prev_next_plots, cplot)
    end
    plot(prev_next_plots..., grid=:none)
end

function plot_split_traces_subset_unnormed(split_trs; selected = [(4, false), (5, false)], ylims=(-3,1))
    prev_next_plots = []
    for (i_a, rev) in selected
        cplot = plot()
        idxs_b = cat(1, 0:i_a-1, i_a+1:5)
        for i_b in idxs_b
            idx = rev ? (i_b, i_a) : (i_a, i_b)
            traces = cat(2, split_trs[idx]...)
            to_plot = mean(traces, 2)[:,1]

            # just plot the second part, with the first one a signle line
            plot!(cplot, axisvalues(to_plot), to_plot,
            ribbon=std(traces,2)[:,1], c=all_colors_r[i_b+1],
            legend=:none, yticks=(i_a==1 && rev) || (i_a==4 && rev) ? :auto : [],
                xticks=i_a<4 ? [] : :auto, xlim=(-10,10), ylim=ylims,
                foreground_color_axis=:transparent,
                title=format("Coherence {} {}",
                    i_a >= 4 ? "right" : "left",
                    i_a % 2 == 0 ? 0.3 : 0.8),
                titlefont=Plots.Font("sans-serif", 8, :hcenter, :vcenter, 0.0, RGB(0,0,0)))
            vline!(cplot, [0], c=RGB(0,0,0), lw=0.5)
        end
        push!(prev_next_plots, cplot)
    end
    plot(prev_next_plots..., grid=:none)
end

function plot_trace_rolled(cell, transition_times, t)
    current_trace = cell.trace
    n_planes = length(cell.planes)
    split_trs = split_trace_per_stim(current_trace, n_planes, transition_times, t.dt_imaging);
    plot_split_traces(split_trs)
end

### Color mapping

function color_by_parameter(vals; hue_short=90, hue_long=270, threshold=10, luminance=50, chroma=60)
    return RGB.([LCHuv{Float32}(luminance, chroma, p>threshold ? hue_long : hue_short) for p in vals])
end


function continuous_colors_from_rois_old(rois)
    knots = ([0.0, 12.0, 35.0], [0.0,1.1,1.9])
    H = Float32.([0 120 240
         0 120 240
         0 120 240]) .+ 90
    C = Float32.([50 50 50
                  50 50 50
                  50 50 60])
    L = Float32.([60 60 60
                  20 20 20
                  10 10 10])
    itp_H =  interpolate(knots, H, Gridded(Linear()))
    itp_C =  interpolate(knots, C, Gridded(Linear()))
    itp_L =  interpolate(knots, L, Gridded(Linear()))
    return RGB.([LCHuvA{Float32}(itp_L[y, x], itp_C[y, x], itp_H[y, x], 1.0) for (x, y) in zip(rois[:ei_θ], rois[:τw])])
end

function scale_lin(x, xmin, xmax, ymin, ymax)
    return ymin + (clamp(x, xmin, xmax) - xmin) *(ymax-ymin)/(xmax-xmin)
end


function continuous_colors_from_rois(rois; h_start=90, h_end=350, l_start=60, l_end=10, τ_threshold=10, chroma=50)
    return RGB.(LCHuvA{Float32}.(scale_lin.(rois.τw,0,τ_threshold, l_start,l_end),
                                scale_lin.(rois.τw,0,τ_threshold, 60, 40),
                                scale_lin.(rois.ei_θ, 0.0,1.9, h_start, h_end)))

end

function scale_quad(x, xmin, xmax, ymin, ymax)
    return ymin + (ymax-ymin) * (1-((clamp(x, xmin, xmax) - xmin)*2/(xmax-xmin)-1)^2)
end

function continuous_color_by_parameter(vals; hue_min=90, hue_max=270,
    param_min=0, param_max=5, luminance_min=70, luminance_max=40, chroma_min=60, chroma_max=30)
    return RGB.(LCHuv{Float32}.(scale_lin.(vals, param_min, param_max, luminance_min, luminance_max),
                                scale_quad.(vals, param_min, param_max, chroma_min, chroma_max),
                                scale_lin.(vals, param_min, param_max, hue_min, hue_max)))
end


function make_colorbar(n_steps=256; width=20, hue_min=90, hue_max=270, luminance_min=70, luminance_max=40, chroma_min=60, chroma_max=30)
    vals = linspace(0, 1, n_steps)
    return transpose(repeat(RGB.(LCHuv{Float32}.(scale_lin.(vals, 0, 1, luminance_min, luminance_max),
                                scale_quad.(vals, 0, 1, chroma_min, chroma_max),
                                scale_lin.(vals, 0, 1, hue_min, hue_max))), inner=(1, width)))
    end

function circular_colors_from_rois(rois; luminance=50, chroma=50)
    return RGB.(LCHuvA{Float32}.(luminance,
                                chroma,
                                scale_lin.(atan2.(rois[:L],rois[:R]), -π/2, π, 0, 320)))

end

function make_diverging_colormap(h1, h2, mid_br, end_br; n_colors=60, alpha=1.0)
    lcolor = LCHuvA{Float32}(end_br, MSC(h1, end_br), h1, alpha)
    rcolor = LCHuvA{Float32}(end_br, MSC(h2, end_br), h2, alpha)
    midcolorl = LCHuvA{Float32}(mid_br,0,h1, alpha)
    midcolorr = LCHuvA{Float32}(mid_br,0,h2, alpha)
    return cat(1, linspace(lcolor, midcolorl, n_colors//2),
        linspace( midcolorr, rcolor, n_colors//2))
end

function make_diverging_colormap_linear_brightness(h1, hm, h2, br1, br2; n_colors=60, alpha=1.0)
    lcolor = LCHuvA{Float32}(br1, MSC(h1, br1), h1, alpha)
    rcolor = LCHuvA{Float32}(br2, MSC(h2, br2), h2, alpha)
    midcolor = LCHuvA{Float32}((br1+br2)/2,MSC(hm, (br1+br2)/2), hm, alpha)
    return cat(1, linspace(lcolor, midcolor, n_colors//2),
        linspace( midcolor, rcolor, n_colors//2))
end

function make_circular_colormap(br; n_colors=60, alpha=1.0)
    s = minimum(MSC.(0:360, br))
    scolor = LCHuvA{Float32}(br, s, 0, alpha)
    ecolor = LCHuvA{Float32}(br, s, 359, alpha)
    return RGB.(linspace(scolor,ecolor, n_colors))
end

"Another way to color the ROIs is to separate them into discrete categories
depending on excitation and inhibition and grading brightness by tau"
function category_colors_from_rois(rois, luminance=nothing, max_τ=10)
    start_c =20
    colors = Array{RGB,1}()
    for (ei, τ) in zip(rois[:ei_θ], rois[:τw])
        if isa(luminance, Nothing)
            l = 75-clamp(τ,0,max_τ)*(40/max_τ)
        else
            l = luminance
        end
        # The colors were chosen from Color Brewer and checked for
        # color-blind friendliness
        if ei < π/8
            c, h =  56, 268
        elseif ei < 3π/8
            c, h =51, 155
        else
            c, h = 112, 25
        end
        push!(colors, RGB(LCHuv(l, c, h)))
    end
    return colors
end

function colormap_c_h(c, h, lmin=75, lmax=40, n_colors=256)
    return RGB.(LCHuv.(linspace(lmin,lmax,n_colors), c, h))
end
