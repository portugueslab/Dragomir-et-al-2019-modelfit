using MAT
using StatsBase
using Glob

"Holds all the necessary data for sampling"
struct Timing
    dt_imaging::Float64
    dt_stim::Float64
    n_upsample::Int64
    n_frames_trial::Int64
    dt_sim::Float64
    trial_duration::Float64
end

Timing(dt_imaging, dt_stim, n_upsample, n_frames_trial) =
    Timing(dt_imaging, dt_stim, n_upsample, n_frames_trial, dt_imaging/n_upsample,
    dt_imaging*n_frames_trial)

"Holds the trace, original id, active planes and center of mass for a ROI"
mutable struct Cell
    original_id::Int64
    trace::Array{Float64, 1}
    planes::Array{Int64, 1}
    coords::Array{Int64,2}
    reference_coords::NTuple{3, Float32}
    regressors::Array{Float32,1}
end

"Z-score the trace plane-wise"
function normalize_trace_per_plane(trace, n_frames_trial)
    norm_trace = similar(trace)
    for i_plane in 1:length(trace)÷n_frames_trial
        norm_trace[(i_plane-1)*n_frames_trial+1:(i_plane)*n_frames_trial] =
            zscore(trace[(i_plane-1)*n_frames_trial+1:(i_plane)*n_frames_trial])
    end
    return norm_trace
end

"Fill a array of cell data-structure with the traces from the mat file"
function fill_cells(data, location_data=nothing; normalize_per_plane=true)
    cells = Array{Cell, 1}()
    for i in 1:length(data["Newcells"]["planes"])
        # Load the active planes
        if isa(data["Newcells"]["planes"][i], AbstractArray)
            planes = round.(Int64, data["Newcells"]["planes"][i][1,:])
        else
            planes = [round(Int64, data["Newcells"]["planes"][i])]
        end

        # Load the coordinates
        ca = Array{Int64, 2}(3, length(data["Newcells"]["Xcoords"][i]))
        if isa(data["Newcells"]["Xcoords"][i], AbstractArray)
            for (i_coord, coords) in enumerate(zip(data["Newcells"]["Xcoords"][i][:],
                              data["Newcells"]["Ycoords"][i][:],
                              data["Newcells"]["Zcoords"][i][:]))
                ca[:, i_coord] .= round.(Int64,coords)
            end
        else
            ca[:, 1] .= round.(Int64,(data["Newcells"]["Xcoords"][i],
                                    data["Newcells"]["Ycoords"][i],
                                    data["Newcells"]["Zcoords"][i]))
        end
        loc_tuple = location_data == nothing ? (-1f0, -1f0, -1f0) :
            Float32.((location_data["roilist"]["Xmorph"][i],
                      location_data["roilist"]["Ymorph"][i],
                      location_data["roilist"]["Zmorph"][i]))

        # Load the trace and optionally normalise it
        trace = data["Newcells"]["autoregtrace"][i][1,:]
        if normalize_per_plane
            trace = normalize_trace_per_plane(trace,
                        length(trace)÷length(planes))
        end

        # Make the data structure and add it to the vector
        push!(cells, Cell(i, trace,
                          planes, ca, loc_tuple,
                          data["Newcells"]["correlation"][i][:]))



    end
    return cells
end

function load_matlab(fish_id,
                     root_path = raw"J:\_data\Exp020_Perception\merged_data_structures")

    data_traces =  matread(glob("*"*fish_id*"*.mat", root_path)[1])
    println("Loaded data file ",glob("*"*fish_id*"*.mat", root_path)[1])
    coord_files = glob("*"*fish_id*"*_M.mat", joinpath(root_path, "coords"), )
    println("Loaded coord file ", coord_files)
    data_coords =  length(coord_files) == 1 ? matread(coord_files[1]) : nothing
    return fill_cells(data_traces, data_coords), data_traces["behavior"]["stim_number"][1,:]
end

function fill_stim(cell, stim_number, timing, map)
    n_t_ups = length(cell.planes)*timing.n_frames_trial*timing.n_upsample
    coh_pos = zeros(UInt8, n_t_ups)
    coh_neg = zeros(UInt8, n_t_ups)

    for i in 1:n_t_ups
        # figure out the time
        t = (i-1)*timing.dt_sim

        # in which plane of the planes where the ROI is
        # is the current t?
        i_plane = floor(Int64, t / timing.trial_duration)+1

        t_in_plane = t - (i_plane-1)*timing.trial_duration
        idx_plane = round(Int64, t_in_plane/timing.dt_stim)+1

        if 0 <=idx_plane <= size(stim_number[cell.planes[i_plane]], 2)
            val = stim_number[cell.planes[i_plane]][idx_plane]
            if val > 0.
                v1 = round(Int64, val)
                coh = get(map, v1, 0.0)
                if coh > 0
                    coh_pos[i] = round(UInt8, coh*10)
                else
                    coh_neg[i] = round(UInt8, -coh*10)
                end
            end
        end
    end


    return coh_pos, coh_neg
end

"Loads all data split across multiple HDF5 files"
function load_all_cells(root="H:/Coherence/exp_20/171018_ED_f1")
    cells = cat(1, (load(root*"_c_$(i).jld2", "cells") for i in 1:7)...);
    return cells
end
