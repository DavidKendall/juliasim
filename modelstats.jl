module ModelStats

using Statistics, Plots

include("model.jl")
import .SwarmModel as SM
using .SwarmModel

export plot_mean_distances, knn_mean_distances, knn_d

#function agent_perimeter_status(config_file="./config/base_400.json"; n_steps=500, boundary=50)
#b, parameters = load_swarm(config_file)
#n_agents = size(b)[1]
#accum = zeros(n_agents, n_steps)
#for i in 1:n_steps
#compute_step(b; parameters...)
#accum[:, i] .= b[:, SM.PRM]
#apply_step(b)
#end
#return vec(Int.((sum(accum, dims=2) ./ n_steps .* 100.0) .> boundary) .+ 1)
#end

function agent_perimeter_status(b, parameters; n_steps=500, boundary=50)
    n_agents = size(b)[1]
    accum = zeros(n_agents, n_steps)
    b_, parameters_ = deepcopy(b), deepcopy(parameters)
    for i in 1:n_steps
        compute_step(b_; parameters_...)
        accum[:, i] .= b_[:, SM.PRM]
        apply_step(b_)
    end
    return vec(Int.((sum(accum, dims=2) ./ n_steps .* 100.0) .> boundary) .+ 1)
end

function knn_d(k, mag, cb, p, perim_pair, nbr_mag)
    n_agents, max_n = size(nbr_mag)
    nbr_mag .= 0.
    Threads.@threads for i in 1:n_agents
        n = 0
        for j in 1:n_agents
            if (p[i], p[j]) == perim_pair && mag[j,i] ≤ cb
                n += 1
                nbr_mag[i, n] = mag[j, i]
            end
        end
        sort!(@view(nbr_mag[i, 1:n]))
        if k < n
            nbr_mag[i, k+1 : n] .= 0.
        end
    end
    not_zero = nbr_mag .!= 0.
    return mean(nbr_mag[not_zero]), std(nbr_mag[not_zero])
end

function knn_mean_distances(b, parameters; n_steps=500, class_ids=[:ii, :pi, :pp], k=[2,1,1,2], perimeter=nothing)
    n_agents = size(b)[1]
    id_to_int = Dict(:ii => 1, :ip => 2, :pi => 3, :pp => 4)
    id_to_perim_pair = Dict(:ii => (1, 1), :ip => (1, 2), :pi => (2, 1), :pp => (2, 2))
    n_classes = length(class_ids)
    means = Array{Float64}(undef, n_steps, n_classes)
    stds = Array{Float64}(undef, n_steps, n_classes)
    for i in 1:n_steps
        xv, yv, mag, p = compute_step(b; parameters...)
        if perimeter !== nothing
            p = perimeter
        end
        max_n = Int(maximum(b[:, SM.COH_N]))
        nbr_mag = Array{Float64}(undef, n_agents, max_n)
        cn = 0
        for c in class_ids
            cn += 1
            cl = id_to_int[c]
            pp = id_to_perim_pair[c]
            m, s = knn_d(k[cl], mag, parameters[:cb], p, pp, nbr_mag)
            means[i, cn] = m
            stds[i, cn] = s
        end
        apply_step(b)
    end
    return means, stds
end

function plot_mean_distances(config_file="config/base_400.json"; means=nothing, stds=nothing, n_steps=500, plots=[:ii, :pi, :pp], k=[2,1,1,2], pre_p=false, boundary=50, with_stdev=false, legend=:best, ax_min_max=nothing, saved_figure=false, overrides=nothing)
    b, parameters = load_swarm(config_file)
    parameters = overrides === nothing ? parameters : merge(parameters, overrides)
    if pre_p
        p = agent_perimeter_status(b, parameters, n_steps=n_steps, boundary=boundary)
    else
        p = nothing
    end
    if means === nothing
        means, stds = knn_mean_distances(b, parameters, n_steps=n_steps, class_ids=plots, k=k, perimeter=p)
    end
    ptype = with_stdev ? "\\Psi_d" : "\\mu_d"
    file_basename = splitext(basename(config_file))[1]
    title = "Distance metric by perimeter class - $(file_basename)\n $(stringify_parameters(parameters))"
    labels = Dict(:ii => "\$$(ptype)(S_i, S_i, $(k[1]))\$",
                  :ip => "\$$(ptype)(S_i, S_p, $(k[2]))\$",
                  :pi => "\$$(ptype)(S_p, S_i, $(k[3]))\$",
                  :pp => "\$$(ptype)(S_p, S_p, $(k[4]))\$")
    colors = Dict(:ii => "black", :ip => "green", :pi => "blue", :pp => "red")
    ribbons = [with_stdev ? stds[:, i] : nothing for i in 1:length(plots)]
    plt = plot(plot_title=title, plot_titlefontsize=10, xlabel="Simulation step number", ylabel="$(ptype)")
    for i in 1:length(plots)
        p = plots[i]
        plot!(plt, means[:, i], ribbon=ribbons[i], color=colors[p], fillalpha=0.25, label=labels[p], legend=legend)
    end
    if saved_figure
        savefig(plt, "$(file_basename).pdf")
    end
    display(plt)
end

function stringify_parameters(params)
    pns = Dict(:cb => "C:", :rb => "R:", :kc => "kc:", :kr => "kr:",
               :kg => "kg:", :rgf => "rgf:", :gain => "gain:"
              )
    trl = Dict(:cb => "", :rb => "\n", :kc => "\n", :kr => "\n",
               :kg => "", :rgf => "", :gain => ""
              )
    result = ""
    for n in [:cb, :rb, :kc, :kr, :kg, :rgf, :gain]
        if get(params, n, nothing) != SM.default_swarm_params[n]
            result *= "$(pns[n]) $(params[n]) $(trl[n])"
        end
    end
    return result
end

end # module
