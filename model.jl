module SwarmModel

using Random, JSON

export compute_step, apply_step, load_swarm

const POS_X = 1
const POS_Y = 2
const COH_X = 3
const COH_Y = 4
const REP_X = 5
const REP_Y = 6
const DIR_X = 7
const DIR_Y = 8
const RES_X = 9
const RES_Y = 10
const GOAL_X = 11
const GOAL_Y = 12
const PRM = 13
const GAP_X = 14
const GAP_Y = 15
const COH_N = 16
const REP_N = 17
const DEBUG = 18
const N_COLS = 18

const default_swarm_params =
Dict(
     :cb => 3.0,
     :rb => [2.0 2.0; 2.0 2.0],
     :kc => [0.15 0.15; 0.15 0.15],
     :kr => [50.0 50.0; 50.0 50.0],
     :kd => 0.0,
     :kg => 0.0,
     :scaling => "linear",
     :exp_rate => 0.2,
     :speed => 0.05,
     :stability_factor => 0.0,
     :perim_coord => false,
     :rgf => false,
     :gain => nothing
    )

function mk_rand_swarm(n; goal=[0. 0.], loc=0., grid=10., seed=nothing)
    b = zeros((n,N_COLS))
    rng = Random.seed!(seed)
    xs = (rand(rng, Float64, (n,1)) * 2. .- 1.) .* grid .+ loc
    ys = (rand(rng, Float64, (n,1)) * 2. .- 1.) .* grid .+ loc
    b[:,POS_X] = xs
    b[:,POS_Y] = ys
    b[:,GOAL_X:GOAL_Y] .= goal
    return b
end

function mk_swarm(xs, ys; goal=[0. 0.])
    n = length(xs)
    b = zeros((n,N_COLS))
    b[:,POS_X] = xs
    b[:,POS_Y] = ys
    b[:,GOAL_X:GOAL_Y] .= goal
    return b
end

function all_pairs_mag(b::Matrix{Float64}, cb::Float64)
    n_agents = size(b)[1]
    b[:,COH_N] .= 0.
    xv = Array{Float64,2}(undef, (n_agents, n_agents))
    yv = Array{Float64,2}(undef, (n_agents, n_agents))
    mag = Array{Float64,2}(undef, (n_agents, n_agents))
    for j in 1:n_agents
        xv[j,j] = 0.
        yv[j,j] = 0.
        mag[j,j] = Inf # ensure no agent is a cohesion neighbour of itself
        for i in (j+1):n_agents
            xv[i,j] = b[i,POS_X] - b[j,POS_X]
            yv[i,j] = b[i,POS_Y] - b[j,POS_Y]
            mag[i,j] = √(xv[i,j]^2 + yv[i,j]^2)
            xv[j,i] = -xv[i,j]
            yv[j,i] = -yv[i,j]
            mag[j,i] = mag[i,j]
            if mag[i,j] ≤ cb
                b[i,COH_N] += 1
                b[j,COH_N] += 1
            end
        end
    end
    return xv, yv, mag
end

function compute_coh(b,xv,yv,mag,cb,kc,p)
    n_agents = size(b)[1]
    Threads.@threads for i in 1:n_agents
        b[i,COH_X] = 0.
        b[i,COH_Y] = 0.
        for j in 1:n_agents
            if mag[j,i] ≤ cb
                b[i, COH_X] += xv[j,i] * kc[p[i],p[j]]
                b[i, COH_Y] += yv[j,i] * kc[p[i],p[j]]
            end
        end
    end
end

function compute_rep_linear(b,xv,yv,mag,rb,kr,p)
    n_agents = size(b)[1]
    Threads.@threads for i in 1:n_agents
        b[i,REP_N] = 0.0
        b[i,REP_X] = 0.0
        b[i,REP_Y] = 0.0
        for j in 1:n_agents
            if mag[j, i] <= rb[p[i],p[j]] # assume mag[j,j] == cb + 1. > rb[x,y] for all x, y
                b[i,REP_N] = b[i,REP_N] + 1
                b[i,REP_X] = b[i,REP_X] + (1. - (rb[p[i],p[j]] / mag[j,i])) * xv[j,i] * kr[p[i],p[j]]
                b[i,REP_Y] = b[i,REP_Y] + (1. - (rb[p[i],p[j]] / mag[j,i])) * yv[j,i] * kr[p[i],p[j]]
            end
        end
    end
end

function on_perim(b,xv,yv,mag,cb,kg,rgf)
    n_agents = size(b)[1]
    b[:,PRM] .= 0. # b[i,PRM] will be either 0 or 1 and can act as a boolean
    p = ones(Int64, n_agents) # p[i] will be either 1 or 2 and can act as an index into an array
    Threads.@threads for i in 1:n_agents
        b[i,GAP_X] = 0.
        b[i,GAP_Y] = 0.
        if b[i,COH_N] < 3
            p[i] = 2
            b[i, PRM] = 1.
            continue
        end
        nbrs = findall(mag[:, i] .≤ cb)
        n_nbrs = length(nbrs)
        ang = atan.(yv[nbrs,i], xv[nbrs,i])
        nbrs = sort!(collect(zip(ang, nbrs)))
        for j in 1:n_nbrs
            k = (j % n_nbrs) + 1
            if mag[nbrs[k][2],nbrs[j][2]] > cb
                p[i] = 2
                b[i,PRM] = 1.
                b[i, GAP_X] += kg * ((0.5 * (b[nbrs[k][2],POS_X] + b[nbrs[j][2],POS_X])) - b[i,POS_X])
                b[i, GAP_Y] += kg * ((0.5 * (b[nbrs[k][2],POS_Y] + b[nbrs[j][2],POS_Y])) - b[i,POS_Y])
                break
            else
                delta = nbrs[k][1] - nbrs[j][1]
                if delta < 0
                    delta += 2π
                end
                if delta > π
                    p[i] = 2
                    b[i,PRM] = 1.
                    if rgf
                        b[i, DEBUG] += 1.0
                        b[i, GAP_X] += kg * ((0.5 * (b[nbrs[k][2],POS_X] + b[nbrs[j][2],POS_X])) - b[i,POS_X])
                        b[i, GAP_Y] += kg * ((0.5 * (b[nbrs[k][2],POS_Y] + b[nbrs[j][2],POS_Y])) - b[i,POS_Y])
                    end
                    break
                end
            end
        end
    end
    return p
end

function update_resultant(b, stability_factor, speed)
    n_agents = size(b)[1]
    for i in 1:n_agents
        mag_res = √(b[i,RES_X] ^ 2 + b[i,RES_Y] ^ 2)
        if mag_res > stability_factor * speed
            b[i,RES_X] = b[i,RES_X] / mag_res * speed
            b[i,RES_Y] = b[i,RES_Y] / mag_res * speed
        else
            b[i,RES_X] = 0.0
            b[i,RES_Y] = 0.0
        end
    end
end

function compute_step(b; scaling="linear",exp_rate=0.2,speed=0.05,perim_coord=false,stability_factor=0.,cb=3.0, rb=Array{Float64,2}([2. 2.; 2. 2.]),kc=Array{Float64,2}([0.15 0.15; 0.15 0.15]),kr=Array{Float64,2}([50. 50.; 50. 50.]),kd=0.,kg=0.,rgf=false, gain=nothing)
    n_agents = size(b)[1]
    xv,yv,mag = all_pairs_mag(b, cb)

    p = on_perim(b,xv,yv,mag,cb,kg,rgf)

    compute_coh(b,xv,yv,mag,cb,kc,p)
    b[:,COH_X:COH_Y] ./= max.(b[:,COH_N], 1.)

    compute_rep_linear(b, xv, yv, mag, rb, kr, p)
    b[:,REP_X:REP_Y] ./= max.(b[:,REP_N], 1.)

    # compute the direction vectors
    b[:,DIR_X:DIR_Y] = kd .* (b[:,GOAL_X:GOAL_Y] .- b[:,POS_X:POS_Y])

    # compute the resultant of the cohesion, repulsion and direction vectors
    if perim_coord
        b[:,DIR_X:DIR_Y] .*= b[:,PRM]
    end
    b[:,RES_X:RES_Y] = b[:,COH_X:COH_Y] .+ b[:,GAP_X:GAP_Y] .+ b[:,REP_X:REP_Y] .+ b[:,DIR_X:DIR_Y]
    if gain === nothing
        # normalise the resultant and update for speed, adjusted for stability
        update_resultant(b, stability_factor, speed)
    else
        # scale by the gain
        b[:,RES_X:RES_Y] .*= gain
    end
    return xv,yv,mag,p
end

function apply_step(b)
    """
    Assuming the step has been computed so that RES fields are up to date, update positions
    """
    #b[:,POS_X:POS_Y] .+= b[:,RES_X:RES_Y]
    #round.(b[:,POS_X:POS_Y], digits=9)
    for i in 1:size(b)[1]
        b[i,POS_X] = round(b[i,POS_X] + b[i,RES_X], digits=9)
        b[i,POS_Y] = round(b[i,POS_Y] + b[i,RES_Y], digits=9)
    end
end

function load_swarm(path="swarm.json")
    state = JSON.parsefile(path)
    b = mk_swarm(state["agents"]["coords"][1], state["agents"]["coords"][2])
    params = state["params"]
    params = Dict(collect(zip(map(Symbol, collect(keys(params))), values(params))))
    params[:rb] = transpose(reshape(collect(Iterators.flatten(params[:rb])), (2,2)))
    params[:kc] = transpose(reshape(collect(Iterators.flatten(params[:kc])), (2,2)))
    params[:kr] = transpose(reshape(collect(Iterators.flatten(params[:kr])), (2,2)))
    return b, params
end

end # module
