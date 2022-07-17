include("model.jl")
import .SwarmModel as SM
using .SwarmModel

include("modelstats.jl")
import .ModelStats as MS
using .ModelStats

using Plots; plotlyjs()

b, parameters = load_swarm("../config/low_density_400.json")
b[:,SM.GOAL_X:SM.GOAL_Y] .= [0.0 0.0]
for i in 1:10000
    compute_step(b; parameters...)
    apply_step(b)
end
prm = findall(b[:,SM.PRM] .> 0.)
_prm = setdiff(1:size(b)[1], prm)
plt = scatter(b[_prm,SM.POS_X],b[_prm,SM.POS_Y]; legend=false, markersize=2, markercolor=:black,
              aspect_ratio=:equal, hover=_prm)
scatter!(b[prm,SM.POS_X],b[prm,SM.POS_Y]; legend=false, markersize=2, markercolor=:red, 
         aspect_ratio=:equal, hover=prm)
plot!([b[1,SM.GOAL_X]], [b[1,SM.GOAL_Y]], markershape=:cross)
display(plt)
# failed = [393, 399, 241, 273, 330, 150, 275, 109, 46, 240, 274, 9, 171, 349, 347, 50, 136, 343, 227, 200, 379, 79, 314, 362, 107, 334, 135, 121, 27, 293, 302, 111, 62, 256, 105, 68, 83, 64, 352, 131]
