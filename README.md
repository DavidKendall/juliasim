## JuliaSim

A swarm simulator in Julia (see [Void reduction in self-healing swarms, Eliot et al.](https://direct.mit.edu/isal/proceedings/isal2019/31/87/99204))

### Installation

Clone this repository, `git clone https://github.com/DavidKendall/juliasim`

To get the latest stable version of Julia go to the downloads page (https://julialang.org/downloads/) and download Generic Linux on x86 64 bit (glibc). Unpack to some suitable sub-directory of $HOME and configure your PATH to search `<latest_stable_julia>/bin` for executables. 

Set `juliasim` as your current working directory and start the Julia REPL:

```bash
$ cd juliasim
$ julia --project=.
```

To create a multi-threaded Julia kernel for use with Jupyter do something like:

```julia-repl
julia> using IJulia
julia> IJulia.installkernel("Julia Multi-threaded", env=Dict(
    "JULIA_NUM_THREADS" => "auto",
))
```

Start a new notebook session:

```bash 
$ jupyter notebook
```

### Usage

In the `jupyter notebook` session, browse to `juliasim/src` and open `simulator_jl.ipynb`.
Select your new mulit-threaded kernel from the dropdown menu at `Kernel -> Change kernel`.
Now run all cells (`Cell -> Run All`) -- this may take 20 minutes or so to complete, depending
on your machine.
