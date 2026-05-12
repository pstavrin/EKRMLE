module EKRMLE

using LinearAlgebra
using Distributions
using CairoMakie
using Random
using StatsBase
using BlockDiagonals
using MAT
using ControlSystems
using MatrixEquations
using JLD2
using PolynomialModelReductionDataset: Heat2DModel, integrate_model_fast, build_fast_be_solver, FastDenseSolver
using ProgressMeter
using UniqueKronecker: invec
using ColorSchemes
using LaTeXStrings
using Dates
using SparseArrays


include("EKRMLEHelpers.jl")
include("DarcyHelpers2D.jl")
include("HeatEqHelpers2D.jl")
include("LinearHelpers.jl")
include("EKI.jl")


end # module EKRMLE
