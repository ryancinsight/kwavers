# ============================================================================
# k-wave-julia diff_homogeneous_medium_source 2D parity driver
# ============================================================================
# Mirrors the MATLAB k-Wave example:
#   example_diff_homogeneous_medium_source.m  (2D Gaussian disk heat source)
#
# Runs KWave.jl kwave_diffusion on a 2D grid (Nx × Ny) with a rotationally
# symmetric Gaussian heat source and outputs the final temperature field as a
# flat CSV so the Python comparator can compare it against pykwavers.
#
# Outputs
#   --output-csv   flat CSV (row-major, ix fastest) of final T field [°C]
#   --output-meta  JSON metadata
# ============================================================================
using DelimitedFiles
using JSON
using KWave

function parse_args(args)
    values = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        if !startswith(key, "--") || i == length(args)
            error("Expected --key value pair, got '$(key)'")
        end
        values[key[3:end]] = args[i + 1]
        i += 2
    end
    return values
end

function main()
    args = parse_args(ARGS)

    nx                 = parse(Int,     args["nx"])
    ny                 = parse(Int,     args["ny"])
    dx                 = parse(Float64, args["dx"])
    dy                 = parse(Float64, args["dy"])
    nt                 = parse(Int,     args["nt"])
    dt_val             = parse(Float64, args["dt"])
    k_th               = parse(Float64, args["thermal-conductivity"])
    rho                = parse(Float64, args["density"])
    cp                 = parse(Float64, args["specific-heat"])
    perfusion          = parse(Float64, args["perfusion-rate"])   # kg/(m³·s)
    t_blood            = parse(Float64, args["blood-temperature"])
    cp_blood           = parse(Float64, args["blood-specific-heat"])
    t0                 = parse(Float64, args["initial-temperature"])
    q_peak             = parse(Float64, args["q-peak"])
    q_sigma            = parse(Float64, args["q-sigma"])           # metres
    output_csv         = args["output-csv"]
    output_meta        = args["output-meta"]

    kgrid = KWaveGrid(nx, dx, ny, dy)
    kgrid.dt[] = dt_val
    kgrid.Nt[] = nt

    medium = ThermalMedium(
        thermal_conductivity = k_th,
        density              = rho,
        specific_heat        = cp,
        perfusion_rate       = perfusion,
        blood_temperature    = t_blood,
        blood_specific_heat  = cp_blood,
    )

    # Rotationally symmetric 2D Gaussian heat source (disk in xy-plane).
    cx = nx ÷ 2
    cy = ny ÷ 2
    Q = zeros(Float64, nx, ny)
    for j in 1:ny
        for i in 1:nx
            rx = (i - cx) * dx
            ry = (j - cy) * dy
            Q[i, j] = q_peak * exp(-(rx^2 + ry^2) / (2.0 * q_sigma^2))
        end
    end
    source = ThermalSource(Q = Q, mode = :steady)

    T_final, _ = kwave_diffusion(kgrid, medium, source, t0, nt; record_every = nt)

    # Export final field as flat row-major CSV: ix fastest (matches pykwavers C-order
    # reshape when nz=1). Row i is the ix-th row; column j is the iy-th column.
    open(output_csv, "w") do io
        write(io, "# T_final_C flat, nx=$(nx) ny=$(ny), row=ix col=iy\n")
        for i in 1:nx
            row = join([string(T_final[i, j]) for j in 1:ny], ",")
            write(io, row, "\n")
        end
    end

    meta = Dict(
        "nx" => nx, "ny" => ny, "dx" => dx, "dy" => dy,
        "nt" => nt, "dt" => dt_val,
        "thermal_conductivity" => k_th,
        "density" => rho, "specific_heat" => cp,
        "perfusion_rate_kg_per_m3_s" => perfusion,
        "blood_temperature" => t_blood, "blood_specific_heat" => cp_blood,
        "initial_temperature" => t0,
        "q_peak" => q_peak, "q_sigma_m" => q_sigma,
        "cx_1based" => cx, "cy_1based" => cy,
        "T_centre_final_C" => T_final[cx, cy],
        "T_max_final_C"    => maximum(T_final),
        "engine"           => "KWave.jl/kwave_diffusion_2D",
    )
    open(output_meta, "w") do io
        JSON.print(io, meta)
    end

    println("[julia] 2D bioheat done: T_centre=", T_final[cx, cy], " C  T_max=", maximum(T_final), " C")
end

main()
