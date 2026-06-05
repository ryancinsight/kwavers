# ============================================================================
# k-wave-julia bioheat 1D parity driver
# ============================================================================
# Mirrors the physics of KWave.jl examples/diff_bioheat_1d.jl but is fully
# parameterised over CLI args so the Python comparator owns the canonical
# parameter set.
#
# Convention (KWave.jl ThermalMedium): perfusion_rate is in [kg/(m^3*s)];
# the Pennes decay coefficient is perfusion_rate * blood_specific_heat.
#
# Pykwavers convention: perfusion_rate is in [1/s] and the decay coefficient
# is perfusion_rate * blood_density * blood_specific_heat. The Python
# comparator handles unit conversion before calling pykwavers; this driver
# uses KWave.jl's native units verbatim.
#
# Outputs:
#   --output-csv  : 2-column CSV (time_s, T_centre_C) over all Nt steps
#   --output-meta : JSON with grid/medium/source metadata
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

    nx                  = parse(Int, args["nx"])
    dx                  = parse(Float64, args["dx"])
    nt                  = parse(Int, args["nt"])
    dt                  = parse(Float64, args["dt"])
    k_th                = parse(Float64, args["thermal-conductivity"])
    rho                 = parse(Float64, args["density"])
    cp                  = parse(Float64, args["specific-heat"])
    perfusion           = parse(Float64, args["perfusion-rate"])  # kg/(m^3*s)
    t_blood             = parse(Float64, args["blood-temperature"])
    cp_blood            = parse(Float64, args["blood-specific-heat"])
    t0                  = parse(Float64, args["initial-temperature"])
    q_peak              = parse(Float64, args["q-peak"])
    q_sigma             = parse(Float64, args["q-sigma"])

    output_csv          = args["output-csv"]
    output_meta         = args["output-meta"]

    kgrid = KWaveGrid(nx, dx)
    kgrid.dt[] = dt
    kgrid.Nt[] = nt

    medium = ThermalMedium(
        thermal_conductivity = k_th,
        density              = rho,
        specific_heat        = cp,
        perfusion_rate       = perfusion,
        blood_temperature    = t_blood,
        blood_specific_heat  = cp_blood,
    )

    # Gaussian heat source centred at Nx/2.
    Q = zeros(Float64, nx)
    x_centre = nx ÷ 2
    for i in 1:nx
        Q[i] = q_peak * exp(-((i - x_centre) * dx)^2 / (2.0 * q_sigma^2))
    end
    source = ThermalSource(Q = Q, mode = :steady)

    # Run with record_every = 1 so we obtain the full time series.
    T_final, T_history = kwave_diffusion(
        kgrid, medium, source, t0, nt; record_every = 1,
    )

    # T_history[:, k] is the temperature field at recorded step k (k=1 is t=0).
    # We export the centre-cell trace so the comparator can do a 1-D Pearson
    # / RMS / PSNR analysis against pykwavers' sensor recording.
    n_rec = size(T_history, 2)
    times = (0:n_rec - 1) .* dt
    centre_trace = T_history[x_centre, :]

    open(output_csv, "w") do io
        write(io, "time_s,T_centre_C\n")
        for k in 1:n_rec
            write(io, string(times[k], ",", centre_trace[k], "\n"))
        end
    end

    meta = Dict(
        "nx" => nx, "dx" => dx, "nt" => nt, "dt" => dt,
        "thermal_conductivity" => k_th, "density" => rho, "specific_heat" => cp,
        "perfusion_rate_kg_per_m3_s" => perfusion,
        "blood_temperature" => t_blood, "blood_specific_heat" => cp_blood,
        "initial_temperature" => t0, "q_peak" => q_peak, "q_sigma_m" => q_sigma,
        "centre_index_1based" => x_centre,
        "T_centre_final_C" => T_final[x_centre],
        "T_max_final_C"    => maximum(T_final),
        "n_records"        => n_rec,
        "engine"           => "KWave.jl/kwave_diffusion",
    )
    open(output_meta, "w") do io
        JSON.print(io, meta)
    end

    println("[julia] bioheat 1D done: T_centre_final = ", T_final[x_centre], " C")
end

main()
