# ============================================================================
# k-wave-julia 2-D elastic parity driver
# ============================================================================
# Mirrors KWave.jl examples/ewp_elastic_2d.jl but uses a velocity (ux)
# source plane instead of an isotropic p0 initial condition. This is
# because pykwavers' elastic solver supports velocity sources directly via
# `Source.from_elastic_velocity_source` and aligning both engines on the
# same source semantics removes a known systematic mismatch.
#
# Output: per-sensor ux time series (CSV) + meta JSON.
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

    nx       = parse(Int, args["nx"])
    ny       = parse(Int, args["ny"])
    dx       = parse(Float64, args["dx"])
    dy       = parse(Float64, args["dy"])
    nt       = parse(Int, args["nt"])
    dt       = parse(Float64, args["dt"])
    cp       = parse(Float64, args["cp"])
    cs       = parse(Float64, args["cs"])
    rho      = parse(Float64, args["rho"])
    pml_size = parse(Int, get(args, "pml-size", "10"))
    src_x    = parse(Int, args["src-x-1based"])

    sig_csv  = args["src-signal-csv"]
    sens_csv = args["sensor-positions-csv"]
    out_csv  = args["output-csv"]
    out_meta = args["output-meta"]

    signal = vec(readdlm(sig_csv, ',', Float64))
    @assert length(signal) == nt "signal length $(length(signal)) ≠ nt $(nt)"

    sensor_pos = readdlm(sens_csv, ',', Int)  # rows of (i_1based, j_1based)
    n_sens = size(sensor_pos, 1)

    kgrid = KWaveGrid(nx, dx, ny, dy)
    kgrid.dt[] = dt
    kgrid.Nt[] = nt
    resize!(kgrid.t_array, nt)
    kgrid.t_array .= (0:nt-1) .* dt

    medium = ElasticMedium(
        sound_speed_compression = cp,
        sound_speed_shear       = cs,
        density                 = rho,
    )

    # Velocity source (ux) on a single column at i = src_x (1-based).
    u_mask = falses(nx, ny)
    u_mask[src_x, :] .= true
    n_src = count(u_mask)
    ux = repeat(signal', n_src, 1)  # (n_src, nt)

    source = ElasticSource(
        u_mask = u_mask,
        ux     = ux,
        u_mode = Additive,
    )

    sensor_mask = falses(nx, ny)
    for r in 1:n_sens
        sensor_mask[sensor_pos[r, 1], sensor_pos[r, 2]] = true
    end
    sensor = KWaveSensor(mask = sensor_mask, record = [:ux])

    println("[julia] launching pstd_elastic_2d nx=$nx ny=$ny nt=$nt n_src=$n_src n_sens=$n_sens")
    result = pstd_elastic_2d(
        kgrid, medium, source, sensor;
        pml_size = pml_size, plot_sim = false,
    )

    # KWave.jl returns velocity components as :ux, :ux in result Dict.
    # Sensor data shape: (n_sens, nt). KWave.jl orders sensors in column-major
    # over the boolean mask; we recorded the mask order as sensor_pos rows
    # (Cartesian-ordered) so we need to reindex.
    ux_data = result[:ux]
    if size(ux_data, 1) != n_sens
        error("KWave.jl returned $(size(ux_data, 1)) sensor traces, expected $(n_sens)")
    end

    # Reorder: KWave.jl returns sensors in column-major order over the mask.
    # We rebuild that order, then look up the index for each sensor_pos row.
    cm_order = Tuple{Int,Int}[]
    for j in 1:ny, i in 1:nx
        if sensor_mask[i, j]
            push!(cm_order, (i, j))
        end
    end
    cm_index = Dict(cm_order[k] => k for k in 1:length(cm_order))

    reordered = zeros(Float64, n_sens, nt)
    for r in 1:n_sens
        key = (sensor_pos[r, 1], sensor_pos[r, 2])
        reordered[r, :] .= ux_data[cm_index[key], :]
    end

    # Write CSV with one row per sensor, comma-separated samples.
    open(out_csv, "w") do io
        for r in 1:n_sens
            write(io, join(reordered[r, :], ","))
            write(io, "\n")
        end
    end

    open(out_meta, "w") do io
        JSON.print(io, Dict(
            "nx"=>nx,"ny"=>ny,"dx"=>dx,"dy"=>dy,"nt"=>nt,"dt"=>dt,
            "cp"=>cp,"cs"=>cs,"rho"=>rho,"pml_size"=>pml_size,
            "src_x_1based"=>src_x,"n_src"=>n_src,"n_sens"=>n_sens,
            "engine"=>"KWave.jl/pstd_elastic_2d",
            "source_mode"=>"Additive ux plane",
            "peak_ux_recorded"=>maximum(abs, reordered),
        ))
    end
    println("[julia] elastic 2D done: peak |ux| = ", maximum(abs, reordered))
end

main()
