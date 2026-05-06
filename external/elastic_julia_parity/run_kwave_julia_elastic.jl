# ============================================================================
# k-wave-julia elastic-wave reference driver
# ============================================================================
# Runs a 2-D homogeneous elastic simulation (compressional + shear) using
# KWave.jl's pstd_elastic_2d, with a uz velocity-source plane and a
# binary sensor mask. Outputs the recorded uz trace as CSV and metadata
# as JSON for comparison against pykwavers' SolverType.Elastic via the
# Phase A.3 velocity-source path.
#
# Invocation (from compare_elastic.py):
#     julia --project=. run_kwave_julia_elastic.jl \
#         --output-csv path.csv --output-meta path.json \
#         --nx 32 --ny 32 --dx 5e-4 --dy 5e-4 \
#         --nt 200 --dt 4.81e-8 \
#         --cp 1800 --cs 1200 --rho 1000 \
#         --source-x 8 \
#         --source-signal-csv signal.csv \
#         --sensor-positions-csv positions.csv
# ============================================================================

using DelimitedFiles
using KWave
using Printf

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
    output_csv = args["output-csv"]
    output_meta = args["output-meta"]

    nx = parse(Int, args["nx"])
    ny = parse(Int, args["ny"])
    dx = parse(Float64, args["dx"])
    dy = parse(Float64, args["dy"])
    nt = parse(Int, args["nt"])
    dt = parse(Float64, args["dt"])
    cp = parse(Float64, args["cp"])
    cs = parse(Float64, args["cs"])
    rho = parse(Float64, args["rho"])
    source_x = parse(Int, args["source-x"])  # 1-based source-plane index
    pml_size = parse(Int, get(args, "pml-size", "10"))

    # Driving signal — broadcast across the whole source plane (uz).
    signal_path = args["source-signal-csv"]
    uz_signal = vec(readdlm(signal_path, ',', Float64))
    if length(uz_signal) != nt
        error("Signal length $(length(uz_signal)) ≠ nt $(nt)")
    end

    # Sensor positions: csv with rows (i, j), 1-based column-major order.
    positions_path = args["sensor-positions-csv"]
    sensor_positions = readdlm(positions_path, ',', Int)
    n_sensors = size(sensor_positions, 1)

    # Build kgrid + time array.
    kgrid = KWaveGrid(nx, dx, ny, dy)
    # KWave.jl's setTime equivalent: explicit field assignment.
    kgrid.dt[] = dt
    kgrid.Nt[] = nt
    resize!(kgrid.t_array, nt)
    kgrid.t_array .= (0:nt-1) .* dt

    # Homogeneous elastic medium.
    medium = ElasticMedium(
        sound_speed_compression = cp,
        sound_speed_shear = cs,
        density = rho,
    )

    # Velocity source on the source-x plane (uz only).
    u_mask = falses(nx, ny)
    u_mask[source_x, :] .= true
    # KWave.jl expects per-active-point signals: shape (n_active, nt).
    n_active = sum(u_mask)
    uz_2d = repeat(reshape(uz_signal, 1, :), n_active, 1)

    # IMPORTANT: KWave.jl's pstd_elastic_2d processes vx and vy velocity
    # sources only — `source.uz` is silently ignored in 2-D (no third
    # spatial axis). Drive the compressional wave by injecting ux into
    # the source plane (the plane is normal to x, so a ux drive launches
    # a +x compressional wave). Use Additive mode (matches MATLAB k-Wave
    # default for elastic velocity sources).
    source = ElasticSource(
        u_mask = u_mask,
        ux = uz_2d,
        u_mode = Additive,
    )

    # Binary sensor mask from the supplied positions.
    sensor_mask = falses(nx, ny)
    for row in 1:n_sensors
        i = sensor_positions[row, 1]
        j = sensor_positions[row, 2]
        sensor_mask[i, j] = true
    end
    # In 2-D, only ux/uy are meaningful (no z-axis). Record ux to match
    # the driven component.
    sensor = KWaveSensor(mask = sensor_mask, record = [:ux])

    start_ns = time_ns()
    output = pstd_elastic_2d(
        kgrid, medium, source, sensor;
        pml_size = pml_size,
    )
    elapsed_s = (time_ns() - start_ns) / 1.0e9

    # Recorded ux: shape (n_sensors, nt).
    ux_data = output[:ux]
    if size(ux_data, 1) != n_sensors
        if size(ux_data, 2) == n_sensors
            ux_data = permutedims(ux_data)
        else
            error("Unexpected ux shape $(size(ux_data)); expected ($(n_sensors), $(nt))")
        end
    end

    writedlm(output_csv, ux_data, ',')

    open(output_meta, "w") do io
        @printf(io, "{\n")
        @printf(io, "  \"engine\": \"KWave.jl\",\n")
        @printf(io, "  \"function\": \"pstd_elastic_2d\",\n")
        @printf(io, "  \"nx\": %d,\n", nx)
        @printf(io, "  \"ny\": %d,\n", ny)
        @printf(io, "  \"nt\": %d,\n", nt)
        @printf(io, "  \"dx\": %.17g,\n", dx)
        @printf(io, "  \"dy\": %.17g,\n", dy)
        @printf(io, "  \"dt\": %.17g,\n", dt)
        @printf(io, "  \"cp\": %.17g,\n", cp)
        @printf(io, "  \"cs\": %.17g,\n", cs)
        @printf(io, "  \"rho\": %.17g,\n", rho)
        @printf(io, "  \"pml_size\": %d,\n", pml_size)
        @printf(io, "  \"n_sensors\": %d,\n", n_sensors)
        @printf(io, "  \"solver_seconds\": %.9f,\n", elapsed_s)
        @printf(io, "  \"peak_abs_ux\": %.17g\n", maximum(abs.(ux_data)))
        @printf(io, "}\n")
    end
end

main()
