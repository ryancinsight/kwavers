# ============================================================================
# k-wave-julia 2-D delay-and-sum beamforming parity driver
# ============================================================================
# Forward-propagates a single point scatterer through homogeneous water,
# records pressure on a linear sensor array, and runs KWave.jl's native
# beamform_delay_and_sum reconstruction. Outputs:
#   - sensor pressure matrix (CSV, n_sensors × nt)
#   - sensor x positions (CSV, n_sensors)
#   - reconstructed image (CSV, Nx_img × Nz_img)
#   - meta JSON (with grid_x, grid_z so the Python comparator runs
#     pykwavers' analog on the same imaging grid)
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
    nx        = parse(Int, args["nx"])
    ny        = parse(Int, args["ny"])
    dx        = parse(Float64, args["dx"])
    c0        = parse(Float64, args["c0"])
    rho0      = parse(Float64, args["rho0"])
    src_x     = parse(Int, args["src-x-1based"])
    src_y     = parse(Int, args["src-y-1based"])
    sens_x    = parse(Int, args["sensor-x-1based"])
    sens_y_lo = parse(Int, args["sensor-y-lo-1based"])
    sens_y_hi = parse(Int, args["sensor-y-hi-1based"])
    pml_size  = parse(Int, get(args, "pml-size", "20"))

    out_sensor   = args["out-sensor-csv"]
    out_pos      = args["out-positions-csv"]
    out_image    = args["out-image-csv"]
    out_meta     = args["out-meta"]

    kgrid = KWaveGrid(nx, dx, ny, dx)
    make_time!(kgrid, c0)
    nt = kgrid.Nt[]
    dt = kgrid.dt[]

    medium = KWaveMedium(sound_speed = c0, density = rho0)

    # Single point scatterer as p0.
    p0 = zeros(nx, ny)
    p0[src_x, src_y] = 1.0
    source = KWaveSource(p0 = p0)

    # Linear sensor array: row at i = sens_x, columns sens_y_lo:sens_y_hi.
    sensor_mask = falses(nx, ny)
    sensor_mask[sens_x, sens_y_lo:sens_y_hi] .= true
    sensor = KWaveSensor(mask = sensor_mask, record = [:p])

    println("[julia] forward kspace_first_order nx=$nx ny=$ny nt=$nt")
    fwd = kspace_first_order(
        kgrid, medium, source, sensor;
        pml_size = pml_size, plot_sim = false,
    )
    p_data = fwd[:p]    # (n_sensors, nt) in column-major mask order

    # Sensor x-positions in metres (KWave.jl beamform_delay_and_sum signature).
    sensor_y_indices = collect(sens_y_lo:sens_y_hi)
    sensor_positions = (sensor_y_indices .- 1) .* dx

    # Build the imaging grid: lateral (y) range covers the array span;
    # depth (x) range starts a few cells beyond the array and goes to
    # nx - PML.
    img_y = collect(range(
        (sens_y_lo - 1) * dx,
        (sens_y_hi - 1) * dx,
        length = 64,
    ))
    img_x = collect(range(
        (sens_x + 4 - 1) * dx,    # 4 cells of buffer past the sensor row
        (nx - pml_size - 1) * dx,
        length = 64,
    ))

    println("[julia] beamform_delay_and_sum n_sensors=$(size(p_data, 1))")
    image_jl = beamform_delay_and_sum(
        p_data, sensor_positions, c0, dt,
        img_y, img_x;
        apodization = :hann,
    )
    # KWave.jl image shape: (length(grid_x), length(grid_z)) with grid_x as
    # the first arg (lateral here = y-positions). image[ix, iz] indexes
    # (lateral, depth).

    open(out_sensor, "w") do io
        for r in 1:size(p_data, 1)
            write(io, join(p_data[r, :], ","))
            write(io, "\n")
        end
    end
    open(out_pos, "w") do io
        for s in sensor_positions
            write(io, "$(s)\n")
        end
    end
    open(out_image, "w") do io
        for j in 1:size(image_jl, 2)
            write(io, join(image_jl[:, j], ","))
            write(io, "\n")
        end
    end

    open(out_meta, "w") do io
        JSON.print(io, Dict(
            "nx"=>nx,"ny"=>ny,"dx"=>dx,"nt"=>nt,"dt"=>dt,
            "c0"=>c0,"rho0"=>rho0,
            "src_x_1based"=>src_x,"src_y_1based"=>src_y,
            "sensor_x_1based"=>sens_x,
            "sensor_y_range_1based"=>[sens_y_lo, sens_y_hi],
            "n_sensors"=>length(sensor_positions),
            "img_y_m"=>img_y, "img_x_m"=>img_x,
            "image_shape_y_first"=>collect(size(image_jl)),
            "image_peak"=>maximum(abs, image_jl),
            "image_peak_loc_idx"=>collect(Tuple(argmax(abs.(image_jl)))),
            "engine"=>"KWave.jl/beamform_delay_and_sum (Hann)",
        ))
    end
    println("[julia] DAS done: peak |image| = ", maximum(abs, image_jl),
            " at idx ", argmax(abs.(image_jl)))
end

main()
