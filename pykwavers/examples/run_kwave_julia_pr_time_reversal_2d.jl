# ============================================================================
# k-wave-julia 2-D photoacoustic time-reversal parity driver
# ============================================================================
# Runs:
#   (a) forward propagation of two p0 discs through homogeneous water,
#       recording pressure on a line sensor; and
#   (b) time-reversal reconstruction by re-injecting the time-reversed
#       sensor data through the same kspace_first_order solver.
#
# Outputs:
#   - sensor pressure matrix (CSV, n_sensors x nt)
#   - reconstructed image (CSV, nx x ny)
#   - p0 source image (CSV, nx x ny)
#   - meta (JSON)
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
    c0       = parse(Float64, args["c0"])
    rho0     = parse(Float64, args["rho0"])
    sensor_x = parse(Int, args["sensor-x-1based"])
    pml_size = parse(Int, get(args, "pml-size", "20"))

    out_sensor = args["out-sensor-csv"]
    out_recon  = args["out-recon-csv"]
    out_p0     = args["out-p0-csv"]
    out_meta   = args["out-meta"]

    kgrid = KWaveGrid(nx, dx, ny, dy)
    make_time!(kgrid, c0)
    nt = kgrid.Nt[]
    dt = kgrid.dt[]

    medium = KWaveMedium(sound_speed = c0, density = rho0)

    # Two p0 disc sources placed inside the non-PML interior. The sensor
    # line lives at i=sensor_x (typically i=1, in the PML); the discs must
    # sit in the central interior so both engines actually propagate them.
    p0 = zeros(nx, ny)
    cx_interior = pml_size + (nx - 2 * pml_size) ÷ 2   # mid of interior
    d1 = make_disc(nx, ny, cx_interior - 5, ny ÷ 2 - 6, 2)
    d2 = make_disc(nx, ny, cx_interior + 4, ny ÷ 2 + 6, 3)
    p0[d1] .= 1.0
    p0[d2] .= 0.5
    source = KWaveSource(p0 = p0)

    # Line sensor at i = sensor_x for ALL j.
    sensor_mask = falses(nx, ny)
    sensor_mask[sensor_x, :] .= true
    sensor = KWaveSensor(mask = sensor_mask, record = [:p])

    println("[julia] forward kspace_first_order nx=$nx ny=$ny nt=$nt")
    fwd = kspace_first_order(
        kgrid, medium, source, sensor;
        pml_size = pml_size, plot_sim = false,
    )

    p_data = fwd[:p]   # (n_sensors, nt) in column-major mask order.

    # Save sensor matrix in column-major order so positions can be recomputed
    # by the Python comparator deterministically.
    open(out_sensor, "w") do io
        for r in 1:size(p_data, 1)
            write(io, join(p_data[r, :], ","))
            write(io, "\n")
        end
    end

    open(out_p0, "w") do io
        for j in 1:ny
            write(io, join(p0[:, j], ","))
            write(io, "\n")
        end
    end

    # Time-reversal reconstruction using KWave.jl's native TR path.
    println("[julia] time_reversal kspace_first_order")
    sensor_tr = KWaveSensor(
        mask                         = sensor_mask,
        time_reversal_boundary_data  = fwd[:p],
    )
    source_tr = KWaveSource()  # no source for reconstruction
    tr = kspace_first_order(
        kgrid, medium, source_tr, sensor_tr;
        pml_size = pml_size, plot_sim = false,
    )
    # KWave.jl returns p_final as a flattened Vector for 2-D grids; reshape
    # it column-major back to (nx, ny).
    recon_raw = tr[:p_final]
    recon = ndims(recon_raw) == 1 ? reshape(recon_raw, nx, ny) : recon_raw

    open(out_recon, "w") do io
        for j in 1:ny
            write(io, join(recon[:, j], ","))
            write(io, "\n")
        end
    end

    open(out_meta, "w") do io
        JSON.print(io, Dict(
            "nx"=>nx,"ny"=>ny,"dx"=>dx,"dy"=>dy,"nt"=>nt,"dt"=>dt,
            "c0"=>c0,"rho0"=>rho0,
            "sensor_x_1based"=>sensor_x,
            "n_sensors"=>size(p_data, 1),
            "pml_size"=>pml_size,
            "p0_max"=>maximum(p0),
            "recon_max"=>maximum(recon),
            "engine"=>"KWave.jl/kspace_first_order + TR",
        ))
    end
    println("[julia] done: peak p0=$(maximum(p0)) peak recon=$(maximum(recon))")
end

main()
