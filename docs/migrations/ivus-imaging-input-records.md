# IVUS imaging input records

The IVUS imaging operations now accept operation-specific input records. This
is a breaking Rust API change. Migrate each call by naming the operation's
fields and passing one record; the arrays remain borrowed and are not copied.

The affected function and record pairs are:

- `ivus_polar_bmode_rf` -> `IvusPolarBmodeRfInput`;
- `ivus_chapter_metrics` -> `IvusChapterMetricsInput`;
- `ivus_therapy_response` -> `IvusTherapyResponseInput`; and
- `ivus_therapy_fields` -> `IvusTherapyFieldsInput`.

Before:

```rust,ignore
ivus_polar_bmode_rf(
    &x_m, &y_m, &backscatter, &attenuation_db_cm_mhz,
    &r_axis_m, &theta_axis_rad, catheter_radius_m,
    frequency_hz, ring_amplitude, ring_width_m,
)?;
```

After:

```rust,ignore
ivus_polar_bmode_rf(IvusPolarBmodeRfInput {
    x_m: &x_m,
    y_m: &y_m,
    backscatter: &backscatter,
    attenuation_db_cm_mhz: &attenuation_db_cm_mhz,
    r_axis_m: &r_axis_m,
    theta_axis_rad: &theta_axis_rad,
    catheter_radius_m,
    frequency_hz,
    ring_amplitude,
    ring_width_m,
})?;
```

Apply the same field-for-argument mapping to the metrics record. The therapy
records additionally group the five anatomy masks and seven shared dose
scalars. Construct those nested records once and reuse them:

```rust,ignore
let masks = IvusTissueMasks {
    eel_mask: &eel_mask,
    lumen_mask: &lumen_mask,
    fibrous_cap_mask: &fibrous_cap_mask,
    lipid_mask: &lipid_mask,
    plaque_mask: &plaque_mask,
};
let dose = IvusTherapyDose {
    catheter_radius_m,
    therapy_frequency_hz,
    therapy_duty_cycle,
    therapy_sonication_s,
    density_kg_m3,
    sound_speed_m_s,
    specific_heat_j_kg_k,
};

ivus_therapy_response(IvusTherapyResponseInput {
    pressure_pa: &pressure_pa,
    radius_m: &radius_m,
    attenuation_db_cm_mhz: &attenuation_db_cm_mhz,
    masks,
    dose,
    delivery_radial_center_m,
    delivery_radial_width_m,
})?;

ivus_therapy_fields(IvusTherapyFieldsInput {
    radius_m: &radius_m,
    theta_rad: &theta_rad,
    attenuation_db_cm_mhz: &attenuation_db_cm_mhz,
    masks,
    dose,
    therapy_pressure_pa,
    therapy_azimuth_rad,
    therapy_sector_width_rad,
    pressure_attenuation_length_m,
    delivery_radial_center_m,
    delivery_radial_width_m,
})?;
```

Do not retain a positional wrapper; update each call site to the record named
by its operation.
