import os

source = r"d:\kwavers\kwavers\src\clinical\therapy\swe_3d_workflows.rs"
dest_dir = r"d:\kwavers\kwavers\src\clinical\therapy\swe_3d_workflows"

with open(source, "r", encoding="utf-8") as f:
    lines = f.readlines()

os.makedirs(dest_dir, exist_ok=True)

with open(os.path.join(dest_dir, "mod.rs"), "w", encoding="utf-8") as f:
    f.writelines(lines[0:21])
    f.write("\n")
    f.write("pub mod roi;\n")
    f.write("pub mod elasticity_map;\n")
    f.write("pub mod statistics;\n")
    f.write("pub mod decision_support;\n")
    f.write("pub mod reconstruction;\n\n")
    f.write("pub use roi::VolumetricROI;\n")
    f.write("pub use elasticity_map::{ElasticityMap3D, ElasticityMap2D};\n")
    f.write("pub use statistics::VolumetricStatistics;\n")
    f.write("pub use decision_support::{ClinicalDecisionSupport, TissueReference, LiverFibrosisStage, BreastLesionClassification, FibrosisStage, ClassificationConfidence};\n")
    f.write("pub use reconstruction::{MultiPlanarReconstruction, SliceOrientation, SlicePositions};\n")

with open(os.path.join(dest_dir, "roi.rs"), "w", encoding="utf-8") as f:
    f.write("use crate::domain::grid::Grid;\n\n")
    f.writelines(lines[26:123])
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n")
    f.writelines(lines[749:775])
    f.write("}\n")

with open(os.path.join(dest_dir, "elasticity_map.rs"), "w", encoding="utf-8") as f:
    f.write("use crate::domain::grid::Grid;\n")
    f.write("use ndarray::{Array3, Axis};\n")
    f.write("use super::roi::VolumetricROI;\n")
    f.write("use super::statistics::VolumetricStatistics;\n\n")
    f.writelines(lines[124:312])
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n")
    f.writelines(lines[776:845])
    f.write("}\n")

with open(os.path.join(dest_dir, "statistics.rs"), "w", encoding="utf-8") as f:
    f.writelines(lines[313:337])

with open(os.path.join(dest_dir, "decision_support.rs"), "w", encoding="utf-8") as f:
    f.write("use std::collections::HashMap;\n")
    f.write("use super::statistics::VolumetricStatistics;\n\n")
    f.writelines(lines[338:613])
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n")
    f.writelines(lines[846:930])
    f.writelines(lines[958:984])
    f.write("}\n")

with open(os.path.join(dest_dir, "reconstruction.rs"), "w", encoding="utf-8") as f:
    f.write("use super::elasticity_map::{ElasticityMap3D, ElasticityMap2D};\n\n")
    f.writelines(lines[614:743])
    f.write("\n#[cfg(test)]\nmod tests {\n    use super::*;\n")
    f.write("    use crate::domain::grid::Grid;\n")
    f.writelines(lines[930:958])
    f.write("}\n")

print("swe_3d_workflows correctly extracted successfully!")
