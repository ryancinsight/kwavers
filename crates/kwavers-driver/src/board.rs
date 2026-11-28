//! Board copper shares storage with Electra and retains the driver grid specification.
pub use electra_board::board::{
    split_domain_from_name, LayerId, Net, NetClassKind, NetId, Pad, SplitDomain, Track, Via,
    ViaKind, Zone, ZoneFill,
};
/// Board over the driver routing grid.
pub type Board = electra_board::board::Board<crate::geom::GridSpec>;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_names_classify_split_domains() {
        assert_eq!(
            split_domain_from_name("ANALOG_SIG"),
            Some(SplitDomain::Analog),
            "explicit ANALOG token maps to the analog placement domain"
        );
        assert_eq!(
            split_domain_from_name("BUS_SPI_MOSI"),
            Some(SplitDomain::Digital),
            "SPI control nets map to the digital placement domain"
        );
        assert_eq!(
            split_domain_from_name("CLK_MAIN"),
            Some(SplitDomain::Digital),
            "clock nets map to the digital placement domain"
        );
        assert_eq!(
            split_domain_from_name("TX_0"),
            None,
            "unmarked high-speed output nets are not forced into an analog/digital split domain"
        );
    }
}
