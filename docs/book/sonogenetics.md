# Sonogenetics

## Scope

Sonogenetics covers acoustic radiation force, membrane tension, mechanosensitive channels, open probability, ion current, and neural response. Code ownership maps to `kwavers::physics::acoustics::therapy::sonogenetics` and clinical therapy integration modules.

## Theorem: Two-State Channel Open Probability

For a two-state channel with free-energy difference `Delta G`, the open probability is

```text
P_open = 1 / (1 + exp(Delta G / (k_B T))).
```

### Proof Sketch

The Boltzmann weight of a state is proportional to `exp(-G/(k_B T))`. Normalizing the open-state weight against open and closed states yields the logistic form.

## Algorithm: Sonogenetic Validation

1. Convert acoustic pressure to intensity and ARF with local medium properties.
2. Convert ARF or pressure to membrane tension using the channel model.
3. Compute channel open probability and current.
4. Validate current sign, channel ordering, and neural membrane response.

## Implementation Targets

- Keep channel parameters in one authoritative enum/table.
- Preserve pressure, tension, probability, current, and neuron state as separate values.
- Validate each channel against literature parameter ranges before neural integration.

## Research Anchors

- Sonogenetic biomolecular-function review: https://pubmed.ncbi.nlm.nih.gov/38197549/
- Mechanosensitive ion-channel methods review: https://pubmed.ncbi.nlm.nih.gov/39402780/
- PIEZO mechanotransduction review: https://www.nature.com/articles/s41580-024-00773-5
