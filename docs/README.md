# Documentation Index - Kwavers Acoustic Simulation Library

## Core Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [`README.md`](../README.md) | Project overview, quick start, installation | All users |
| [`prd.md`](prd.md) | Product requirements and feature specifications | Product managers, architects |
| [`srs.md`](srs.md) | Software requirements with verification criteria | Engineers, testers |
| [`adr.md`](adr.md) | Architecture decisions and design trade-offs | Senior engineers, architects |
| [`checklist.md`](checklist.md) | Development progress and status tracking | Development team |
| [`backlog.md`](backlog.md) | Sprint priorities and development tasks | Scrum masters, developers |

## Technical Documentation

| Document | Focus Area | Complexity |
|----------|------------|------------|
| [`technical/amr_usage.md`](technical/amr_usage.md) | Adaptive Mesh Refinement usage patterns | Advanced |
| [`technical/hybrid_spectral_dg.md`](technical/hybrid_spectral_dg.md) | Spectral-DG hybrid method implementation | Expert |
| [`technical/multi_rate_time_integration.md`](technical/multi_rate_time_integration.md) | Temporal coupling algorithms | Advanced |
| [`technical/plugin_architecture.md`](technical/plugin_architecture.md) | Extensibility and plugin system | Intermediate |

## User Guides

| Document | Target | Skill Level |
|----------|--------|-------------|
| [`guides/advanced_features.md`](guides/advanced_features.md) | Performance optimization, benchmarking | Advanced |

## Document Hierarchy

```
docs/
├── README.md                    # This index
├── Core Development Documents
│   ├── prd.md                   # Product requirements  
│   ├── srs.md                   # Software requirements
│   ├── adr.md                   # Architecture decisions
│   ├── checklist.md             # Progress tracking
│   └── backlog.md               # Sprint planning
├── technical/                   # Technical implementation guides
│   ├── amr_usage.md
│   ├── hybrid_spectral_dg.md
│   ├── multi_rate_time_integration.md
│   └── plugin_architecture.md
└── guides/                      # User-focused guides
    └── advanced_features.md
```

## Navigation by Role

### **New Users**
Start with: [`README.md`](../README.md) → [`guides/advanced_features.md`](guides/advanced_features.md)

### **Contributors** 
Review: [`checklist.md`](checklist.md) → [`backlog.md`](backlog.md) → [`adr.md`](adr.md)

### **Architects**
Study: [`adr.md`](adr.md) → [`prd.md`](prd.md) → [`technical/`](technical/)

### **Researchers**
Reference: [`srs.md`](srs.md) → [`technical/`](technical/) → API documentation

---

*Documentation maintained following bonsai-pruned hierarchy principles*  
*All documents verified for accuracy and completeness*