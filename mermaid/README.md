# EASTASIA-FWI Mermaid Diagrams

Scientific workflow diagrams for the EASTASIA-FWI project.

## Diagram Files

| File | Description | Recommended Theme |
|------|-------------|-------------------|
| `01_scientific_goals.mmd` | Project scientific goals and phases | github-light |
| `02_system_architecture.mmd` | 5-module system architecture | zinc-light |
| `03_data_preparation.mmd` | Data preparation pipeline | catppuccin-latte |
| `04_waveform_preprocessing.mmd` | Waveform processing steps | github-light |
| `05_model_space_analysis.mmd` | Model analysis workflow | zinc-light |
| `06_clustering_methods.mmd` | Clustering algorithm comparison | catppuccin-latte |
| `07_data_space_simulation.mmd` | SPECFEM3D simulation flow | github-light |
| `08_voting_and_bma.mmd` | Voting and BMA fusion | zinc-light |
| `09_sdl_fusion.mmd` | SDL dictionary learning pipeline | catppuccin-latte |
| `10_pgm_fusion.mmd` | PGM probabilistic fusion | github-light |
| `11_complete_dataflow.mmd` | Complete project data flow | zinc-light |
| `12_fusion_pipeline.mmd` | Full fusion pipeline v2.0 | catppuccin-latte |

## Rendered Outputs

Three theme variants are available:

```
mermaid/
├── output/                    # github-light theme (default)
├── output-zinc-light/         # zinc-light theme
└── output-catppuccin-latte/   # catppuccin-latte theme
```

## Usage

### Render with pretty-mermaid skill

```bash
# Single file
cd ~/.agents/skills/pretty-mermaid
node scripts/render.mjs \
  --input ~/Github/EASTASIA_FWI/mermaid/01_scientific_goals.mmd \
  --output ~/Github/EASTASIA_FWI/mermaid/output/01_scientific_goals.svg \
  --theme github-light

# Batch render
node scripts/batch.mjs \
  --input-dir ~/Github/EASTASIA_FWI/mermaid \
  --output-dir ~/Github/EASTASIA_FWI/mermaid/output \
  --theme github-light
```

### Available Light Themes

| Theme | Background | Style |
|-------|------------|-------|
| `github-light` | #ffffff | Clean, GitHub-style |
| `zinc-light` | #ffffff | Minimal, grayscale accents |
| `catppuccin-latte` | #eff1f5 | Warm, pastel tones |

## Design Principles

1. **No emojis** - Professional scientific diagrams
2. **Clear labels** - Descriptive node names with technical terms
3. **Logical flow** - Top-to-bottom or left-to-right progression
4. **Subgraph grouping** - Related steps organized into phases
5. **Consistent styling** - Unified appearance across all diagrams

## References

- [beautiful-mermaid](https://github.com/lukilabs/beautiful-mermaid) - Rendering engine
- [Pretty-mermaid-skills](https://github.com/imxv/Pretty-mermaid-skills) - AI skill integration
