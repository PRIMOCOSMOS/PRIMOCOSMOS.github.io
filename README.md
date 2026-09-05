# PRIMOCOSMOS — Multimodal Signal Lab

Dark, interactive personal homepage with a dedicated qMRI research notebook. The homepage opens on an independent deep-space cover; Research leads to three selectable academic-note tabs for MMHVAE, PnP-CoSMo, and MMVAE++.

Each paper note follows the same reading order: translated abstract, short orientation, detailed Method, visible equations, and concise evaluation. The architecture views are browser-native SVG and Canvas, preserve the full training topology, support keyboard input, and remain horizontally scrollable on narrow screens.

The note prose can be edited directly in the site through **编辑正文**. Saved copy is stored in the current browser with `localStorage`; **恢复默认** clears those local overrides. Formula fields use a conventional math-font stack led by Cambria Math, while diagram microcopy keeps the compact telemetry type treatment.

## Local development

```bash
npm install
npm run dev
```

Production build:

```bash
npm run build
npm run preview
```

## GitHub Pages

The workflow in `.github/workflows/deploy.yml` builds `dist/` and deploys it through GitHub Pages. In the repository settings, select **GitHub Actions** as the Pages source.

## Research-content boundary

The site distinguishes paper-reported facts from interactive explanatory redraws. Canvas-rendered MRI slices, temperature readouts, and trend curves are labeled as mechanism illustrations rather than experimental measurements.
