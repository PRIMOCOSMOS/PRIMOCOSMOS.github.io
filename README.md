# PRIMOCOSMOS — Multimodal Signal Lab

Independent Silicon Idol homepage with two destinations:

- **Research archive** (`#/research/<paper>`): six searchable notes in 编码器 (MMHVAE, PnP-CoSMo, MMVAE++) and 课题核心 (SSDiff, METSC, PIGMENT). Shared recursive Three.js laboratories, KaTeX/MathML, source links and full-screen exploration. PIGMENT is explicitly a paper-only reconstruction while official code is not publicly verifiable.
- **Neural-network workstation** (`#/workstation/conv2d`): 50 configurable forward-computation module definitions, complete tensor values, parameters, intermediate operators and output dependency traces. Includes MobileNet V1/V2/V3 functional units, Transformer layer stacks, and recurrent cells unfolded over time. It does not claim to load pretrained networks or implement a complete optimizer/training framework.

Workstation tensors use one uniform crystal per actual scalar. Structural folders collapse operators for navigation; entering an operator renders all its operands and results without sampling. Large configurations are rejected explicitly (65,536 scalars per tensor; 500,000 stored scalars per run). Dimensions, seed, training/eval behavior and input patterns persist in the browser. Custom JSON inputs must exactly match the configured shape. Exports preserve the computation's numerical records.

Use **编辑正文** to edit long-form copy, **导出备份** / **导入备份** to move local edits between browsers. Mathematical annotations use local KaTeX assets. Edits are stored in `localStorage`, not automatically synchronized to GitHub.

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

## Verification

```bash
npm run typecheck
npm run build
npm run verify:dist
npm run verify:model
npm run verify:crystals
npm run verify:research
npm run verify:workstation
```

`verify:workstation` runs a deterministic browser-engine fixture generator, then an independent CPU PyTorch oracle (`python` must provide PyTorch). The oracle checks complete outputs of every operator across default and altered configurations, including grouped/dilated convolutions, transpose operations, statistics, attention, recurrent gates and shape transformations. The browser uses float64; the numerical tolerance is 3e-7 for the GELU erf approximation and floating-point reduction differences. These checks validate forward arithmetic, not trained model quality.

## GitHub Pages

The workflow in `.github/workflows/deploy.yml` builds and verifies `dist/`, then a separate deployment job publishes that artifact through GitHub Pages. In **Settings → Pages → Build and deployment → Source**, select **GitHub Actions**. Do not use **Deploy from a branch → main / (root)**: the root `index.html` is a Vite source entry and cannot run directly on a static host.

The `github-pages` environment belongs to the deployment **job**, not a step. A misplaced environment makes the workflow invalid, allowing an existing branch-based Pages build to keep publishing the source tree instead.

Before publishing, run `npm run build && npm run verify:dist`. A deployed homepage must reference compiled `/assets/*.js` and `/assets/*.css`, never `/src/main.tsx`. Preview the production artifact using `npm run preview`.

## Research-content boundary

The site distinguishes paper-reported facts from interactive explanatory redraws. Canvas-rendered MRI slices, temperature readouts, and trend curves are labeled as mechanism illustrations rather than experimental measurements.
