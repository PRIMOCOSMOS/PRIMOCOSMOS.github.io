# PRIMOCOSMOS — Multimodal Signal Lab

Independent Silicon Idol homepage with two destinations:

- **Research archive** (`#/research/<paper>`): six searchable notes in 编码器 (MMHVAE, PnP-CoSMo, MMVAE++) and 课题核心 (SSDiff, METSC, PIGMENT). Shared recursive Three.js laboratories, KaTeX/MathML, source links and full-screen exploration. PIGMENT is explicitly a paper-only reconstruction while official code is not publicly verifiable.
- **Neural-network workstation** (`#/workstation/conv2d`): 61 configurable forward-computation module definitions, complete tensor values, parameters, intermediate operators and output dependency traces. Includes MobileNet V1/V2/V3 functional units, Transformer layer stacks, recurrent cells unfolded over time, five CUT and six Diffusion modules. It does not claim to load pretrained networks or implement a complete optimizer/training framework.

Workstation tensors preserve every actual scalar and batch coordinate. Overview, group, layer and operator views share an execution graph that retains dense neuron layers and activation source layers; every source operator in the current scope remains present. Large contractions stream exact per-output contributions and partial sums while preserving all original tensor coordinates. Large configurations are rejected explicitly (65,536 scalars per tensor; 500,000 stored scalars per run). Dimensions, seed, training/eval behavior and input patterns persist in the browser. Custom JSON inputs must exactly match the configured shape. Exports preserve the computation's numerical records.

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

## Workstation tensor scaffold

Dense MLP input, hidden and output neuron layers remain visible with every weight connection at baseline. Linear and activation source layers are retained rather than replaced by intermediate arithmetic arrays. Each W entry maps one-to-one to an edge, numerical W remains inspectable, and bias sits beside its output. Smooth overlapping Gaussian highlights sweep neighboring outputs and terms without hiding inactive weights. Per-coordinate activation traces show f(x) in aligned lanes, actual formulas and moving crystal units using the implemented activation mathematics.

Convolution and reductions retain concrete streamed terms and partial sums, persistent inter-stage links and the main trunk. Auxiliary parameters sit near consumers and zero padding docks to input. The catalog follows four roots → task family → module, with six entries per collapsed stage index page. Graphs deeper than 24 levels support camera following with the whole graph retained. The upright camera also supports full-graph fitting and mathematical navigation. A quick MLP/Linear rail above the canvas and in fullscreen exposes Batch, input/output features and MLP hidden width/layer count; sidebar controls remain. The top editing toolbar still provides persistent explanatory prose editing independent of live values.

Tensor planes follow their last two dimensions; preceding coordinates address a tiled plane array. Leading planes and long vectors use exact factor rectangles (48 becomes 6 × 8), preserving all coordinates without empty corners. Vector neuron layers use larger uniform .58 cells and a 3.6 horizontal pitch factor; other tensor fields use uniform .36 cells. Batch and exact coordinates remain preserved. Cyan marks nonnegative values, amber negative values and pale white the selection; signed edges use a workstation-specific optional style. Parallel branches use dependency depth, and long skip edges route outside a vertical longest-dependency trunk with local spacing. The implementation adapts ideas from TensorSpace's [LayerLocator](https://github.com/tensorspace-team/tensorspace/blob/master/src/utils/LayerLocator.js), [FeatureMap](https://github.com/tensorspace-team/tensorspace/blob/master/src/elements/FeatureMap.js), and [Conv2d](https://github.com/tensorspace-team/tensorspace/blob/master/src/layer/intermediate/Conv2d.js), inspected on 2026-09-14, using the project's current Three.js runtime and crystal primitives.

Complex labels appear on hover or explicit reveal, with KaTeX stage formulas and full formulas below. Camera controls include cursor-targeted wheel zoom, plus/minus, pan/rotate, double-click tensor framing, reset and fullscreen.

CUT uses the no-antialias convolution path. Diffusion includes a configured initialized two-scale time-conditioned U-Net; DDPM/DDIM consume externally initialized epsilon predictions. PatchNCE and LSGAN are independent forward objectives, not a complete trainer or checkpoint reproduction. Official implementation references are linked in `src/components/workstation/generative.ts`.

Computation exports use JSON schema version 2: every full tensor is stored once; operator inputs and outputs reference tensor IDs. Formulas and operator settings remain in the record. Per-output dependencies are evaluated interactively rather than eagerly duplicating every trace in the download. Configured weights are initialized examples, not trained checkpoint weights.

The workstation also adapts the weighted-connection and activation-color teaching approach of [3Blue1Brown’s neural-network lesson](https://www.3blue1brown.com/lessons/neural-networks/). Its geometry is implemented in current Three.js; actual values remain in the numerical engine. Focused stages retain attached local input/output and active-operation labels.

Current local verification covers 61 modules, 124 configurations, all MLP weight counts for Batch 1/2 and ten activation transfer functions matching the implemented mathematics. Earlier verification compared 3,149 outputs with PyTorch and covered 4,241 available expanded mathematical stages; that stage count does not describe the current default visible scene. The neural-layer revision has not been deployed.
