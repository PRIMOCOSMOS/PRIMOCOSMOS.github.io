# PRIMOCOSMOS — Multimodal Signal Lab

Independent Silicon Idol homepage with two destinations:

- **Research archive** (`#/research/<paper>`): six searchable notes in 编码器 (MMHVAE, PnP-CoSMo, MMVAE++) and 课题核心 (SSDiff, METSC, PIGMENT). Shared recursive Three.js laboratories, KaTeX/MathML, source links and full-screen exploration. PIGMENT is explicitly a paper-only reconstruction while official code is not publicly verifiable.
- **Neural-network workstation** (`#/workstation/conv2d`): 67 configurable forward-computation module definitions, complete tensor values, parameters, intermediate operators and output dependency traces. Includes MobileNet V1/V2/V3 functional units, Transformer layer stacks, recurrent cells unfolded over time, five CUT, six Diffusion and six VAE modules. It does not claim to load pretrained networks or implement a complete optimizer/training framework.

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
npm run verify:regions
npm run verify:receptive-field
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

Convolution and reductions retain concrete streamed terms and partial sums, persistent inter-stage links and the main trunk. Auxiliary parameters sit near consumers and zero padding docks to input. The catalog follows four roots → task family → module, with six entries per collapsed stage index page. Graphs deeper than 80 levels support camera following with the whole graph retained. The upright camera uses a more level angle and supports full-graph fitting and mathematical navigation; graph-depth canvas height ranges from 580px to 1100px. A quick MLP/Linear rail above the canvas and in fullscreen exposes Batch, input/output features and MLP hidden width/layer count; sidebar controls remain. The top editing toolbar still provides persistent explanatory prose editing independent of live values.

Tensor planes follow their last two dimensions; preceding coordinates address a tiled plane array. Leading planes and long vectors use exact factor rectangles (48 becomes 6 × 8), preserving all coordinates without empty corners. Vector neuron layers use larger uniform .58 cells and a 3.6 horizontal pitch factor; other tensor fields use uniform .36 cells. Batch and exact coordinates remain preserved. Cyan marks nonnegative values, amber negative values and pale white the selection; signed edges use a workstation-specific optional style. Stratification follows topological depth, then separates role, operation, shape and parameter consumer into independent planes without changing real dependency edges. Unlike steps or data never share a plane. Independent branches move nearer their consumers; Linear projections stay immediately after the encoder so the μ/log-variance heads retain their proper position. The implementation adapts ideas from TensorSpace's [LayerLocator](https://github.com/tensorspace-team/tensorspace/blob/master/src/utils/LayerLocator.js), [FeatureMap](https://github.com/tensorspace-team/tensorspace/blob/master/src/elements/FeatureMap.js), and [Conv2d](https://github.com/tensorspace-team/tensorspace/blob/master/src/layer/intermediate/Conv2d.js), inspected on 2026-09-14, using the project's current Three.js runtime and crystal primitives.

The label selector has three modes: hover, show all and hide all. Hover reveals only the exact pointed tensor/module, not every item in its group; labels and leader lines follow the same visibility state. KaTeX stage formulas and full formulas remain available below. Camera controls include cursor-targeted wheel zoom, plus/minus, pan/rotate, double-click tensor framing, reset and fullscreen.

Six VAE modules add dual Gaussian heads (vae-head), an MLP encoder (vae-mlp), a teaching convolutional encoder (vae-conv: stride-2 convolution plus global average pooling), reparameterization (vae-reparameter), diagonal-Gaussian KL (vae-kl: latent-axis sum retaining Batch) and two-expert Gaussian fusion with a standard-normal prior (vae-poe). The convolutional example is not an official full-network reproduction. Box–Muller noise is reproducible but does not claim the same random sequence as torch.randn_like. Scalar exponential, square and negative-log operations use the existing numerical-transfer curves.

CUT uses the no-antialias convolution path. Diffusion includes a configured initialized two-scale time-conditioned U-Net; DDPM/DDIM consume externally initialized epsilon predictions. PatchNCE and LSGAN are independent forward objectives, not a complete trainer or checkpoint reproduction. Official implementation references are linked in `src/components/workstation/generative.ts`.

Computation exports use JSON schema version 2: every full tensor is stored once; operator inputs and outputs reference tensor IDs. Formulas and operator settings remain in the record. Per-output dependencies are evaluated interactively rather than eagerly duplicating every trace in the download. Configured weights are initialized examples, not trained checkpoint weights.

The workstation also adapts the weighted-connection and activation-color teaching approach of [3Blue1Brown’s neural-network lesson](https://www.3blue1brown.com/lessons/neural-networks/). Its geometry is implemented in current Three.js; actual values remain in the numerical engine. Boundary tensors remain included in camera fitting; label visibility follows the selected mode.

Current local verification covers 67 modules, 136 configurations and 3,261 operator comparisons with PyTorch, with maximum error 8.79e-8. Additional end-to-end checks cover torch.distributions KL, Gaussian product-of-experts and reparameterization. Earlier MLP edge-count and activation-transfer checks remain applicable. This revision has not been deployed.


## Functional regions and source compositions

Transformer scenes keep all source tensors and functional topology. Only the current operator allocates its connection fabric and runs its transfer animation. The automatic sequence visits source functions in order; hovering a function freezes that sequence and runs its own local arithmetic clock, and leaving resumes the sequence. Self-attention/FFN and decoder cross-attention are independent regions. A standalone attention function further exposes projection, score normalization, and value/output aggregation. Region routes also appear in the keyboard-accessible tree. Continuous 3D slabs cover inter-layer spaces; a minimum 28 px picking corridor supports narrow full-model views without merging vertical tiers. MLP retains every baseline-visible weight.

The six literature viewers compile their existing audited source entries into `SourceModule` compositions and render them through `ModuleAssembly`. The overview displays source-shaped crystal coordinate windows at the original source nodes, with real dependency edges and shortcut paths. Hover activates the reusable operator composition in that node; entering it exposes its functional children as a vertical tensor trunk. Static source geometry is batched, and inactive numerical visuals are disposed. Hover selects a visible child or cycles the children of a containing function. Navigation resets hover ownership so a previous source ID cannot stall the new local scene. MMHVAE clears the overview's active numerical visual on entry to a detail view.

Source dimensions and mathematical demonstrations remain distinct: static crystals represent contiguous source-coordinate windows, while atomic animations use the declared deterministic numerical examples. They do not execute trained paper checkpoints or stand in for full-resolution activation tensors. Source identities, repeated block instances, shared-weight notes, mathematical parameterizations and exact dimension labels remain in the existing source-audited catalogs. PIGMENT remains paper-derived where no official implementation is confirmed.

`verify:regions` checks seven Transformer families, continuous region boundaries (including overlapping source extents), lazy allocation/disposal, 135 paper compositions with 904 source-linked functions, and all MMHVAE top-level compositions. `verify:receptive-field` checks actual traced convolution neighborhoods, including an independently constructed grouped/dilated/batched 54-voxel Conv3D stencil and true 3D tensor coordinates. Desktop and 390 px browser checks cover hover locking, local cycles, leaving, in-scene descent, fullscreen and all six paper routes. These changes are local, not deployed.


### Literature tensor-window rendering (2026-09-22)

The paper tensor renderer no longer falls back to operator icons. Known dimensions produce equal-sided glass cells at explicit source coordinates; purely symbolic shapes retain an unresolved extent. Large fields use contiguous, pageable windows, with the full shape and current C/H/W (or token/feature) index ranges disclosed. These are not downsampled full tensors or fabricated activation values. A numeric window control is available in all six paper viewers, including fullscreen. MMHVAE internal channel-only descriptions inherit their spatial dimensions from their source level; SE channel vectors remain vectors. Static value-neutral cells carry modality colors; animated numerical examples use signed-value colors.

Only the active functional region receives a holographic title in hover mode. All/none modes remain, and hover text no longer applies block formatting to KaTeX's internal spans. Detail views retain the selected outer context rather than the entire surrounding network. MMHVAE now enables cursor-directed wheel/pinch zoom and pan; plus/minus controls support 0.05–100 zoom instead of the previous 2.4 ceiling. Convolution example views retain all actual weighted dependencies of their displayed numerical calculation, with smooth shared-workstation connection-fabric highlights.

This remains a source-architecture and mathematical teaching viewer, not full-resolution checkpoint inference: a full-source coordinate window is distinct from the independently labeled small numerical operator example. PIGMENT's unavailable implementation details remain unspecified.


文献 3D 场景已共用工作站渲染器。大型网络采用性能分层：功能总览 → 完整数学计算，块内点击聚焦；静态张量合批，计算连接按功能区播放。`verify:source-execution` 检查六篇文献的源码坐标、关键 MMHVAE 通道/卷积参数、切片边界及全部公式。文献的符号计算不等于加载预训练权重的数值推理。
