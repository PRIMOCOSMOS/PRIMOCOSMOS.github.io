# MMHVAE orbital explorer

The existing MMHVAE research tab mounts `Explorer.tsx`. `OrbitScene.tsx` is loaded only near the viewport. It renders the 2D network's topology in 3D, not a 3D convolutional network. `model.ts` owns the module inventory, exact operator sequences, tensor dimensions, and commit-pinned source links. All explanatory math uses local KaTeX assets with HTML + MathML output.

## Evidence and configuration

- Paper: `D:\WUDAN_RESEARCH_1\TPAMI_Dorent_HAL.pdf`, section 4.2.
- Repository: https://github.com/ReubenDo/MMHVAE
- Audited commit: `31a988e77adc42ff786bcbed2f066be051c47c66`.
- Default training/prediction configuration: base features 16, maximum features 128, six pools, 192×192 slices, six final ResNet blocks with eight channels, residual and SE enabled.
- Encoder feature sizes (fine to coarse): 16×192², 32×96², 64×48², 128×24², 128×12², 128×6², 128×3².
- Latent sizes (coarse to fine): 256-vector, 64×6², 64×12², 64×24², 32×48², 16×96², 8×192².
- `bottleneck_down`, `bottleneck_up`, `qz`, `pz`, upsampling and central decoder weights are shared across modalities. Each encoder and final image decoder is modality-specific. Four head visualizations at one level mean four calls, not four sets of weights.

## Deliberate distinctions

1. The paper describes MobileNetV2 residual units and five image ResNet blocks; the public source has two plain convolutions in `BlockEncoder`, expanded depthwise units in `BlockDecoder`, and six image ResNet blocks by default.
2. Standard Gaussian PoE uses inverse variance. `compute_full` instead sums inverse `Normal.scale`, then uses its reciprocal as the output standard deviation. The two formula views must remain distinct.
3. `compute_marginal` also adds `1e-3` to its inverse-scale denominator. It constructs additional unimodal distributions for KL diagnostics; the joint forward sample comes from `compute_full`, which does not add that constant.
4. Training iterates over every nonempty MRI subset and performs three forwards per iteration (full observed set, iUS, MRI subset). Paper mixture weights assign 1/3 to each of the three groups, with uniform weights within the MRI group, not across all observation subsets.
5. CLI GAN weight defaults to 0.05; README example and paper use 0.025. Generator GAN term begins at epoch 800 in code; discriminator updates start at 790.

## Interaction and fallback

Drag to orbit, two-finger/right-button pan, explicit zoom buttons, keyboard arrows and +/−. Camera presets preserve vertical encoder towers. Module focus centers the camera on the selection. The module directory exposes every block and all seven Gaussian heads per modality, with full operator details in a regular HTML inspector. Missing encoders are muted, but their labels remain legible. All four output branches always exist.

Rendering pauses when offscreen or the document is hidden. Paused scenes render on demand. Reduced motion disables auto playback and moving particles. Context loss / WebGL failure offers the same HTML module directory. All Three.js resources, observers, events, and the animation frame are disposed on unmount. No weights are loaded and no medical image inference runs in the browser.

## Validation

Run `npm run build`. Browser checks cover desktop/mobile overflow, all seven layer/head choices, source detail selection, observation guards, camera controls, tensor tables, formula rendering, temperature changes, tab unmount/remount, playback, keyboard input, and WebGL fallback. Local review artifacts belong in ignored `qa/`.
