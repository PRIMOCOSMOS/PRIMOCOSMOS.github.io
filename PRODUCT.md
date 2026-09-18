# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Confirmed: React + Vite + TypeScript, built as a static site for GitHub Pages. The implementation must not require a server-side runtime.

## Users

- Primary: prospective supervisors, research collaborators, and peers evaluating PRIMOCOSMOS's research interests and depth of understanding.
- Secondary: graduate students and technical readers studying multimodal representation learning, qMRI, and medical image reconstruction.

## Product Purpose

Create a personal homepage and expandable research-reading system. Its first research collection explains three papers related to multimodal latent-variable modeling and MRI, allowing visitors to understand and interact with the proposed neural-network architectures rather than only reading prose summaries.

Success means a first-time visitor can quickly identify the owner's research direction, then accurately explain each paper's core mechanism, mathematical idea, evidence, and limitation after exploring the page.

## Positioning

The site turns literature review into an explorable model laboratory: readers can manipulate missing modalities, shared/private factors, content/style paths, and iterative reconstruction states while the accompanying explanation and equations remain synchronized.

## Operating Context

- Public personal site hosted through GitHub Pages.
- Research content is read on desktop and mobile and may be used during discussions with supervisors or peers.
- Source material consists of research PDFs, official paper pages, figures, equations, experiments, and cited repositories.
- Initial collection:
  - Unified Cross-Modal Medical Image Synthesis with Hierarchical Mixture of Product-of-Experts.
  - A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction based on Content/Style Modeling.
  - Disentangling Shared and Private Latent Factors in Multimodal Variational Autoencoders.

## Capabilities and Constraints

- Interactive, diagram-first explanations of all three neural-network architectures.
- Preserve the mathematical derivations that are necessary to understand the mechanism, paired with concise plain-language explanations.
- Clearly separate paper claims, interpretive explanation, experimental evidence, and limitations.
- Static deployment only; all interactions must run in the browser.
- Responsive and keyboard-accessible, with reduced-motion support and a readable non-enhanced fallback.
- The public repository could not be fetched from this environment because the GitHub connection was reset. The local workspace currently contains no incumbent site implementation, so the build proceeds as a greenfield static site.
- Personal biography, portrait, CV, contact destinations, and additional research projects are explicitly undecided and must not be fabricated.

## Brand Commitments

- Public identity: PRIMOCOSMOS.
- Independent home identity: PRIMOCOSMOS：SILICONDEVINE, preserving the user-supplied spelling. The approved S2 Silicon Idol direction uses a black ground, white industrial lettering, a curved emblem composed with the title, with clear space between symbol and letters, and restrained cold motion. Visitors enter the existing literature archive from this separate home surface.
- Science-fiction, geek, and avant-garde character.
- Overdrive mode: ambitious browser-native interaction and expressive art direction are required, while the research remains legible and credible.
- The visual language must be specific to multimodal medical imaging and latent-variable research, not a generic neon technology template.

## Evidence on Hand

- `D:\WUDAN_RESEARCH_1\1-s2.0-S136184152600229X-main.pdf` — PnP-CoSMo paper.
- `D:\WUDAN_RESEARCH_1\martens24a.pdf` — MMVAE++ paper.
- `D:\WUDAN_RESEARCH_1\TPAMI_Dorent_HAL.pdf` — MMHVAE paper.
- Official web records and paper pages were cross-checked for titles, authors, publication details, abstracts, and open-source links.
- No verified portrait, biography, institutional affiliation, awards, publications by the site owner, or contact details were supplied.

## Product Principles

1. Mechanism before decoration: every expressive interaction must clarify information flow, inference, or reconstruction.
2. Scientific honesty: surface assumptions, failure modes, and evidence boundaries next to the claim they qualify.
3. Learn by intervention: let readers change inputs and observe consequences instead of presenting passive diagrams.
4. One shared conceptual spine: connect the three papers through shared/private information, modality fusion, and missing-data inference.
5. Progressive depth: provide a fast conceptual read, then formulas and implementation detail without forcing either audience into the other's path.

## Accessibility & Inclusion

The experience must remain understandable without color alone, support keyboard navigation, preserve text contrast, work at narrow mobile widths, and respect `prefers-reduced-motion`.

## Research archive and neural-network workstation (2026-09-14)
The home now provides two real destinations: the classified research archive and an independent operating workstation. Existing papers belong to 编码器; SSDiff, METSC, PIGMENT belong to 课题核心. The archive must grow through searchable metadata, not fixed-width paper tabs. PIGMENT has a paper-backed visualization until an official implementation is publicly verifiable.
The workstation follows the user's TensorSpace-style crystal visualization direction: configurable actual tensor dimensions, complete numerical intermediates, precise dependencies, same-scene recursive exploration, full-screen calculation and exportable input/parameter/output records. It is a forward-computation inspector for initialized networks, not a claim of trained-model inference or a full training framework. MobileNet module units and Transformer layer stacks are named accordingly. Browser arithmetic is verified against PyTorch, with numerical tolerance disclosed.

### Workstation arithmetic revision (2026-09-18)
The four-root searchable catalog contains 61 executable modules: the previous 50 plus five CUT and six Diffusion modules. Every scope, including the overview, presents the actual expanded arithmetic dependency graph. Linear exposes all products, sums and bias; activations, normalization and generative operators expose concrete intermediate stages. Convolution and reductions stream exact per-output contributions and partial sums. Dense connections highlight the current output and term according to the execution trace. Full tensor coordinates remain available, with exact factor rectangles for leading planes and long vectors.

The stage operator index starts collapsed and pages six entries at a time. Graphs deeper than 24 levels enable camera following of the current operator while retaining the full graph; mathematical step navigation and full-graph fitting remain available. Explanatory prose is persistently editable from the top toolbar, independently of live equations and values. Verification covers 61 models, 124 configurations and 3,149 PyTorch-compared outputs; default configurations expand to 4,241 mathematical stages. This revision is local and has not been deployed.

CUT follows the no-antialias convolution path. The diffusion U-Net is a configured, initialized two-scale time-conditioned network. PatchNCE and LSGAN are independent forward objectives; DDPM/DDIM accept externally initialized epsilon predictions. These modules do not constitute a complete trainer or a trained checkpoint. Official source links and these boundaries are recorded in `src/components/workstation/generative.ts`.
