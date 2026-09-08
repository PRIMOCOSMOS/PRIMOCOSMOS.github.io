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
- Independent home identity: PRIMOCOSMOS：SILICONDEVINE, preserving the user-supplied spelling. The approved S2 Silicon Idol direction uses a black ground, white industrial lettering, an aggressive emblem integrated with the lettering, and restrained cold motion. Visitors enter the existing literature archive from this separate home surface.
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
