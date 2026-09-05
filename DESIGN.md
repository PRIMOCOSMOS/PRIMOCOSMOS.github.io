---
name: PRIMOCOSMOS Multimodal Signal Lab
description: A deep-space personal cover opening into an interactive multimodal research notebook.
colors:
  event-horizon-black: "#05080b"
  orbital-navy: "#07111f"
  notebook-black: "#05090d"
  method-deck-navy: "#071018"
  architecture-night: "#071118"
  diagram-bay-black: "#050b10"
  active-tab-slate: "#0c1820"
  warm-schematic-paper: "#f2e9d8"
  phosphor-tinted-ink: "#eef2df"
  muted-sensor-sage: "#a8b8b3"
  faint-calibration-teal: "#738889"
  observed-signal-lime: "#d8ff45"
  shared-latent-blue: "#78b8ff"
  correction-orange: "#ff5c35"
  private-factor-violet: "#c896ff"
  modality-one-coral: "#ff8c66"
  modality-two-lavender: "#c6a6ff"
  comparison-amber: "#e4cf45"
  comparison-green: "#68d6a4"
  cool-telemetry-hairline: "rgba(203, 227, 213, 0.18)"
  active-lime-hairline: "rgba(216, 255, 69, 0.34)"
typography:
  display:
    fontFamily: '"Tektur Variable", "Noto Sans SC", sans-serif'
    fontSize: "clamp(4rem, 8.2vw, 6rem)"
    fontWeight: 625
    lineHeight: 0.84
    letterSpacing: "-0.035em"
  headline:
    fontFamily: '"Tektur Variable", "Noto Sans SC", sans-serif'
    fontSize: "clamp(2rem, 4vw, 4rem)"
    fontWeight: 600
    lineHeight: 1.08
    letterSpacing: "-0.03em"
  title:
    fontFamily: '"Tektur Variable", "Noto Sans SC", sans-serif'
    fontSize: "clamp(1.45rem, 2.4vw, 2.25rem)"
    fontWeight: 580
    lineHeight: 1.2
  body:
    fontFamily: '"Noto Sans SC", "Microsoft YaHei UI", "PingFang SC", sans-serif'
    fontSize: "0.84rem"
    fontWeight: 400
    lineHeight: 1.88
  label:
    fontFamily: '"Tektur Variable", "Noto Sans SC", sans-serif'
    fontSize: "0.66rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0.12em"
  mono:
    fontFamily: '"Cascadia Code", "SFMono-Regular", monospace'
    fontSize: "0.76rem"
    fontWeight: 400
    lineHeight: 1.75
  math:
    fontFamily: '"Cambria Math", "STIX Two Math", "Latin Modern Math", "Times New Roman", serif'
    fontSize: "0.76rem"
    fontWeight: 400
    lineHeight: 1.75
rounded:
  node-tight: "2px"
  node: "3px"
  pill: "999px"
spacing:
  xs: "0.5rem"
  sm: "0.75rem"
  md: "1rem"
  lg: "1.5rem"
  xl: "2rem"
  section: "clamp(4.75rem, 9vw, 9rem)"
components:
  primary-action:
    backgroundColor: "{colors.observed-signal-lime}"
    textColor: "{colors.event-horizon-black}"
    padding: "0.8rem 1.15rem"
    height: "2.9rem"
  primary-action-hover:
    backgroundColor: "{colors.warm-schematic-paper}"
    textColor: "{colors.event-horizon-black}"
  control-pill:
    backgroundColor: "transparent"
    textColor: "{colors.muted-sensor-sage}"
    rounded: "{rounded.pill}"
    padding: "0.52rem 0.78rem"
    height: "2.75rem"
  control-pill-active:
    backgroundColor: "{colors.observed-signal-lime}"
    textColor: "{colors.event-horizon-black}"
    rounded: "{rounded.pill}"
    padding: "0.52rem 0.78rem"
    height: "2.75rem"
  paper-tab:
    backgroundColor: "{colors.method-deck-navy}"
    textColor: "{colors.muted-sensor-sage}"
    padding: "1.2rem"
    height: "10.5rem"
  paper-tab-active:
    backgroundColor: "{colors.active-tab-slate}"
    textColor: "{colors.warm-schematic-paper}"
    padding: "1.2rem"
    height: "10.5rem"
  source-link:
    backgroundColor: "transparent"
    textColor: "{colors.muted-sensor-sage}"
    rounded: "{rounded.pill}"
    padding: "0.52rem 0.78rem"
    height: "2.75rem"
  navigation-link:
    backgroundColor: "transparent"
    textColor: "{colors.muted-sensor-sage}"
    padding: "1.55rem 0 1.4rem"
  architecture-console:
    backgroundColor: "{colors.architecture-night}"
    textColor: "{colors.phosphor-tinted-ink}"
    padding: "clamp(1rem, 2.6vw, 2.25rem)"
  equation-disclosure:
    backgroundColor: "transparent"
    textColor: "{colors.warm-schematic-paper}"
    padding: "1rem 0"
  telemetry-panel:
    backgroundColor: "rgba(5, 12, 17, 0.92)"
    textColor: "{colors.warm-schematic-paper}"
    padding: "clamp(1.25rem, 2.4vw, 2rem)"
  signal-key:
    backgroundColor: "transparent"
    textColor: "{colors.muted-sensor-sage}"
    padding: "0"
---

# Design System: PRIMOCOSMOS Multimodal Signal Lab

## Overview

**Creative North Star: "Deep-Space Telemetry Array"**

PRIMOCOSMOS is a dark science-fiction personal field station rather than a conventional portfolio. An independent, full-viewport cover introduces the identity through a live signal field, clipped orbital framing, a compact field-index console, and a three-cell archive status rail. Entering Research changes the pacing from cinematic identity to a rigorous, tabbed academic notebook without leaving the same material world.

The notebook is intentionally dense and operational. Three mutually exclusive paper tabs expose complete notes in a stable Abstract → Method → Evaluation sequence. The Method band expands beyond the reading column and gives each paper a browser-native model laboratory: large labeled SVG schematics, controls that alter observed modalities or gradient routes, generated MRI slices, trend plots, formula disclosures, and plain-language status readouts. Expressive details must explain identity, hierarchy, state, information flow, or evidence—not decorate empty space.

**Key Characteristics:**

- Separate cinematic personal cover and dense research notebook, joined by one phosphor-lime entry action.
- Matte near-black and navy tonal decks, clipped instrument housings, one-pixel calibration rails, and almost no soft elevation.
- Semantic signal colors: lime for observed/active/measurement state, blue for shared/reference structure, orange for correction or blocked gradients, and violet/coral/lavender for distinct latent factors.
- One complete paper mounted at a time through a keyboard-operable, three-item tab bar.
- Exhaustive interactive architectures whose controls, routes, captions, readouts, and equations stay synchronized.
- Responsive reflow around fixed scientific topology: every dense model canvas stays horizontally scrollable and keyboard focusable.

## Colors

The palette resembles a scientific instrument deck: near-black tonal layers carry the reading experience, warm neutrals carry prose, and chromatic signals have fixed scientific meanings.

### Primary

- **Observed-Signal Lime:** Used for the sole cover CTA, active paper rail, selected controls, measured inputs, live reconstruction phases, numeric outputs, selection highlight, and all focus outlines.

### Secondary

- **Shared-Latent Blue:** Used for shared representations, reference content, formulas, architecture borders, hover strokes, and structural information.

### Tertiary

- **Correction Orange:** Identifies content-refinement gradients, stop-gradient states, warnings, danger regimes, and other scientifically corrective or skeptical states.
- **Private-Factor Violet:** Identifies private-factor comparison routes and the emphasized MMVAE++ series.
- **Modality-One Coral / Modality-Two Lavender:** Keep the two modality-specific channels distinguishable inside multimodal diagrams and legends without borrowing the meanings of lime or blue.
- **Comparison Amber / Comparison Green:** Reserved for MVAE and MoPoE-VAE comparison curves; these are analytical series colors, not general accents.

### Neutral

- **Event-Horizon Black:** Page canvas, dark action text, and the base behind all instrument decks.
- **Orbital Navy:** Full-viewport personal-cover field.
- **Notebook Black:** Research notebook ground, slightly lifted from the page void.
- **Method-Deck Navy:** Paper tabs, full-bleed Method bands, and the narrow-screen scrollbar track.
- **Architecture Night / Diagram-Bay Black:** Outer architecture housings and their inset SVG/Canvas bays.
- **Active-Tab Slate:** Hovered and selected paper-tab field.
- **Warm Schematic Paper / Phosphor-Tinted Ink:** Highest-emphasis headings and default foreground.
- **Muted Sensor Sage / Faint Calibration Teal:** Explanatory prose, inactive controls, metadata, axes, and micro labels.
- **Cool Telemetry Hairline / Active Lime Hairline:** Default separation and stronger active structure.

### Named Rules

**The Signal-Is-Data Rule.** Lime means observed, active, or measurement-anchored; blue means shared, referenced, or informational; orange means corrective, stopped, or risky; violet, coral, and lavender separate latent factors. Every chromatic state also needs a label, position, icon, stroke, or shape cue.

**The Comparison-Only Rule.** Amber and green belong to comparative plot series. Do not promote them into navigation, calls to action, focus, or general decoration.

**The Phosphor Rarity Rule.** Lime is the sharpest voice and stays sparse outside active states, focus, measurements, and the primary entry action.

## Typography

**Display Font:** Tektur Variable (bundled through `@fontsource-variable/tektur`, with Noto Sans SC and sans-serif fallbacks)  
**Body Font:** Noto Sans SC (with Microsoft YaHei UI, PingFang SC, and sans-serif fallbacks)  
**Label/Mono Font:** Tektur Variable for telemetry labels; Cascadia Code (with SFMono-Regular and monospace fallbacks) for dense SVG microcopy and edit-state messaging  
**Math Font:** Cambria Math, followed by STIX Two Math, Latin Modern Math, Times New Roman, and serif fallbacks

**Character:** The title stack supplies an engineered orbital-instrument voice; the Chinese system-sans stack keeps the long scientific reading calm and legible. Cascadia Code keeps diagram annotations compact. Formula fields use conventional mathematical letterforms so operators, indices, and Greek symbols retain an academic reading texture.

### Hierarchy

- **Display** (625, fluid 4–6rem, 0.84): The two-line `FIELD NOTES / FROM THE EDGE` cover transmission. At the narrowest breakpoint it shifts to a fluid 2.8–4.1rem with 0.9 line height.
- **Headline** (570–600, fluid 1.7–4rem, 1.02–1.08): Research title and paper titles; always balanced and tightly tracked.
- **Title** (580, fluid 1.45–2.25rem, approximately 1.2): Abstract, Method, and Evaluation block headings. Compact instrument subheads fall to 0.9–1rem.
- **Body** (400, usually 0.83–0.84rem, 1.7–1.88): Translations, method explanation, evidence, and caveats. Emphasis comes from space and color rather than increased weight.
- **Label** (400–580, 0.58–0.72rem, tracking up to 0.14em): Uppercase section codes, statuses, coordinates, axes, and control metadata.
- **Mono** (400, 0.61–0.76rem, 1.5–1.75): Edit-state messaging and dense SVG implementation annotations.
- **Math** (400, 0.68–0.76rem, 1.75): Scroll-safe mathematical expressions with normal style, zero tracking, and tabular lining numerals.

### Named Rules

**The Four-Channel Type Rule.** Tektur carries identity, headings, measurements, and machine voice; the Chinese system-sans stack carries explanation and evidence; Cascadia Code carries micro notation and editing telemetry; the conventional math stack carries equations.

**The Readable Science Rule.** Sustained prose keeps generous 1.7–1.88 line height and restrained widths even when the surrounding console is dense.

## Layout

The site alternates full-bleed telemetry fields with centered reading shells. Standard content uses `min(84rem, 88vw)`; the personal cover uses `min(90rem, 86vw)`; the full-bleed Method band centers a wider `min(92rem, 92vw)` working surface. Section spacing follows a fluid 4.75–9rem vertical rhythm. One-pixel rails, numbered block headings, and 0.5–2rem internal spacing organize high information density.

The cover is a full-viewport two-column identity/console composition with a fixed 7.9rem three-cell index along the bottom. The notebook begins with a thesis/explanation split, then three equal-width paper tabs and one mounted `tabpanel`. Each paper repeats a stable reading sequence. The Method section extends beyond the standard shell; abstract and paper headers use asymmetric two-column layouts, and evaluation uses a three-column evidence ledger.

Responsive changes are explicit. At 1080px wide diagram/trend companions rebalance. At 980px the cover, research introduction, paper title, and abstract stack; the comparison trend also becomes one column. At 840px shared console-lower areas and equation pairs stack and the fixed header hides its center links. At 720px the paper tablist becomes a horizontal row of three 15rem columns, Method prose and evidence become single-column, and the compact cover console becomes a three-cell ledger. At 560px action rows become full width, segmented controls become two columns, and the Method console deliberately bleeds slightly toward the viewport edges.

All three final architecture SVGs use a `92rem` minimum width inside focusable `overflow-x: auto` regions. Preserve that canvas, node order, label scale, and left-to-right graph logic at every viewport; scroll the diagram rather than compressing or rearranging it.

**The Topology-First Responsive Rule.** Reflow prose, metadata, controls, and evidence freely, but preserve the model graph itself through a keyboard-focusable horizontal viewport.

**The One-Paper Rule.** Keep exactly one tabpanel mounted. The tab row is orientation-stable—three columns on wide screens and one horizontal scroller on narrow screens—so comparison order never changes.

## Elevation & Depth

This system has no box-shadow vocabulary. Depth is flat and structural: the page void supports slightly lighter cover, notebook, Method, console, and diagram surfaces; one-pixel hairlines and clipped silhouettes express containment. The fixed header is the only translucent surface, using a near-black field with 12px backdrop blur so content may pass behind it without reducing navigation legibility.

### Named Rules

**The Tonal-Deck Rule.** Use controlled near-black steps, strokes, clipping, and inset fields; do not introduce drop shadows, luminous card halos, or floating white surfaces.

**The One-Glass Exception Rule.** Backdrop blur belongs to the fixed header only, never to paper tabs, notes, telemetry panels, or model consoles.

## Shapes

Structural surfaces are rectilinear instrument housings with opposing clipped corners: 28px on the large cover frame, 18px on the cover console, 20px on architecture consoles, and 10px on the primary action. Paper tabs, note blocks, evidence ledgers, and equation disclosures remain square and rail-driven.

Pills (`999px`) are reserved for compact controls and outbound source links, each at least 2.75rem high. SVG modules use restrained 2px node corners and 3px zone corners. Circles, diamonds, and hexagons represent orbit paths, state indicators, fusion operators, and latent cores; they are semantic instrument geometry, not a decorative pattern.

**The Hard-Housing Rule.** Structural containers are square or clipped; pill geometry belongs to touch targets, not content surfaces.

## Components

### Buttons

- **Shape:** The primary entry action is a clipped plate with 10px opposing cuts; compact architecture controls are 2.75rem-high pills.
- **Primary:** Observed-Signal Lime field, Event-Horizon Black text, `0.8rem 1.15rem` padding, and a 2.9rem minimum height.
- **Hover / Focus:** Primary hover shifts to Warm Schematic Paper. All interactive elements use a 2px lime `:focus-visible` outline with a 4px offset; custom switch tracks use a 3px offset.
- **Segmented:** Transparent, hairline, muted-text controls turn blue on hover. The selected state fills with the semantic color of the modality or model and uses dark text.
- **Transport:** PnP-CoSMo playback uses a lime primary transport button plus a separate reset button; under reduced motion, the primary label becomes an explicit single-step action.

### Editable Note Copy

The Research introduction includes a compact text-edit toolbar. `编辑正文` exposes dashed blue boundaries around explanatory copy across the cover description, section captions, translations, Method prose, diagram captions, equation explanations, and evaluation evidence. A focused field changes to a lime boundary. `保存修改` exits editing, and text is persisted per stable content key in the current browser through `localStorage`. `恢复默认` is disabled when no overrides exist and requires confirmation before clearing saved copy. Structural labels, paper metadata, controls, formulas, and SVG node names remain fixed to protect the information architecture.

### Chips

- **Style:** Signal keys are unboxed legends composed of a 1.25rem by 2px semantic line and a readable label.
- **State:** They explain modality and route color; they never become unlabeled decorative dots.

### Cards / Containers

- **Corner Style:** Square or clipped hard housings; SVG nodes alone receive 2–3px rounding.
- **Background:** Near-black tonal decks with a darker inset diagram bay.
- **Shadow Strategy:** None; use the Elevation & Depth tonal hierarchy.
- **Border:** Cool hairlines at rest; blue or lime strokes for structural and active emphasis.
- **Internal Padding:** Architecture consoles use `clamp(1rem, 2.6vw, 2.25rem)`; the cover telemetry console uses `clamp(1.25rem, 2.4vw, 2rem)`.

### Inputs / Fields

- **Range:** Full-width native range inputs have a 2.75rem hit area, lime accent color, paired label/output, and adjacent conceptual readouts.
- **Switch:** The Content Refinement switch uses a 2.4rem × 1.3rem pill track and 0.8rem indicator. Checked state changes both border and indicator to lime and moves the indicator 1.05rem.
- **Status:** Changing a field must update adjacent textual output or an `aria-live` status, not only SVG color.
- **Focus:** Hidden native switch focus transfers to its visible track; diagram-contained SVG button groups receive a 2px lime stroke.

### Navigation

The fixed header is a three-column instrument bar: brand left, two research anchors centered, online state right. Muted 0.78rem links reveal a lime underline from left to right over 220ms. Below 840px, the center links are hidden while brand, online state, and the focus-revealed skip link remain.

The paper selector is a three-item WAI-ARIA tablist. Each tab contains a code, short model name, full paper title, and venue. Active and hovered tabs use Active-Tab Slate; active selection adds a three-pixel lime rail at the top. Arrow Left/Right cycle, Home/End jump, and focus follows selection. At 720px the tablist scrolls horizontally rather than collapsing labels.

### Architecture Console

This is the signature component: a clipped blue-hairline housing with a three-pixel lime calibration tick, wrapping toolbar, 92rem focusable architecture canvas, synchronized caption/readout, controls, legend, and equation stack.

- **MMHVAE:** Toggles any non-empty subset of four observed modalities, selects a synthesis target, focuses one of seven latent levels by pointer or keyboard, and explains sampling temperature with dual readouts.
- **PnP-CoSMo:** Shows code-generated reference/estimate MRI slices, an eight-step reconstruction transport, live CC/DC/CR routing, an optional content-refinement switch, k-space-center control, and polite live phase status. The 920ms automatic cadence is replaced by explicit stepping under reduced motion.
- **MMVAE++:** Switches between MVAE, MoPoE-VAE, MMVAE, and MMVAE++; updates the objective-route matrix and backpropagation map; couples private-feature count to a labeled regime state and qualitative comparison plot.

Every SVG carries a title and description. Interactive SVG groups expose keyboard activation. Visual trends are labeled as conceptual or qualitative when they are not literal paper measurements.

### Equation Disclosure

Equations use native `details`/`summary` rows with lime-tinted rails and a rotating chevron. Open content pairs a scroll-safe conventional math-font field with an editable plain-language explanation; the pair stacks below 840px. Each paper presents three disclosures, open by default, after the interactive architecture.

## Do's and Don'ts

### Do:

- **Do** keep the cover and research notebook distinct in density while preserving one telemetry material language.
- **Do** use signal colors according to scientific meaning and repeat every color-coded state in text, geometry, iconography, stroke, or position.
- **Do** keep paper switching as an accessible tab interaction with Arrow, Home, and End behavior and exactly one mounted panel.
- **Do** preserve every dense architecture as a 92rem internal canvas inside a focusable horizontal scroller.
- **Do** keep the title stack on Tektur Variable, long Chinese reading on the system-sans stack, diagram micro notation on Cascadia Code, and formulas on the declared conventional math stack.
- **Do** keep focus visible, expose status changes in adjacent text or live regions, include SVG title/description, and make reduced-motion transport explicit.

### Don't:

- **Don't** turn the world into a generic neon gradient, glassmorphism dashboard, rounded SaaS card system, or shadow-based elevation stack.
- **Don't** use lime, blue, orange, violet, coral, lavender, amber, or green interchangeably; every one has a defined data or comparison role.
- **Don't** shrink, reorder, or redraw a scientific graph merely to fit a narrow viewport.
- **Don't** round structural consoles, paper tabs, note blocks, or evidence ledgers; reserve pills for compact controls and source links.
- **Don't** let an interactive diagram change without a readable label, caption, output, or status describing the new state.
- **Don't** invent biography, affiliation, achievements, contacts, or additional research projects to fill the personal cover.
