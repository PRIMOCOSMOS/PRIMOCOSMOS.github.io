import { useMemo, useState } from 'react'
import { Ban, GitCompareArrows } from 'lucide-react'
import { EquationBlock, SignalKey } from './Primitives'
import { EditableText } from './EditableContent'

const MODELS = [
  { id: 'mvae', label: 'MVAE', color: '#e4cf45', fusion: 'PoE', note: '联合输入经过乘积专家形成单一后验；训练目标包含各输入集合的同视图重建。', trend: [0.55, 0.24, 0.08, 0.03, 0.02, 0.02, 0.02] },
  { id: 'mopoe', label: 'MoPoE-VAE', color: '#68d6a4', fusion: 'mixture of PoE', note: '每个非空模态子集形成一个 PoE，随后对这些子集后验求混合；训练下界包含跨视图项。', trend: [0.79, 0.57, 0.24, 0.08, 0.04, 0.03, 0.02] },
  { id: 'mmvae', label: 'MMVAE', color: '#78b8ff', fusion: 'MoE', note: '从每个单模态后验采样，并让样本解码全部模态；同视图和跨视图误差共同更新共享表示。', trend: [0.86, 0.68, 0.35, 0.1, 0.05, 0.03, 0.02] },
  { id: 'plus', label: 'MMVAE++', color: '#c896ff', fusion: 'MoE + stop-gradient', note: '同视图分支使用私有变量与停止梯度的共享变量；跨视图误差负责更新共享后验。', trend: [0.86, 0.8, 0.76, 0.74, 0.7, 0.64, 0.52] },
]

const FEATURE_COUNTS = [20, 50, 100, 200, 400, 700, 1000]

function trendPoint(count: number, value: number) {
  const x = 54 + (Math.log10(count / 20) / Math.log10(50)) * 336
  const y = 34 + (1 - value) * 170
  return `${x},${y}`
}

export function MMVAEPlusExplorer() {
  const [activeModel, setActiveModel] = useState('plus')
  const [privateFeatures, setPrivateFeatures] = useState(500)
  const model = MODELS.find((entry) => entry.id === activeModel) ?? MODELS[3]

  const regime = useMemo(() => {
    if (privateFeatures < 80) return { label: 'shared signal visible', tone: 'stable' }
    if (privateFeatures < 350) return { label: 'private variation competing', tone: 'warning' }
    return { label: activeModel === 'plus' ? 'shared path protected' : 'shared path overwhelmed', tone: activeModel === 'plus' ? 'stable' : 'danger' }
  }, [activeModel, privateFeatures])

  const routeState = (source: number, target: number) => {
    if (activeModel === 'mvae') return source === target ? 'active' : 'inactive'
    if (activeModel === 'plus' && source === target) return 'stopped'
    return 'active'
  }

  return (
    <div className="architecture-console mmvae-console">
      <div className="console-toolbar">
        <div>
          <span className="control-label">比较推断目标</span>
          <div className="segmented-control model-tabs" role="group" aria-label="选择多模态 VAE 模型">
            {MODELS.map((entry) => <button key={entry.id} className={activeModel === entry.id ? 'is-active' : ''} type="button" onClick={() => setActiveModel(entry.id)} aria-pressed={activeModel === entry.id} style={{ '--control-color': entry.color } as React.CSSProperties}>{entry.label}</button>)}
          </div>
        </div>
        <div className={`regime-status regime-status--${regime.tone}`} aria-live="polite"><span>LATENT STATUS</span><strong>{regime.label}</strong></div>
      </div>

      <div className="mmvae-stage mmvae-stage--stacked">
        <div className="latent-diagram diagram-frame--dense" role="region" aria-label="MMVAE++ 完整潜变量与梯度路径图，可横向滚动" tabIndex={0}>
          <svg className="architecture-svg architecture-svg--mmvae" viewBox="0 0 1460 760" role="img" aria-labelledby="mmvae-title mmvae-desc">
            <title id="mmvae-title">共享与私有潜变量、多模态融合、四条重建路径和 MMVAE++ 梯度规则</title>
            <desc id="mmvae-desc">两个模态分别经过私有和共享后验头。私有变量进入本模态解码器，共享变量进入两个解码器。目标矩阵列出四条预测路径；MMVAE++ 在两条同视图路径上停止共享后验梯度。</desc>
            <defs><marker id="mv-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor" /></marker></defs>

            <rect x="20" y="18" width="1010" height="480" rx="3" className="diagram-zone" />
            <text x="38" y="47" className="svg-label">A · STRUCTURED MULTIMODAL LATENT MODEL</text>
            <text x="38" y="69" className="svg-measure">z = [ zprivate-1, zshared, zprivate-2 ] · selected inference: {model.fusion}</text>

            {[{ y: 104, id: 1, color: '#ff8c66' }, { y: 320, id: 2, color: '#c6a6ff' }].map((modality) => (
              <g key={modality.id} className={`mv-encoder-row mv-encoder-row--${modality.id}`}>
                <rect x="38" y={modality.y} width="108" height="74" rx="2" style={{ stroke: modality.color }} /><text x="72" y={modality.y + 31} className="svg-copy">x{modality.id}</text><text x="54" y={modality.y + 54} className="svg-tiny">modality {modality.id}</text>
                <path d={`M146 ${modality.y + 37}H178`} markerEnd="url(#mv-arrow)" />
                <rect x="178" y={modality.y} width="156" height="74" rx="2" /><text x="194" y={modality.y + 26} className="svg-copy">encoder φ{modality.id}</text><text x="194" y={modality.y + 48} className="svg-micro">shared feature trunk</text><text x="194" y={modality.y + 64} className="svg-measure">parameters θφ{modality.id}</text>
                <path d={`M334 ${modality.y + 25}H368`} markerEnd="url(#mv-arrow)" /><path d={`M334 ${modality.y + 51}H352V${modality.y + 93}H368`} markerEnd="url(#mv-arrow)" />
                <rect x="368" y={modality.y - 6} width="164" height="62" rx="2" /><text x="382" y={modality.y + 17} className="svg-tiny">private posterior head</text><text x="382" y={modality.y + 38} className="svg-copy">qφ{modality.id}(zᵖʳ{modality.id}|x{modality.id})</text><text x="382" y={modality.y + 51} className="svg-micro">μpr{modality.id}, σpr{modality.id} → sample</text>
                <rect x="368" y={modality.y + 68} width="164" height="62" rx="2" /><text x="382" y={modality.y + 91} className="svg-tiny">shared posterior head</text><text x="382" y={modality.y + 112} className="svg-copy">qφ{modality.id}(zˢʰ|x{modality.id})</text><text x="382" y={modality.y + 125} className="svg-micro">μsh{modality.id}, σsh{modality.id} → sample</text>
              </g>
            ))}

            <g className="fusion-module">
              <path d="M532 203H574" markerEnd="url(#mv-arrow)" /><path d="M532 419H554V239H574" markerEnd="url(#mv-arrow)" />
              <rect x="574" y="174" width="174" height="104" rx="2" className="svg-accent-box" /><text x="590" y="198" className="svg-copy">shared inference</text><text x="590" y="221" className="svg-formula">{model.fusion}</text><text x="590" y="243" className="svg-micro">qφ(zshared | xI)</text><text x="590" y="262" className="svg-measure">I ⊆ {'{'}1,2{'}'}</text>
              <rect x="574" y="302" width="174" height="72" rx="2" /><text x="590" y="326" className="svg-tiny">factorized priors</text><text x="590" y="349" className="svg-micro">p(zpr1)p(zshared)p(zpr2)</text>
            </g>

            <g className="mv-decoders">
              <rect x="798" y="104" width="200" height="116" rx="2" /><text x="814" y="129" className="svg-copy">decoder pθ1(x1|·)</text><text x="814" y="153" className="svg-micro">same: [zpr1, zshared(x1)]</text><text x="814" y="174" className="svg-micro">cross: [prior zpr1, zshared(x2)]</text><rect x="814" y="188" width="166" height="20" rx="2" /><text x="823" y="202" className="svg-tiny">likelihood → x̂1</text>
              <rect x="798" y="320" width="200" height="116" rx="2" /><text x="814" y="345" className="svg-copy">decoder pθ2(x2|·)</text><text x="814" y="369" className="svg-micro">same: [zpr2, zshared(x2)]</text><text x="814" y="390" className="svg-micro">cross: [prior zpr2, zshared(x1)]</text><rect x="814" y="404" width="166" height="20" rx="2" /><text x="823" y="418" className="svg-tiny">likelihood → x̂2</text>
              <path d="M532 129H798" markerEnd="url(#mv-arrow)" /><path d="M748 213H776V177H798" markerEnd="url(#mv-arrow)" /><path d="M532 345H798" markerEnd="url(#mv-arrow)" /><path d="M748 239H776V393H798" markerEnd="url(#mv-arrow)" />
              <path d="M532 203C692 203 662 393 798 393" className="cross-prediction" markerEnd="url(#mv-arrow)" /><path d="M532 419C692 419 662 177 798 177" className="cross-prediction" markerEnd="url(#mv-arrow)" />
            </g>

            <rect x="1060" y="18" width="380" height="480" rx="3" className="diagram-zone" />
            <text x="1078" y="47" className="svg-label">B · OBJECTIVE ROUTE MATRIX</text>
            <text x="1078" y="70" className="svg-measure">source posterior → target likelihood</text>
            <text x="1178" y="112" className="svg-tiny">target x1</text><text x="1304" y="112" className="svg-tiny">target x2</text>
            {[1, 2].map((source) => (
              <g key={source}>
                <text x="1080" y={151 + (source - 1) * 112} className="svg-copy">qφ{source}(z|x{source})</text>
                {[1, 2].map((target) => {
                  const state = routeState(source, target)
                  const x = 1170 + (target - 1) * 126
                  const y = 126 + (source - 1) * 112
                  return (
                    <g key={target} className={`objective-cell objective-cell--${state}`}>
                      <rect x={x} y={y} width="112" height="82" rx="2" />
                      <text x={x + 12} y={y + 23} className="svg-tiny">L{'{'}{target}{'}'}←{'{'}{source}{'}'}</text>
                      <text x={x + 12} y={y + 46} className="svg-micro">{source === target ? 'same-view' : 'cross-view'}</text>
                      <text x={x + 12} y={y + 66} className="svg-measure">{state === 'active' ? 'gradient ON' : state === 'stopped' ? 'shared sg(·)' : 'term absent'}</text>
                    </g>
                  )
                })}
              </g>
            ))}
            <rect x="1078" y="368" width="344" height="104" rx="2" className="svg-accent-box" />
            <text x="1094" y="393" className="svg-copy">MMVAE++ gradient allocation</text>
            <text x="1094" y="418" className="svg-tiny">same-view: zpr + sg(zshared) → x̂m</text>
            <text x="1094" y="441" className="svg-tiny">cross-view: zshared → x̂m′ · gradient ON</text>
            <text x="1094" y="462" className="svg-micro">optional label head: p(y|zshared), weight β</text>

            <rect x="20" y="526" width="1420" height="208" rx="3" className="diagram-zone" />
            <text x="38" y="555" className="svg-label">C · BACKPROPAGATION MAP</text>
            <rect x="38" y="580" width="302" height="118" rx="2" /><text x="54" y="604" className="svg-copy">same-view reconstruction</text><text x="54" y="628" className="svg-tiny">x1→x̂1 / x2→x̂2</text><text x="54" y="652" className="svg-micro">updates private posterior heads + decoders</text><text x="54" y="677" className="svg-measure">shared posterior: {activeModel === 'plus' ? 'STOP-GRADIENT' : activeModel === 'mvae' || activeModel === 'mmvae' || activeModel === 'mopoe' ? 'GRADIENT ON' : '—'}</text>
            <path d="M340 639H374" markerEnd="url(#mv-arrow)" />
            <rect x="374" y="580" width="302" height="118" rx="2" /><text x="390" y="604" className="svg-copy">cross-view prediction</text><text x="390" y="628" className="svg-tiny">x1→x̂2 / x2→x̂1</text><text x="390" y="652" className="svg-micro">updates shared posterior heads + decoders</text><text x="390" y="677" className="svg-measure">shared semantics follow common variation</text>
            <path d="M676 639H710" markerEnd="url(#mv-arrow)" />
            <rect x="710" y="580" width="302" height="118" rx="2" /><text x="726" y="604" className="svg-copy">KL regularization</text><text x="726" y="628" className="svg-tiny">KL private + KL shared</text><text x="726" y="652" className="svg-micro">posterior → factorized unit Gaussian priors</text><text x="726" y="677" className="svg-measure">included in each applicable ELBO</text>
            <path d="M1012 639H1046" markerEnd="url(#mv-arrow)" />
            <rect x="1046" y="580" width="376" height="118" rx="2" className="svg-accent-box" /><text x="1062" y="604" className="svg-copy">resulting shared channel</text><text x="1062" y="628" className="svg-tiny">cross-view predictive factors remain available</text><text x="1062" y="652" className="svg-micro">private variation is represented by zpr1 and zpr2</text><text x="1062" y="677" className="svg-measure">selected model: {model.label}</text>
          </svg>
          <EditableText className="diagram-explanation" textKey={`mmvae-model-note-${model.id}`}>{model.note}</EditableText>
        </div>

        <div className="trend-console">
          <div className="trend-heading"><GitCompareArrows size={17} aria-hidden="true" /><span>私有维度增加时的跨视图预测趋势</span></div>
          <svg viewBox="0 0 440 250" role="img" aria-label="论文图 1B 的定性趋势重绘，不用于读取精确数值">
            <line x1="54" y1="204" x2="404" y2="204" className="chart-axis" /><line x1="54" y1="30" x2="54" y2="204" className="chart-axis" />
            <text x="12" y="43" className="chart-label">HIGH</text><text x="18" y="205" className="chart-label">LOW</text><text x="214" y="234" className="chart-label">PRIVATE FEATURES →</text>
            {MODELS.map((entry) => <polyline key={entry.id} points={FEATURE_COUNTS.map((count, index) => trendPoint(count, entry.trend[index])).join(' ')} fill="none" stroke={entry.color} strokeWidth={entry.id === activeModel ? 4 : 1.3} opacity={entry.id === activeModel ? 1 : 0.25} />)}
            <line x1={54 + (Math.log10(privateFeatures / 20) / Math.log10(50)) * 336} y1="30" x2={54 + (Math.log10(privateFeatures / 20) / Math.log10(50)) * 336} y2="204" className="chart-cursor" />
          </svg>
          <EditableText className="micro-note" textKey="mmvae-trend-note">曲线重绘论文 Fig. 1B 的相对趋势。实验数值请以论文图表为准。</EditableText>
          <div className="range-control"><div className="range-heading"><label htmlFor="private-features">私有特征数量</label><output htmlFor="private-features">{privateFeatures}</output></div><input id="private-features" type="range" min="20" max="1000" step="10" value={privateFeatures} onChange={(event) => setPrivateFeatures(Number(event.target.value))} /></div>
        </div>
      </div>

      <div className="console-lower"><div className="warning-line"><Ban size={16} aria-hidden="true" /><EditableText as="span" textKey="mmvae-gradient-note">共享潜变量的语义由允许更新它的误差路径决定。</EditableText></div><div className="signal-legend"><SignalKey color="#78b8ff">共享因素</SignalKey><SignalKey color="#ff8c66">模态 1 私有因素</SignalKey><SignalKey color="#c6a6ff">模态 2 私有因素</SignalKey></div></div>

      <div className="equation-stack">
        <EquationBlock id="mmvae-fusion" title="三类多模态后验融合" formula="MVAE: q(z|x1,x2)∝p(z)∏m qφm(z|xm)/p(z);  MMVAE: q(z|x1,x2)=1/M Σm qφm(z|xm);  MoPoE: q(z|x)=1/|P(X)| ΣX′⊆X qPoE(z|X′)" defaultOpen>MVAE 使用乘积专家集中多个模态的证据。MMVAE 混合单模态后验。MoPoE-VAE 对所有非空输入子集的乘积专家再求混合。</EquationBlock>
        <EquationBlock id="mmvae-objectives" title="同视图与跨视图下界" formula="MVAE: L{1,2}←{1,2}+L{1}←{1}+L{2}←{2};   MoPoE adds L{1}←{2}+L{2}←{1};   MMVAE: (1/M)ΣmΣm′ Eqφm log pθm′(xm′|z)" defaultOpen>跨视图项要求从一个模态的后验预测另一个模态。该误差直接衡量共享潜变量保留了多少跨模态可预测信息。</EquationBlock>
        <EquationBlock id="mmvae-stop-gradient" title="MMVAE++ 的共享梯度分配" formula="m=m′: Ezpr~q Ezsh~sg(q) log pθm(xm|zsh,zpr);    m≠m′: Ezsh~q log pθm′(xm′|zsh)" defaultOpen>同视图重建使用私有变量和共享变量，其中共享后验经过 stop-gradient。跨视图预测使用共享变量并更新共享后验，使其集中表达两个模态之间的关联因素。</EquationBlock>
      </div>
    </div>
  )
}
