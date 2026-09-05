import { useMemo, useState } from 'react'
import { Eye, EyeOff, Layers3 } from 'lucide-react'
import { EquationBlock, SignalKey } from './Primitives'
import { EditableText } from './EditableContent'

const MODALITIES = [
  { id: 'ius', label: 'iUS', color: '#d8ff45' },
  { id: 't2', label: 'T2', color: '#78b8ff' },
  { id: 'cet1', label: 'ceT1', color: '#ff8c66' },
  { id: 'flair', label: 'FLAIR', color: '#c6a6ff' },
]

const LEVELS = [
  { level: 7, resolution: '1×1×256', note: 'global code' },
  { level: 6, resolution: 'spatial ×2', note: 'coarse field' },
  { level: 5, resolution: 'spatial ×2', note: 'coarse field' },
  { level: 4, resolution: 'spatial ×2', note: 'mid field' },
  { level: 3, resolution: 'spatial ×2', note: 'mid field' },
  { level: 2, resolution: 'spatial ×2', note: 'local field' },
  { level: 1, resolution: '192×192×8', note: 'pixel code' },
]

export function MMHVAEExplorer() {
  const [observed, setObserved] = useState(['ius', 't2'])
  const [target, setTarget] = useState('flair')
  const [temperature, setTemperature] = useState(0.5)
  const [focusedLevel, setFocusedLevel] = useState(4)

  const observedLabels = useMemo(
    () => MODALITIES.filter((modality) => observed.includes(modality.id)).map((modality) => modality.label),
    [observed],
  )

  const toggleObserved = (id: string) => {
    setObserved((current) => {
      if (current.includes(id)) return current.length === 1 ? current : current.filter((item) => item !== id)
      return [...current, id]
    })
  }

  const reliability = Math.round(96 - temperature * 21)
  const detail = Math.round(35 + temperature * 58)

  return (
    <div className="architecture-console mmhvae-console">
      <div className="console-toolbar" aria-label="MMHVAE 输入配置">
        <div>
          <span className="control-label">观测模态 r</span>
          <div className="segmented-control" role="group" aria-label="切换可观测模态">
            {MODALITIES.map((modality) => {
              const active = observed.includes(modality.id)
              return (
                <button key={modality.id} className={active ? 'is-active' : ''} type="button" onClick={() => toggleObserved(modality.id)} aria-pressed={active} style={{ '--control-color': modality.color } as React.CSSProperties}>
                  {active ? <Eye size={14} /> : <EyeOff size={14} />}{modality.label}
                </button>
              )
            })}
          </div>
        </div>
        <div>
          <span className="control-label">合成目标 j</span>
          <div className="segmented-control" role="group" aria-label="选择待合成模态">
            {MODALITIES.map((modality) => (
              <button key={modality.id} className={target === modality.id ? 'is-active' : ''} type="button" onClick={() => setTarget(modality.id)} aria-pressed={target === modality.id}>{modality.label}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="diagram-frame diagram-frame--dense" role="region" aria-label="MMHVAE 完整架构图，可横向滚动" tabIndex={0}>
        <svg className="architecture-svg architecture-svg--mmhvae" viewBox="0 0 1480 780" role="img" aria-labelledby="mmhvae-title mmhvae-desc">
          <title id="mmhvae-title">MMHVAE 七层层次潜变量、子集混合、乘积专家与训练目标</title>
          <desc id="mmhvae-desc">四个单模态编码器产生七层专家因子。训练时从实际观测模式中采样三个子模式，每层将条件先验和观测专家按高斯精度相乘。最细层进入各模态解码器，并由重建、KL 和对抗损失训练。</desc>
          <defs>
            <marker id="mm-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor" /></marker>
          </defs>

          <text x="24" y="28" className="svg-label">01 · UNIMODAL EXPERT BANK</text>
          <text x="342" y="28" className="svg-label">02 · SUBSET MIXTURE</text>
          <text x="550" y="28" className="svg-label">03 · HIERARCHICAL PoE · L = 7</text>
          <text x="1170" y="28" className="svg-label">04 · MODALITY DECODERS</text>

          {MODALITIES.map((modality, index) => {
            const y = 60 + index * 122
            const active = observed.includes(modality.id)
            return (
              <g key={modality.id} className={active ? 'expert-module is-observed' : 'expert-module'}>
                <rect x="24" y={y} width="286" height="100" rx="3" />
                <rect x="34" y={y + 12} width="48" height="34" rx="2" className="svg-input-tile" style={{ stroke: modality.color }} />
                <circle cx="48" cy={y + 29} r="4" style={{ fill: modality.color }} />
                <text x="58" y={y + 34} className="svg-copy">x{index + 1} · {modality.label}</text>
                <text x="96" y={y + 22} className="svg-tiny">Eφ{index + 1} · modality encoder</text>
                <rect x="96" y={y + 32} width="46" height="24" rx="2" /><text x="105" y={y + 48} className="svg-micro">CONV</text>
                <path d={`M142 ${y + 44}H156`} /><rect x="156" y={y + 32} width="54" height="24" rx="2" /><text x="164" y={y + 48} className="svg-micro">MBV2</text>
                <path d={`M210 ${y + 44}H224`} /><rect x="224" y={y + 32} width="34" height="24" rx="2" /><text x="232" y={y + 48} className="svg-micro">SE</text>
                <path d={`M258 ${y + 44}H272`} /><rect x="272" y={y + 32} width="28" height="24" rx="2" /><text x="276" y={y + 48} className="svg-micro">SW</text>
                <text x="36" y={y + 72} className="svg-micro">7 residual feature scales</text>
                <text x="166" y={y + 72} className="svg-micro">hφl^{index + 1}: μe,l / De,l</text>
                <text x="36" y={y + 92} className="svg-measure">{active ? 'included in observed mask' : 'excluded by observed mask'}</text>
                {LEVELS.map((entry, levelIndex) => <circle key={entry.level} cx={108 + levelIndex * 27} cy={y + 88} r="4" className={active ? 'expert-scale is-live' : 'expert-scale'} />)}
                <path d={`M310 ${y + 50}H340`} markerEnd="url(#mm-arrow)" />
              </g>
            )
          })}

          <g className="subset-sampler">
            <rect x="340" y="60" width="180" height="466" rx="3" />
            <text x="357" y="88" className="svg-copy">mask r → submask r′</text>
            <text x="357" y="110" className="svg-tiny">Sᵣ = all non-empty subsets</text>
            <text x="357" y="131" className="svg-tiny">training: sample T = 3</text>
            {[0, 1, 2].map((sample) => (
              <g key={sample}>
                <rect x="357" y={154 + sample * 78} width="146" height="58" rx="2" />
                <text x="369" y={175 + sample * 78} className="svg-micro">subpattern r′{sample + 1}</text>
                {MODALITIES.map((modality, index) => <circle key={modality.id} cx={374 + index * 27} cy={194 + sample * 78} r="6" style={{ fill: observed.includes(modality.id) && ((index + sample) % 3 !== 2 || observed.length === 1) ? modality.color : '#17212d' }} />)}
              </g>
            ))}
            <rect x="357" y="402" width="146" height="92" rx="2" className="svg-accent-box" />
            <text x="369" y="425" className="svg-tiny">mixture posterior</text><text x="369" y="450" className="svg-formula">Σ αᵣ′ qφ(z|xᵒᵣ′)</text><text x="369" y="476" className="svg-micro">αᵣ′ = 1 / |Sᵣ|</text>
            <path d="M520 293H548" markerEnd="url(#mm-arrow)" />
          </g>

          <g className="hierarchy-shell">
            <rect x="548" y="60" width="594" height="466" rx="3" />
            {LEVELS.map((entry, index) => {
              const y = 80 + index * 62
              const isFocused = focusedLevel === entry.level
              return (
                <g key={entry.level} className={isFocused ? 'hierarchy-level is-focused' : 'hierarchy-level'} onClick={() => setFocusedLevel(entry.level)} role="button" tabIndex={0} aria-label={`查看 z${entry.level} 层`} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') setFocusedLevel(entry.level) }}>
                  <rect x="560" y={y} width="570" height="50" rx="2" />
                  <text x="572" y={y + 20} className="svg-copy">z{entry.level}</text><text x="572" y={y + 38} className="svg-measure">{entry.resolution}</text>
                  <rect x="654" y={y + 9} width="104" height="32" rx="2" /><text x="663" y={y + 22} className="svg-micro">conditional prior</text><text x="667" y={y + 36} className="svg-tiny">pθ{entry.level}(μp,Dp)</text>
                  <path d={`M758 ${y + 25}H782`} markerEnd="url(#mm-arrow)" />
                  <rect x="782" y={y + 9} width="118" height="32" rx="2" /><text x="791" y={y + 22} className="svg-micro">observed experts</text><text x="791" y={y + 36} className="svg-tiny">∏j∈r′ hφ{entry.level}ʲ</text>
                  <path d={`M900 ${y + 25}H924`} markerEnd="url(#mm-arrow)" /><circle cx="949" cy={y + 25} r="19" /><text x="942" y={y + 31} className="svg-copy">Π</text>
                  <path d={`M968 ${y + 25}H990`} markerEnd="url(#mm-arrow)" />
                  <rect x="990" y={y + 9} width="126" height="32" rx="2" /><text x="1001" y={y + 22} className="svg-micro">Gaussian posterior</text><text x="1002" y={y + 36} className="svg-tiny">qφ{entry.level}(μl,Dl) → sample</text>
                  <text x="1121" y={y + 31} className="svg-measure">{entry.note}</text>
                  {index < LEVELS.length - 1 && <path d={`M1050 ${y + 41}V${y + 70}H706V${y + 71}`} className="topdown-link" markerEnd="url(#mm-arrow)" />}
                </g>
              )
            })}
          </g>

          <g className="decoder-bank">
            <rect x="1170" y="60" width="286" height="466" rx="3" />
            <text x="1188" y="88" className="svg-copy">z1 → modality heads</text><path d="M1116 477H1180V113" markerEnd="url(#mm-arrow)" />
            {MODALITIES.map((modality, index) => {
              const y = 120 + index * 87
              const active = target === modality.id
              return (
                <g key={modality.id} className={active ? 'decoder-row is-target' : 'decoder-row'}>
                  <rect x="1188" y={y} width="248" height="66" rx="2" /><text x="1200" y={y + 18} className="svg-tiny">pθ{index + 1}(x{index + 1}|z1)</text>
                  <rect x="1200" y={y + 27} width="74" height="24" rx="2" /><text x="1208" y={y + 43} className="svg-micro">5× ResNet</text><path d={`M1274 ${y + 39}H1293`} markerEnd="url(#mm-arrow)" />
                  <rect x="1293" y={y + 27} width="62" height="24" rx="2" /><text x="1301" y={y + 43} className="svg-micro">likelihood</text><path d={`M1355 ${y + 39}H1374`} markerEnd="url(#mm-arrow)" />
                  <rect x="1374" y={y + 22} width="50" height="34" rx="2" style={{ stroke: modality.color }} /><text x="1381" y={y + 43} className="svg-tiny">x̂ · {modality.label}</text>
                </g>
              )
            })}
          </g>

          <g className="training-rack">
            <rect x="24" y="554" width="1432" height="196" rx="3" />
            <text x="42" y="582" className="svg-label">05 · TRAINING ROUTES · OPTIMIZE θ,φ / ψ</text>
            <rect x="42" y="604" width="318" height="118" rx="2" /><text x="58" y="627" className="svg-copy">A · reconstruction / synthesis</text><text x="58" y="650" className="svg-tiny">observed xᵒ → qφ(z|xᵒᵣ′) → pθj(xj|z1)</text><text x="58" y="674" className="svg-tiny">log pθ(xᵒᵣ | z) · known modalities only</text><text x="58" y="698" className="svg-micro">backprop → encoders Eφ + decoders pθ</text>
            <path d="M360 663H390" markerEnd="url(#mm-arrow)" />
            <rect x="390" y="604" width="296" height="118" rx="2" /><text x="406" y="627" className="svg-copy">B · hierarchical KL × 7</text><text x="406" y="650" className="svg-tiny">KL[qφl(zl|·) || pθl(zl|z&gt;l)]</text><text x="406" y="674" className="svg-tiny">weighted by λKL = 0.001</text><text x="406" y="698" className="svg-micro">regularizes every sampled latent level</text>
            <path d="M686 663H716" markerEnd="url(#mm-arrow)" />
            <rect x="716" y="604" width="350" height="118" rx="2" /><text x="732" y="627" className="svg-copy">C · modality GAN × M</text><text x="732" y="650" className="svg-tiny">x̂j → Dψj ← real-image memoryj</text><text x="732" y="674" className="svg-tiny">handles absent real xj in incomplete batches</text><text x="732" y="698" className="svg-micro">epochs 801–1000 · λGAN = 0.025</text>
            <path d="M1066 663H1096" markerEnd="url(#mm-arrow)" />
            <rect x="1096" y="604" width="340" height="118" rx="2" className="svg-accent-box" /><text x="1112" y="627" className="svg-copy">TOTAL</text><text x="1112" y="652" className="svg-formula">−ELBO + λGAN · LGAN</text><text x="1112" y="676" className="svg-tiny">epoch 1–800: ELBO / epoch 801–1000: + GAN</text><text x="1112" y="700" className="svg-micro">inference sampling temperature T = {temperature.toFixed(1)}</text>
          </g>
        </svg>
        <div className="diagram-caption"><span><Layers3 size={15} aria-hidden="true" /><EditableText as="span" textKey="mmhvae-diagram-caption">七个潜变量层均已展开；点击任一层可追踪其先验、专家与后验。</EditableText></span><span>当前层 z{focusedLevel} · 观测专家 {observedLabels.join(' + ')} · 目标 {MODALITIES.find((modality) => modality.id === target)?.label}</span></div>
      </div>

      <div className="console-lower">
        <div className="range-control">
          <div className="range-heading"><label htmlFor="temperature">推理采样温度 T</label><output htmlFor="temperature">{temperature.toFixed(1)}</output></div>
          <input id="temperature" type="range" min="0.1" max="1" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
          <div className="dual-readout" aria-label="温度变化的概念性影响"><span>结构稳定度 <b>{reliability}</b></span><span>模态细节幅度 <b>{detail}</b></span></div>
          <EditableText className="micro-note" textKey="mmhvae-temperature-note">交互曲线用于解释采样温度的作用。论文推理阶段采用 T = 0.5。</EditableText>
        </div>
        <div className="signal-legend" aria-label="图例"><SignalKey color="#d8ff45">当前观测输入</SignalKey><SignalKey color="#78b8ff">层次先验与共享结构</SignalKey><SignalKey color="#ff5c35">缺失模态生成与训练信号</SignalKey></div>
      </div>

      <div className="equation-stack">
        <EquationBlock id="mmhvae-generative" title="生成模型的七层分解" formula="pθ(z)=p(zL)∏ₗ₌₁ᴸ⁻¹pθₗ(zₗ|z₍>ₗ₎),     pθ(x|z)=∏ⱼ₌₁ᴹpθⱼ(xⱼ|z₁)" defaultOpen>粗层潜变量向细层逐级提供条件先验。z7 表示全局结构，空间分辨率沿层级递增，z1 提供各模态解码器所需的像素级描述。</EquationBlock>
        <EquationBlock id="mmhvae-mixture" title="子集混合后验与训练下界" formula="qᴹᴹᴴⱽᴬᴱφ(z|xᵒᵣ)=Σᵣ′∈Sᵣ αᵣ′qφ(z|xᵒᵣ′),     L=Eᵣ′E_q[log pθ(xᵒᵣ|z)]−ΣₗE[KL(qφₗ||pθₗ)]" defaultOpen>Sᵣ 收集当前可观测集合的非空子集。训练时抽取三个子集近似外层期望，使同一组单模态编码器覆盖不同缺失模式。</EquationBlock>
        <EquationBlock id="mmhvae-poe" title="层内高斯乘积专家" formula="Dₗ⁻¹=Dₚ,ₗ⁻¹+Σⱼ∈r′Dₑ,ₗʲ⁻¹,     μₗ=Dₗ(Dₚ,ₗ⁻¹μₚ,ₗ+Σⱼ∈r′Dₑ,ₗʲ⁻¹μₑ,ₗʲ)" defaultOpen>每个已观测模态给出均值与对角协方差。精度较高的专家对融合均值贡献更大，条件先验同时进入乘积，为缺失信息保留可采样的概率空间。</EquationBlock>
      </div>
    </div>
  )
}
