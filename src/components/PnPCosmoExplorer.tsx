import { useEffect, useMemo, useState } from 'react'
import { Pause, Play, RefreshCcw } from 'lucide-react'
import { EquationBlock, SignalKey } from './Primitives'
import { EditableText } from './EditableContent'
import { SyntheticMRISlice } from './SyntheticMRISlice'

export function PnPCosmoExplorer() {
  const [iteration, setIteration] = useState(0)
  const [running, setRunning] = useState(false)
  const [contentRefinement, setContentRefinement] = useState(true)
  const [centerFraction, setCenterFraction] = useState(8)
  const reducedMotion = useMemo(() => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches, [])

  useEffect(() => {
    if (!running || reducedMotion) return
    const timer = window.setInterval(() => setIteration((current) => (current >= 8 ? 0 : current + 1)), 920)
    return () => window.clearInterval(timer)
  }, [running, reducedMotion])

  const phase = iteration === 0 ? -1 : (iteration - 1) % 3
  const aliasing = Math.max(0.06, 0.94 - iteration * (contentRefinement ? 0.105 : 0.078))
  const residual = Math.max(0.07, 0.91 / (iteration + 1) + (contentRefinement ? 0 : 0.13))
  const styleConfidence = Math.min(96, Math.round(44 + centerFraction * 3.1))

  return (
    <div className="architecture-console pnp-console">
      <div className="console-toolbar pnp-toolbar">
        <div className="transport" role="group" aria-label="迭代播放控制">
          <button className="transport__primary" type="button" onClick={() => reducedMotion ? setIteration((current) => (current >= 8 ? 0 : current + 1)) : setRunning((current) => !current)}>
            {running && !reducedMotion ? <Pause size={15} /> : <Play size={15} />}{reducedMotion ? '推进一步' : running ? '暂停迭代' : '运行迭代'}
          </button>
          <button type="button" onClick={() => { setIteration(0); setRunning(false) }} aria-label="重置迭代"><RefreshCcw size={15} /></button>
        </div>
        <div className="iteration-readout"><span>ITERATION</span><strong>{String(iteration).padStart(2, '0')}</strong><span>/ 08</span></div>
        <label className="switch-control"><input type="checkbox" checked={contentRefinement} onChange={(event) => setContentRefinement(event.target.checked)} /><span aria-hidden="true" />Content refinement</label>
      </div>

      <div className="pnp-stage pnp-stage--stacked">
        <div className="slice-rail" aria-label="代码生成的重建状态示意">
          <SyntheticMRISlice label="T1W reference" contrast={0.44} aliasing={0} />
          <SyntheticMRISlice label="T2W estimate" contrast={0.78} aliasing={aliasing} />
          <div className="residual-readout"><span>k-space residual</span><strong>{residual.toFixed(2)}</strong><small>交互机制示意</small></div>
        </div>

        <div className="pnp-diagram-wrap diagram-frame--dense" role="region" aria-label="PnP-CoSMo 训练与在线重建完整架构图，可横向滚动" tabIndex={0}>
          <svg className="architecture-svg architecture-svg--pnp" viewBox="0 0 1500 820" role="img" aria-labelledby="pnp-title pnp-desc">
            <title id="pnp-title">PnP-CoSMo 内容风格模型训练和三步在线重建</title>
            <desc id="pnp-desc">上半部显示两个域的内容编码器、风格编码器、AdaIN 解码器、判别器与无配对和配对训练损失。下半部显示反投影初始化、内容一致、数据一致与内容修正循环。</desc>
            <defs><marker id="pnp-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor" /></marker></defs>

            <rect x="20" y="18" width="1460" height="360" rx="3" className="diagram-zone" />
            <text x="38" y="47" className="svg-label">A · OFFLINE IMAGE-DOMAIN CONTENT / STYLE MODEL</text>
            <text x="38" y="70" className="svg-measure">training data: paired or unpaired fully sampled images · no k-space samples required</text>

            {[{ y: 100, id: '1', label: 'X1 · reference contrast', color: '#78b8ff' }, { y: 235, id: '2', label: 'X2 · target contrast', color: '#d8ff45' }].map((domain) => (
              <g key={domain.id} className="pnp-domain-row">
                <rect x="38" y={domain.y} width="130" height="72" rx="2" style={{ stroke: domain.color }} /><text x="52" y={domain.y + 26} className="svg-copy">x{domain.id}</text><text x="52" y={domain.y + 50} className="svg-tiny">{domain.label}</text>
                <path d={`M168 ${domain.y + 24}H196`} markerEnd="url(#pnp-arrow)" /><path d={`M168 ${domain.y + 52}H196`} markerEnd="url(#pnp-arrow)" />
                <rect x="196" y={domain.y - 1} width="220" height="56" rx="2" /><text x="210" y={domain.y + 16} className="svg-copy">Eᶜ{domain.id} · content encoder</text><text x="210" y={domain.y + 34} className="svg-micro">input conv → stride conv↓ → residual blocks</text><text x="210" y={domain.y + 49} className="svg-measure">spatial code c{domain.id} · capacity JM</text>
                <rect x="196" y={domain.y + 63} width="220" height="56" rx="2" /><text x="210" y={domain.y + 80} className="svg-copy">Eˢ{domain.id} · style encoder</text><text x="210" y={domain.y + 98} className="svg-micro">conv↓ → adaptive average pool → FC</text><text x="210" y={domain.y + 113} className="svg-measure">global code s{domain.id}</text>
                <path d={`M416 ${domain.y + 27}H462`} markerEnd="url(#pnp-arrow)" /><path d={`M416 ${domain.y + 91}H442V${domain.y + 45}H462`} markerEnd="url(#pnp-arrow)" />
                <rect x="462" y={domain.y} width="214" height="104" rx="2" /><text x="476" y={domain.y + 21} className="svg-copy">G{domain.id} · AdaIN decoder</text><text x="476" y={domain.y + 42} className="svg-micro">residual blocks + AdaIN(c{domain.id},s{domain.id})</text><text x="476" y={domain.y + 61} className="svg-micro">upsampling conv↑ → output conv</text><rect x="476" y={domain.y + 73} width="83" height="20" rx="2" /><text x="484" y={domain.y + 87} className="svg-micro">self x{domain.id}→x̂{domain.id}</text><rect x="568" y={domain.y + 73} width="94" height="20" rx="2" /><text x="576" y={domain.y + 87} className="svg-micro">cross c/s swap</text>
                <path d={`M676 ${domain.y + 52}H712`} markerEnd="url(#pnp-arrow)" /><rect x="712" y={domain.y + 18} width="124" height="68" rx="2" /><text x="727" y={domain.y + 43} className="svg-copy">D{domain.id}</text><text x="727" y={domain.y + 63} className="svg-tiny">domain GAN</text>
              </g>
            ))}

            <g className="cross-route">
              <path d="M416 127C440 127 438 287 462 287" markerEnd="url(#pnp-arrow)" /><text x="426" y="202" className="svg-micro">c1 + s2 → x̂2</text>
              <path d="M416 326C440 326 438 152 462 152" markerEnd="url(#pnp-arrow)" /><text x="426" y="245" className="svg-micro">c2 + s1 → x̂1</text>
            </g>

            <g className="loss-rack">
              <rect x="872" y="92" width="574" height="240" rx="2" />
              <text x="890" y="118" className="svg-copy">TRAINING OBJECTIVES</text>
              <rect x="890" y="136" width="258" height="78" rx="2" /><text x="904" y="158" className="svg-tiny">stage 1 · unpaired MUNIT</text><text x="904" y="179" className="svg-micro">LGAN + α1 Lself-image</text><text x="904" y="198" className="svg-micro">+ α2 Lself-content + α3 Lself-style</text>
              <rect x="1170" y="136" width="258" height="78" rx="2" /><text x="1184" y="158" className="svg-tiny">stage 2 · paired fine-tuning</text><text x="1184" y="179" className="svg-micro">LGAN + β1 Lself-image</text><text x="1184" y="198" className="svg-micro">+ β2 Lcross-image + β3 Lcross-content</text>
              <rect x="890" y="234" width="538" height="72" rx="2" className="svg-accent-box" /><text x="904" y="255" className="svg-tiny">gradient destinations</text><text x="904" y="277" className="svg-micro">image losses → Ec, Es, G · content alignment → Ec1, Ec2 · adversarial losses → G / D1,D2</text><text x="904" y="294" className="svg-measure">saved prior M = {'{'}Ec1,Ec2,Es1,Es2,G1,G2{'}'}</text>
            </g>

            <rect x="20" y="402" width="1460" height="390" rx="3" className="diagram-zone" />
            <text x="38" y="431" className="svg-label">B · ONLINE GUIDED RECONSTRUCTION · FIXED NETWORK M</text>
            <g className="online-inputs">
              <rect x="38" y="464" width="152" height="64" rx="2" /><text x="51" y="487" className="svg-copy">measured y</text><text x="51" y="509" className="svg-tiny">undersampled k-space</text>
              <path d="M190 496H226" markerEnd="url(#pnp-arrow)" /><rect x="226" y="464" width="118" height="64" rx="2" /><text x="246" y="487" className="svg-copy">Aᴴ y</text><text x="242" y="509" className="svg-tiny">x2⁰ init</text>
              <rect x="38" y="588" width="152" height="64" rx="2" style={{ stroke: '#78b8ff' }} /><text x="51" y="611" className="svg-copy">xref = x1</text><text x="51" y="633" className="svg-tiny">aligned reference</text>
              <path d="M190 620H226" markerEnd="url(#pnp-arrow)" /><rect x="226" y="588" width="118" height="64" rx="2" /><text x="253" y="611" className="svg-copy">Eᶜ1</text><text x="247" y="633" className="svg-tiny">c⁰ init</text>
            </g>

            <g className={phase === 0 ? 'pnp-operation online-step is-current' : 'pnp-operation online-step'}>
              <rect x="394" y="456" width="284" height="210" rx="2" /><text x="410" y="482" className="svg-copy">1 · CONTENT CONSISTENCY · CC</text>
              <rect x="410" y="506" width="108" height="54" rx="2" /><text x="425" y="528" className="svg-tiny">x2ᵏ⁻¹</text><text x="425" y="547" className="svg-micro">current target</text>
              <path d="M518 533H544" markerEnd="url(#pnp-arrow)" /><rect x="544" y="506" width="116" height="54" rx="2" /><text x="558" y="528" className="svg-tiny">Eˢ2</text><text x="558" y="547" className="svg-micro">style s2ᵏ⁻¹</text>
              <rect x="410" y="592" width="108" height="54" rx="2" /><text x="425" y="614" className="svg-tiny">cᵏ⁻¹</text><text x="425" y="633" className="svg-micro">content state</text>
              <path d="M518 619H544V574" markerEnd="url(#pnp-arrow)" /><path d="M602 560V574" markerEnd="url(#pnp-arrow)" />
              <rect x="544" y="574" width="116" height="72" rx="2" className="svg-accent-box" /><text x="558" y="597" className="svg-copy">G2 + AdaIN</text><text x="558" y="618" className="svg-micro">cᵏ⁻¹ ⊕ s2ᵏ⁻¹</text><text x="558" y="636" className="svg-tiny">z2ᵏ</text>
            </g>
            <path d="M344 496H394" markerEnd="url(#pnp-arrow)" /><path d="M344 620H394" markerEnd="url(#pnp-arrow)" />

            <g className={phase === 1 ? 'pnp-operation online-step is-current' : 'pnp-operation online-step'}>
              <rect x="724" y="456" width="308" height="210" rx="2" /><text x="740" y="482" className="svg-copy">2 · DATA CONSISTENCY · DC</text>
              <rect x="740" y="510" width="90" height="48" rx="2" /><text x="755" y="539" className="svg-tiny">z2ᵏ</text><path d="M830 534H852" markerEnd="url(#pnp-arrow)" />
              <rect x="852" y="510" width="72" height="48" rx="2" /><text x="873" y="539" className="svg-copy">A</text><path d="M924 534H946" markerEnd="url(#pnp-arrow)" />
              <rect x="946" y="510" width="68" height="48" rx="2" /><text x="959" y="529" className="svg-tiny">− y</text><text x="956" y="547" className="svg-micro">residual</text>
              <path d="M980 558V580H914" markerEnd="url(#pnp-arrow)" /><rect x="826" y="580" width="88" height="48" rx="2" /><text x="852" y="610" className="svg-copy">Aᴴ</text><path d="M826 604H804" markerEnd="url(#pnp-arrow)" />
              <rect x="740" y="580" width="64" height="48" rx="2" /><text x="753" y="599" className="svg-tiny">−η·</text><text x="751" y="617" className="svg-micro">step</text>
              <path d="M772 628V646H914" markerEnd="url(#pnp-arrow)" /><rect x="914" y="580" width="100" height="66" rx="2" className="svg-accent-box" /><text x="937" y="608" className="svg-copy">x2ᵏ</text><text x="928" y="629" className="svg-micro">data-consistent</text>
            </g>
            <path d="M678 560H724" markerEnd="url(#pnp-arrow)" />

            <g className={phase === 2 && contentRefinement ? 'pnp-operation online-step is-current' : 'pnp-operation online-step'}>
              <rect x="1078" y="456" width="354" height="210" rx="2" /><text x="1094" y="482" className="svg-copy">3 · CONTENT REFINEMENT · CR</text>
              <rect x="1094" y="506" width="118" height="48" rx="2" /><text x="1110" y="526" className="svg-tiny">Eˢ2(x2ᵏ)</text><text x="1110" y="543" className="svg-micro">updated style</text>
              <rect x="1094" y="578" width="118" height="48" rx="2" /><text x="1110" y="598" className="svg-tiny">cᵏ⁻¹</text><text x="1110" y="615" className="svg-micro">optimizable code</text>
              <path d="M1212 530H1234V562" markerEnd="url(#pnp-arrow)" /><path d="M1212 602H1234V578" markerEnd="url(#pnp-arrow)" /><rect x="1234" y="542" width="82" height="56" rx="2" /><text x="1255" y="566" className="svg-copy">G2</text><text x="1248" y="584" className="svg-micro">synthesis</text>
              <path d="M1316 570H1336" markerEnd="url(#pnp-arrow)" /><rect x="1336" y="542" width="78" height="56" rx="2" /><text x="1357" y="565" className="svg-tiny">A · −y</text><text x="1352" y="584" className="svg-micro">loss</text>
              <path d="M1375 598V632H1153V626" markerEnd="url(#pnp-arrow)" /><text x="1220" y="648" className="svg-micro">−γ ∇c ||A G2(c,s2) − y||² → cᵏ</text>
              {!contentRefinement && <g className="cr-disabled"><path d="M1088 466L1422 654M1422 466L1088 654" /><text x="1190" y="690" className="svg-tiny">CR BYPASSED</text></g>}
            </g>
            <path d="M1032 614H1078" markerEnd="url(#pnp-arrow)" />
            <path d="M1254 666V744H536V666" className="iteration-loop" markerEnd="url(#pnp-arrow)" /><text x="790" y="733" className="svg-label">k ← k + 1 · feed x2ᵏ and cᵏ into next pass</text>
          </svg>
          <div className="operation-status" aria-live="polite"><span className={phase === 0 ? 'is-live' : ''}>01 内容一致 CC</span><span className={phase === 1 ? 'is-live' : ''}>02 数据一致 DC</span><span className={phase === 2 && contentRefinement ? 'is-live' : ''}>03 内容修正 CR</span></div>
        </div>
      </div>

      <div className="console-lower">
        <div className="range-control">
          <div className="range-heading"><label htmlFor="center-fraction">k-space 中心采样占比</label><output htmlFor="center-fraction">{centerFraction}%</output></div>
          <input id="center-fraction" type="range" min="4" max="16" step="4" value={centerFraction} onChange={(event) => setCenterFraction(Number(event.target.value))} />
          <div className="dual-readout"><span>style 估计置信度 <b>{styleConfidence}</b></span><span>content 来源 <b>REF</b></span></div>
          <EditableText className="micro-note" textKey="pnp-sampling-note">交互示意：低频中心采样越充分，目标对比 style 的估计通常越稳定。数值只表示机制趋势。</EditableText>
        </div>
        <div className="signal-legend"><SignalKey color="#78b8ff">共享内容</SignalKey><SignalKey color="#d8ff45">测量约束</SignalKey><SignalKey color="#ff5c35">内容修正梯度</SignalKey></div>
      </div>

      <div className="equation-stack">
        <EquationBlock id="pnp-training" title="内容—风格模型的两阶段目标" formula="Lᴹᵁᴺᴵᵀ=Lᴳᴬᴺ+α₁Lself-img+α₂Lself-content+α₃Lself-style;   Lᴾᶠᵀ=Lᴳᴬᴺ+β₁Lself-img+β₂Lcross-img+β₃Lcross-content" defaultOpen>无配对阶段通过自重建、内容回归、风格回归和域判别学习两个图像域。少量配对数据可继续约束双向交叉合成和内容编码的一致性。</EquationBlock>
        <EquationBlock id="pnp-capacity" title="共享内容的空间容量" formula="Jᴹ = (Hc · Wc) / (Hx · Wx)" defaultOpen>Jᴹ 表示内容编码相对输入图像保留的空间采样密度。较高容量保存更多解剖细节，同时也可能把对比度特征带入共享内容；论文通过该量分析内容与风格的分工。</EquationBlock>
        <EquationBlock id="pnp-iteration" title="重建迭代：CC、DC 与 CR" formula="z₂ᵏ=G₂(cᵏ⁻¹,Eˢ₂(x₂ᵏ⁻¹));  x₂ᵏ=z₂ᵏ−ηAᴴ(Az₂ᵏ−y);  cᵏ=cᵏ⁻¹−γ∇c||AG₂(cᵏ⁻¹,Eˢ₂(x₂ᵏ))−y||²₂" defaultOpen>CC 组合参考内容与当前目标风格。DC 沿测量残差的反投影方向更新图像。CR 使用同一测量误差更新内容编码，逐步修正参考图像与目标解剖之间的差异。</EquationBlock>
      </div>
    </div>
  )
}
