import { EditableText, NoteEditToolbar, useEditableContent } from '../EditableContent'
import { lazy, Suspense, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { ArrowDown, ArrowUpRight, BookOpen, ChevronRight, Code2, Eye, EyeOff, Focus, Layers3, Maximize2, Minimize2, Pause, Play, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { COMMIT, LEVELS, MODALITIES, NODE_MAP, STAGES, source } from './model'
import './mmhvae.css'
import HierarchyBrowser from './HierarchyBrowser'
import { atomId, canonicalPath, graphFor, navName } from './navigation'
const OrbitScene=lazy(()=>import('./OrbitScene'))
function ShapeLegend(){
  return <div className="mm-shape-legend" aria-label="三维形态图例">
    <span><svg viewBox="0 0 28 24" aria-hidden="true"><path d="M3 7h16v13H3zM7 4h16v13M11 1h16v13"/></svg>层片 · 特征张量</span>
    <span><svg viewBox="0 0 28 24" aria-hidden="true"><path d="M1 20c7 0 7-17 13-17s6 17 13 17M1 20h26M14 2v20"/></svg>钟形 · Gaussian</span>
    <span><svg viewBox="0 0 28 24" aria-hidden="true"><ellipse cx="14" cy="12" rx="12" ry="6"/><path d="M9 12h10M14 8v8"/></svg>汇合盘 · PoE</span>
    <span><svg viewBox="0 0 28 24" aria-hidden="true"><path d="m14 2 10 10-10 10L4 12Z M4 12h20M14 2v20"/></svg>晶体 · 随机采样</span>
    <span><svg viewBox="0 0 28 24" aria-hidden="true"><path d="M3 3h22L19 20H9ZM3 3l6 17M25 3l-6 17M9 20h10"/></svg>尺度锥 · 上采样</span>
  </div>
}
function MathFormula({formula}:{formula:string}) {
  const html=useMemo(()=>katex.renderToString(formula,{displayMode:true,throwOnError:false,strict:'ignore',trust:false,output:'htmlAndMathml'}),[formula])
  return <div className="mm-math" tabIndex={0} aria-label="数学公式，可横向滚动" dangerouslySetInnerHTML={{__html:html}} />
}
const THEORY=String.raw`\begin{aligned}\Lambda_l&=\Sigma_{p,l}^{-1}+\sum_{j\in r}\Sigma_{j,l}^{-1}\\\Sigma_l&=\Lambda_l^{-1}\\\mu_l&=\Sigma_l\left(\Sigma_{p,l}^{-1}\mu_{p,l}+\sum_{j\in r}\Sigma_{j,l}^{-1}\mu_{j,l}\right)\end{aligned}`
const IMPLEMENTATION=String.raw`\begin{aligned}w_l&=s_{p,l}^{-1}+\sum_{j\in r}e^{-a_{j,l}}\\s_l&=w_l^{-1}\\\mu_l&=s_l\left(\mu_{p,l}/s_{p,l}+\sum_{j\in r}\mu_{j,l}e^{-a_{j,l}}\right)\\q_l&=\mathcal N\!\left(\mu_l,\operatorname{diag}((T s_l)^2)\right)\end{aligned}`

export function MMHVAEExplorer() {
  const { editing } = useEditableContent()
  const [observed,setObserved]=useState<string[]>(['us','t2'])
  const [target,setTarget]=useState('flair'),[selected,setSelected]=useState('poe-4'),[level,setLevel]=useState(4)
  const [isolate,setIsolate]=useState(false),[spread,setSpread]=useState(0)
  const [view,setView]=useState<'orbit'|'front'|'top'>('orbit'),[reset,setReset]=useState(0),[zoom,setZoom]=useState(1),[focus,setFocus]=useState(0)
  const [temperature,setTemperature]=useState(0.5)
  const [playing,setPlaying]=useState(()=>!window.matchMedia('(prefers-reduced-motion: reduce)').matches),[step,setStep]=useState(5)
  const [reducedMotion,setReducedMotion]=useState(()=>window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [near,setNear]=useState(false),[inView,setInView]=useState(false),[expanded,setExpanded]=useState(false)
  const [formulaTab,setFormulaTab]=useState<'code'|'theory'>('code')
  const root=useRef<HTMLDivElement>(null),sceneWrap=useRef<HTMLDivElement>(null),viewer=useRef<HTMLDivElement>(null)
  const stage=step===0?0:step===1?1:step<9?2:3
  const [location,setLocation]=useState('root'),[detailMotion,setDetailMotion]=useState(true)
  const [detailZoom,setDetailZoom]=useState(1),[fullscreenNotice,setFullscreenNotice]=useState('')
  const detailPath=canonicalPath(location)
  const anatomy=useMemo(()=>graphFor(location,observed),[location,observed])
  const detailId=anatomy?location:undefined
  const activePart=location.startsWith('atom:')?anatomy?.parts.find(p=>!p.role)?.id??null:null
  const openDetail=(id:string)=>{setLocation(id);setDetailZoom(1);setPlaying(false);const n=NODE_MAP.get(id.split('/')[0]);if(n){setSelected(n.id);setLevel(n.l)}}
  const closeDetail=()=>{setLocation('root')}
  const enterPart=(id:string)=>{const p=anatomy?.parts.find(part=>part.id===id);if(!p)return;if(p.child)openDetail(p.child);else if(!location.startsWith('atom:'))openDetail(atomId(location,id))}
  const back=()=>{const path=canonicalPath(location);openDetail(path.at(-2)??'root')}
  useEffect(()=>{const keyboard=(e:KeyboardEvent)=>{if(e.key==='Escape'&&!(e.target as HTMLElement).isContentEditable&&!expanded){e.preventDefault();back()}};window.addEventListener('keydown',keyboard);return()=>window.removeEventListener('keydown',keyboard)},[location,expanded])
  useEffect(()=>{const changed=()=>{setExpanded(document.fullscreenElement===viewer.current);setFullscreenNotice('')};document.addEventListener('fullscreenchange',changed);return()=>document.removeEventListener('fullscreenchange',changed)},[])
  const fullscreen=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await viewer.current?.requestFullscreen();}catch{setFullscreenNotice('当前浏览器限制了全屏。请在 Edge 或 Chrome 的独立标签页中使用全屏按钮。')}}
  useEffect(()=>{if(editing)setPlaying(false)},[editing])
  const select=(id:string)=>{const n=NODE_MAP.get(id)!;setSelected(id);setLevel(n.l);openDetail(id);setStep(['input','stem','encoder','down'].includes(n.kind)?0:['output','resnet','image','discriminator'].includes(n.kind)?9:n.l===7?1:9-n.l)}
  useEffect(()=>{
    const media=window.matchMedia('(prefers-reduced-motion: reduce)')
    const change=()=>{setReducedMotion(media.matches);if(media.matches)setPlaying(false)}
    media.addEventListener('change',change)
    const ob=new IntersectionObserver(([e])=>{if(e.isIntersecting)setNear(true);setInView(e.isIntersecting)},{rootMargin:'150px'});ob.observe(sceneWrap.current!)
    return()=>{media.removeEventListener('change',change);ob.disconnect()}
  },[])
  useEffect(()=>{if(!playing||!inView)return;const timer=window.setInterval(()=>{if(!document.hidden)setStep(s=>(s+1)%10)},3400);return()=>window.clearInterval(timer)},[playing,inView])
  useEffect(()=>{
    if(!playing)return
    if(step===0){setSelected(`${observed[0]}-encoder-1`);setLevel(1)}
    else if(step===1){setSelected('poe-7');setLevel(7)}
    else if(step<9){setSelected(`poe-${9-step}`);setLevel(9-step)}
    else{setSelected(`${target}-output`);setLevel(1)}
  },[step,playing,observed,target])
  const toggle=(id:string)=>setObserved(prev=>prev.includes(id)?prev.length===1?prev:prev.filter(m=>m!==id):[...prev,id])
  const jumpStage=(index:number)=>{setPlaying(false);setStep([0,1,5,9][index]);select(index===0?`${observed[0]}-encoder-1`:index===1?'poe-7':index===2?`poe-${level}`:`${target}-output`)}
  const gaussianPath=(t:number)=>Array.from({length:121},(_,i)=>{const x=i/120*6-3;return`${i===0?'M':'L'}${12+i*1.9},${100-78*Math.exp(-x*x/(2*t*t))}`}).join(' ')
  const showInspector=()=>root.current?.querySelector('.mm-browser')?.scrollIntoView({behavior:reducedMotion?'instant':'smooth',block:'start'})
  return <div ref={root} className={`architecture-console mm-lab${expanded?' mm-lab--expanded':''}`} id="mmhvae-orbit-lab">
    <header className="mm-lab-heading"><div><h3>MMHVAE <span>轨道架构实验台</span></h3><EditableText textKey="mm-copy-c9410968f3">四个观测世界，一套层次生成空间。沿信号进入模型的每一个模块。</EditableText></div><a href={source('network/mhvae.py',18)} target="_blank" rel="noreferrer"><Code2 size={15}/> MHVAE2D <ArrowUpRight size={14}/></a></header>
    <NoteEditToolbar compact />
    <div className="mm-config">
      <fieldset><legend>观测输入 <span>至少保留一个模态</span></legend><div className="mm-modality-buttons">{MODALITIES.map(m=>{const active=observed.includes(m.id);return <button type="button" key={m.id} aria-pressed={active} aria-label={`${active?'移除':'添加'} ${m.label} 观测`} disabled={active&&observed.length===1} onClick={()=>toggle(m.id)} style={{'--modality':m.color} as CSSProperties}>{active?<Eye size={14}/>:<EyeOff size={14}/>}<span>{m.label}</span><i/></button>})}</div></fieldset>
      <div className="mm-config-bridge"><span>{observed.length} 个观测专家</span><ChevronRight size={16}/><span>7 层融合</span></div>
      <fieldset><legend>追踪输出 <span>模型同时生成四个模态</span></legend><div className="mm-output-buttons">{MODALITIES.map(m=><button type="button" key={m.id} aria-pressed={target===m.id} onClick={()=>{setTarget(m.id);select(`${m.id}-output`)}}>{m.label}</button>)}</div></fieldset>
    </div>
    <div className="mm-workbench"><div className="mm-scene-column">
      <div ref={viewer} className="mm-viewer"><div className="mm-view-toolbar"><div role="group" aria-label="相机预设">{([['orbit','轨道'],['front','正视'],['top','俯视']] as const).map(([v,label])=><button key={v} aria-pressed={view===v} onClick={()=>{closeDetail();setView(v);setReset(n=>n+1);setZoom(1)}}>{label}</button>)}</div><div>
        <button title="聚焦所选模块" aria-label="聚焦所选模块" onClick={()=>{openDetail(selected)}}><Focus size={16}/></button>
        <button title="放大" aria-label="放大模型" onClick={()=>detailId?setDetailZoom(z=>Math.min(z+.2,2.4)):setZoom(z=>Math.min(z+0.2,2.4))}><ZoomIn size={16}/></button><button title="缩小" aria-label="缩小模型" onClick={()=>detailId?setDetailZoom(z=>Math.max(z-.2,.6)):setZoom(z=>Math.max(z-0.2,0.6))}><ZoomOut size={16}/></button>
        <button title="重置视角" aria-label="重置视角" onClick={()=>{closeDetail();setView('orbit');setReset(n=>n+1);setZoom(1);setSpread(0);setIsolate(false)}}><RotateCcw size={16}/></button>
        <button title={expanded?'退出全屏':'全屏观看'} aria-label={expanded?'退出全屏':'全屏观看'} aria-pressed={expanded} onClick={fullscreen}>{expanded?<Minimize2 size={16}/>:<Maximize2 size={16}/>}</button>
      </div></div>
      <div className={`mm-scene-wrap${anatomy?' has-anatomy':''}`} ref={sceneWrap}>
        <div className="mm-scene-status"><span className={playing?'mm-live-dot is-playing':'mm-live-dot'}/>{playing?'信号流播放中':'自由探索'}<span>2D 网络 / 3D 拓扑</span></div>
        {near?<Suspense fallback={<div className="mm-loading"><Layers3 size={26}/><span>正在构建七层模型空间…</span></div>}><OrbitScene selected={selected} observed={observed} target={target} level={level} isolate={isolate} spread={spread} playing={playing&&inView&&!detailId} stage={stage} temperature={temperature} view={view} reset={reset} zoom={zoom} focus={focus} reducedMotion={reducedMotion} onSelect={select} onNavigate={openDetail} detailId={detailId} detailGraph={anatomy} activePart={activePart} detailMotion={detailMotion&&inView&&!editing} detailZoom={detailZoom} onPart={enterPart}/></Suspense>:<div className="mm-loading">三维模型将在进入视野后加载</div>}
        {anatomy&&<div className="mm-anatomy-overlay">
          <div className="mm-anatomy-nav"><button onClick={back}><RotateCcw size={14}/>返回上一级</button><div aria-label="拆解路径">{detailPath.map((id,i)=><button key={`${id}-${i}`} aria-current={i===detailPath.length-1?'location':undefined} onClick={()=>openDetail(id)}>{i>0&&<ChevronRight size={12}/>}<span>{navName(id,observed)}</span></button>)}</div><button aria-label={detailMotion?'暂停内部数据流':'播放内部数据流'} onClick={()=>setDetailMotion(v=>!v)}>{detailMotion?<Pause size={14}/>:<Play size={14}/>}</button></div>
          <div className="mm-anatomy-hint"><span>点击组件继续深入 · 拖动观察立体结构</span><span>边界外：输入来源 / 输出去向</span></div>
        </div>}
        <div className="mm-scene-foot"><span>拖动旋转 · 点击模块 · 双指平移</span><span>Encoder ↑ <i/> 生成路径 ↓</span></div>
      </div>
      {fullscreenNotice&&<p role="status" className="mm-fullscreen-notice">{fullscreenNotice}</p>}
      </div>
      <HierarchyBrowser id={location} observed={observed} onNavigate={openDetail}/>
      <div className="mm-layer-entry"><span>z{level} · 上一级样本 → 条件先验 + 残差专家 → PoE → 采样</span><button onClick={()=>openDetail(`layer-${level}`)}><Layers3 size={15}/>展开本层完整推导<ChevronRight size={14}/></button></div>
      <div className="mm-level-bar"><span><Layers3 size={15}/> 潜变量层</span><div role="group" aria-label="选择潜变量层">{LEVELS.map(v=><button key={v.l} aria-pressed={level===v.l} onClick={()=>select(`poe-${v.l}`)}>z<sub>{v.l}</sub><small>{v.size===1?'global':`${v.size}²`}</small></button>)}</div><button className="mm-isolate" aria-pressed={isolate} onClick={()=>setIsolate(v=>!v)}><Focus size={15}/>{isolate?'显示全模型':'隔离本层'}</button></div>
      <div className="mm-playback"><button className="mm-play-button" aria-label={playing?'暂停信号流':'播放信号流'} onClick={()=>{closeDetail();setPlaying(v=>!v);setSpread(0);setIsolate(false)}}>{playing?<Pause size={16}/>:<Play size={16}/>}</button><div className="mm-stage-buttons" role="group" aria-label="信号流阶段">{STAGES.map((s,i)=><button key={s.title} aria-pressed={stage===i} onClick={()=>jumpStage(i)}><span>{i+1}</span>{s.title}</button>)}</div></div>
      <EditableText textKey={`mm-stage-description-${stage}`} className="mm-stage-description" aria-live="polite">{STAGES[stage].description}</EditableText>
    </div></div>
    <div className="mm-under-scene"><div className="mm-legend">{MODALITIES.map(m=><span key={m.id}><i style={{background:m.color}}/>{m.label}</span>)}<span><i style={{background:'#ebdfb9'}}/>PoE 融合</span><span><i className="mm-legend-shared"/>同层 Q / P 共享权重</span></div><label className="mm-spread-control">塔间展开<input aria-label="调整编码塔间距" type="range" min="0" max="1" step="0.1" value={spread} onChange={e=>{setSpread(Number(e.target.value));setPlaying(false)}}/><span>{spread>0?'模块检视 · 连线暂隐':'连接视图'}</span></label></div>

    <ShapeLegend/>
    <section className="mm-directory">
      <details className="mm-shapes"><summary>查看完整张量尺寸与模块数量 <ChevronRight size={16}/></summary><div className="mm-table-scroll"><table><caption>train.py 默认配置 · base_features=16 · max_features=128 · pools=6</caption><thead><tr><th>潜变量</th><th>编码特征 C×H×W</th><th>Gaussian 参数</th><th>潜变量 C×H×W</th><th>连接</th></tr></thead><tbody>{LEVELS.map(v=><tr key={v.l}><th>z{v.l}</th><td>{v.feature} × {v.encoderSize} × {v.encoderSize}</td><td>{v.l===7?'Linear 1152 → 512':`BlockQ ${2*v.feature} → ${v.feature}`} → split</td><td>{v.channels} × {v.size} × {v.size}</td><td>{v.l===7?'FC → 128×3×3':v.l===1?'四个图像解码器':'Bilinear ×2 → Conv'}</td></tr>)}</tbody></table></div><EditableText textKey="mm-copy-d523fad9f5">独立编码塔：4 ×（1 stem + 7 BlockEncoder + 6 下采样 Conv）。共享核心：1 bottleneck_down、1 bottleneck_up、6 Upsample、6 BlockDecoder、6 BlockQ、6 可学习 prior head；z₇ 先验固定。输出：4 ×（6 ResnetBlock + 2 Conv 7×7）。</EditableText></details>
    </section>

    <section className="mm-math-section"><div className="mm-section-heading"><div><h4>从分布到采样</h4><EditableText textKey="mm-formula-intro">保留论文的数学含义，也精确标出公开实现的参数化。</EditableText></div><BookOpen size={22}/></div><div className="mm-equation-layout"><div>
      <div className="mm-math-tabs" role="group" aria-label="公式版本"><button aria-pressed={formulaTab==='code'} onClick={()=>setFormulaTab('code')}>源码 · compute_full</button><button aria-pressed={formulaTab==='theory'} onClick={()=>setFormulaTab('theory')}>理论 · Gaussian PoE</button></div>
      <MathFormula formula={formulaTab==='code'?IMPLEMENTATION:THEORY}/>
      <EditableText textKey={`mm-poe-explanation-${formulaTab}`}>{formulaTab==='code'?'a 表示 soft-clamp 后的 log-scale，s 是标准差。代码按 1/s 累加，并把 T·s 传给 torch.distributions.Normal。这里如实呈现代码，不把它写成倒数方差融合。':'Σ 表示协方差，Λ 表示精度。标准 Gaussian PoE 按逆协方差相加；对角情形即按 1/s² 加权。该式与公开代码的 1/s 加权存在差异。'}</EditableText>
      <a className="mm-inline-source" href={source('network/mhvae.py',190)} target="_blank" rel="noreferrer">核对 compute_full <ArrowUpRight size={13}/></a>
    </div><div className="mm-sampling"><div className="mm-range-title"><label htmlFor="mm-temperature">采样温度 T</label><output htmlFor="mm-temperature">{temperature.toFixed(1)}</output></div><input id="mm-temperature" type="range" min="0.1" max="1" step="0.1" value={temperature} onChange={e=>setTemperature(Number(e.target.value))}/>
      <svg viewBox="0 0 252 120" role="img" aria-label={`标准化高斯曲线示意：温度 ${temperature.toFixed(1)}，均值保持不变，标准差随温度缩放。曲线按峰值归一化。`}><path className="mm-gaussian-axis" d="M12 100H240M126 10V105"/><path className="mm-gaussian-reference" d={gaussianPath(1)}/><path className="mm-gaussian-live" d={`${gaussianPath(temperature)} L240 100 L12 100 Z`}/><text x="122" y="118">μ</text><text x="204" y="18">T = 1</text></svg>
      <MathFormula formula={String.raw`z_l=\mu_l+${temperature.toFixed(1)}\,s_l\odot\epsilon`}/><EditableText textKey="mm-copy-b710e20d5d">温度仅缩放采样标准差。曲线按峰值归一化以比较宽度，不表示实测后验或生成质量。论文推理采用 T = 0.5。</EditableText>
    </div></div>
    <details className="mm-equation-detail" open><summary>七层生成模型与子集混合后验 <ChevronRight size={16}/></summary><MathFormula formula={String.raw`\begin{aligned}p_\theta(z,x)&=p(z_7)\prod_{l=1}^{6}p_\theta(z_l\mid z_{>l})\prod_{j=1}^{4}p_{\theta_j}(x_j\mid z_1)\\q_\phi(z\mid x_r)&=\sum_{r'\in\mathcal S_r}\alpha_{r'}^{(r)}q_\phi(z\mid x_{r'})\end{aligned}`}/><EditableText textKey="mm-copy-46ffe7a5a1">混合发生在输入子集的后验分布上；每个子集内部逐层进行 PoE。论文训练将完整观测、仅 iUS、MRI 子集三类各赋 1/3 权重，再在 MRI 的非空子集中均匀分配。</EditableText></details></section>

    <section className="mm-training"><div className="mm-section-heading"><div><h4>训练时，三条路线共同约束。</h4><EditableText textKey="mm-copy-6a0ff595a2">以下对应公开 train.py；判别器只参与训练。</EditableText></div><ArrowDown size={20}/></div><div className="mm-training-routes">{[['完整观测','全部实际存在的模态 → 模型 → 重建已知模态'],['仅 iUS','iUS → 模型 → 已知模态重建 + 对抗约束'],['MRI 子集','每个非空 MRI 子集 → 模型 → 重建 + 对抗约束']].map(([title,copy],i)=><div key={title}><span>{i+1}</span><h5>{title}</h5><EditableText textKey={`mm-training-route-${i}`}>{copy}</EditableText></div>)}</div>
      <MathFormula formula={String.raw`\mathcal L=\mathcal L_{\rm img}+\lambda_{\rm KL}\frac{\sum_{b,l}(\log q_l(z_l)-\log p_l(z_l))}{N_{\rm voxels}\,M}+\lambda_{\rm adv}\mathcal L_{\rm adv}`}/>
      <div className="mm-training-notes"><div><strong>逐项对应源码</strong><EditableText textKey="mm-training-16f46892">每个 MRI 子集内执行完整观测、仅 iUS、MRI 子集三次前向。图像损失为已知模态上的平均 L1 / 2；KL 使用采样点的 log 概率差估计。权重默认在前 50 个 epoch 将 KL 升至 0.001。</EditableText></div><div><strong>对抗分支</strong><EditableText textKey="mm-training-83c1a1de">四个 PatchGAN 配合模态独立样本池。判别器训练从 epoch 790 开始，生成器对抗权重从 800 开启；CLI 默认 0.05，README 示例与论文为 0.025。</EditableText></div></div>
      <div className="mm-discriminators">{MODALITIES.map(m=><button key={m.id} onClick={()=>{select(`${m.id}-discriminator`);sceneWrap.current?.scrollIntoView({behavior:reducedMotion?'instant':'smooth',block:'start'})}}>查看 {m.label} PatchGAN <ArrowUpRight size={14}/></button>)}</div>
    </section>
    <details className="mm-source-audit" open><summary><Code2 size={17}/> 论文与公开实现：四处需要区分的细节 <ChevronRight size={16}/></summary><div className="mm-table-scroll"><table><thead><tr><th>位置</th><th>论文 §4.2</th><th>本展示采用的公开实现</th></tr></thead><tbody><tr><th>残差单元</th><td>描述为 MobileNetV2 + SE + Swish</td><td>Encoder：两次 3×3 Conv；中央 Decoder：×6 扩展 + DW 5×5 + 1×1 投影。</td></tr><tr><th>图像解码器</th><td>5 个 ResNet block</td><td>train.py / pred.py 默认 nb_finalblocks=6；本页逐个展开 6 个。</td></tr><tr><th>高斯融合</th><td>按逆协方差相加</td><td>compute_full 使用倒数 scale；Normal 第二参数是标准差。</td></tr><tr><th>子集与权重</th><td>三类输入各 1/3，MRI 子集均匀</td><td>遍历 MRI 非空子集，每次三路前向；并非所有非空观测子集均匀混合。</td></tr></tbody></table></div><div className="mm-audit-note"><EditableText textKey="mm-audit-note">张量尺寸由默认参数推导，未加载预训练权重。动画表达计算路径，不生成医学影像。</EditableText><span>展示固定于源码提交 </span><a href={`https://github.com/ReubenDo/MMHVAE/tree/${COMMIT}`} target="_blank" rel="noreferrer">{COMMIT.slice(0,7)} <ArrowUpRight size={12}/></a></div><div className="mm-audit-links"><a href={source('train.py',163)} target="_blank" rel="noreferrer">训练循环 <ArrowUpRight size={13}/></a><a href={source('train.py',505)} target="_blank" rel="noreferrer">默认配置 <ArrowUpRight size={13}/></a><a href={source('network/blocks.py',54)} target="_blank" rel="noreferrer">全部基础模块 <ArrowUpRight size={13}/></a></div></details>
  </div>
}
