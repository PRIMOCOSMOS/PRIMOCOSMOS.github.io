import { lazy, Suspense, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { ArrowDown, ArrowUpRight, BookOpen, Box, ChevronRight, Code2, Eye, EyeOff, Focus, Layers3, Maximize2, Minimize2, Pause, Play, RotateCcw, Search, ZoomIn, ZoomOut } from 'lucide-react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { COMMIT, LEVELS, MODALITIES, NODES, NODE_MAP, STAGES, source, type ModelNode } from './model'
import './mmhvae.css'
const OrbitScene=lazy(()=>import('./OrbitScene'))
function MathFormula({formula}:{formula:string}) {
  const html=useMemo(()=>katex.renderToString(formula,{displayMode:true,throwOnError:false,strict:'ignore',trust:false,output:'htmlAndMathml'}),[formula])
  return <div className="mm-math" tabIndex={0} aria-label="数学公式，可横向滚动" dangerouslySetInnerHTML={{__html:html}} />
}
function Inspector({node,onSelect}:{node:ModelNode;onSelect:(id:string)=>void}) {
  const color=MODALITIES.find(m=>m.id===node.mod)?.color??'#78b8ff'
  return <aside className="mm-inspector" style={{'--module-color':color} as CSSProperties} aria-label="所选模块实现细节">
    <div className="mm-inspector-title"><Box size={17}/><span>模块解剖</span><span>{node.shared?'共享参数':node.mod?'模态独立':'实现细节'}</span></div>
    <h4>{node.title}</h4>
    <div className="mm-shape"><span>张量 / 通道</span><strong>{node.shape}</strong><small>空间张量按 C × H × W 标注，省略批次 B</small></div>
    <p className="mm-description">{node.description}</p>
    <ol className="mm-operators">{node.operations.map((op,i)=><li key={op}><span>{String(i+1).padStart(2,'0')}</span><p>{op}</p></li>)}</ol>
    <MathFormula formula={node.formula}/>
    {node.kind==='encoder'&&<button className="mm-inspector-next" onClick={()=>onSelect(`${node.mod}-expert-${node.l}`)}>追踪 Gaussian head <ChevronRight size={16}/></button>}
    {node.kind==='expert'&&<button className="mm-inspector-next" onClick={()=>onSelect(`poe-${node.l}`)}>追踪 PoE 融合 <ChevronRight size={16}/></button>}
    {node.kind==='poe'&&<button className="mm-inspector-next" onClick={()=>onSelect(`sample-${node.l}`)}>追踪重参数化采样 <ChevronRight size={16}/></button>}
    <a className="mm-source-link" href={source(node.file,node.line)} target="_blank" rel="noreferrer"><Code2 size={14}/><span>{node.file}:{node.line}</span><ArrowUpRight size={14}/></a>
  </aside>
}
const THEORY=String.raw`\begin{aligned}\Lambda_l&=\Sigma_{p,l}^{-1}+\sum_{j\in r}\Sigma_{j,l}^{-1}\\\Sigma_l&=\Lambda_l^{-1}\\\mu_l&=\Sigma_l\left(\Sigma_{p,l}^{-1}\mu_{p,l}+\sum_{j\in r}\Sigma_{j,l}^{-1}\mu_{j,l}\right)\end{aligned}`
const IMPLEMENTATION=String.raw`\begin{aligned}w_l&=s_{p,l}^{-1}+\sum_{j\in r}e^{-a_{j,l}}\\s_l&=w_l^{-1}\\\mu_l&=s_l\left(\mu_{p,l}/s_{p,l}+\sum_{j\in r}\mu_{j,l}e^{-a_{j,l}}\right)\\q_l&=\mathcal N\!\left(\mu_l,\operatorname{diag}((T s_l)^2)\right)\end{aligned}`

export function MMHVAEExplorer() {
  const [observed,setObserved]=useState<string[]>(['us','t2'])
  const [target,setTarget]=useState('flair'),[selected,setSelected]=useState('poe-4'),[level,setLevel]=useState(4)
  const [isolate,setIsolate]=useState(false),[spread,setSpread]=useState(0)
  const [view,setView]=useState<'orbit'|'front'|'top'>('orbit'),[reset,setReset]=useState(0),[zoom,setZoom]=useState(1),[focus,setFocus]=useState(0)
  const [temperature,setTemperature]=useState(0.5)
  const [playing,setPlaying]=useState(()=>!window.matchMedia('(prefers-reduced-motion: reduce)').matches),[step,setStep]=useState(5)
  const [reducedMotion,setReducedMotion]=useState(()=>window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [near,setNear]=useState(false),[inView,setInView]=useState(false),[expanded,setExpanded]=useState(false)
  const [formulaTab,setFormulaTab]=useState<'code'|'theory'>('code'),[search,setSearch]=useState(''),[directory,setDirectory]=useState('layer')
  const root=useRef<HTMLDivElement>(null),sceneWrap=useRef<HTMLDivElement>(null)
  const stage=step===0?0:step===1?1:step<9?2:3
  const node=NODE_MAP.get(selected)!
  const select=(id:string)=>{const n=NODE_MAP.get(id)!;setSelected(id);setLevel(n.l);setPlaying(false);setStep(['input','stem','encoder','down'].includes(n.kind)?0:['output','resnet','image','discriminator'].includes(n.kind)?9:n.l===7?1:9-n.l)}
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
  const modules=useMemo(()=>NODES.filter(n=>{
    const category=directory==='layer'?n.l===level:directory==='core'?!n.mod:directory==='outputs'?['output','resnet','image','discriminator'].includes(n.kind):n.mod===directory
    return category&&(!search||`${n.title} ${n.kind} ${n.shape}`.toLowerCase().includes(search.toLowerCase()))
  }),[directory,level,search])
  const gaussianPath=(t:number)=>Array.from({length:121},(_,i)=>{const x=i/120*6-3;return`${i===0?'M':'L'}${12+i*1.9},${100-78*Math.exp(-x*x/(2*t*t))}`}).join(' ')
  const showInspector=()=>root.current?.querySelector('.mm-inspector')?.scrollIntoView({behavior:reducedMotion?'instant':'smooth',block:'start'})
  return <div ref={root} className={`architecture-console mm-lab${expanded?' mm-lab--expanded':''}`} id="mmhvae-orbit-lab">
    <header className="mm-lab-heading"><div><h3>MMHVAE <span>轨道架构实验台</span></h3><p>四个观测世界，一套层次生成空间。沿信号进入模型的每一个模块。</p></div><a href={source('network/mhvae.py',18)} target="_blank" rel="noreferrer"><Code2 size={15}/> MHVAE2D <ArrowUpRight size={14}/></a></header>
    <div className="mm-config">
      <fieldset><legend>观测输入 <span>至少保留一个模态</span></legend><div className="mm-modality-buttons">{MODALITIES.map(m=>{const active=observed.includes(m.id);return <button type="button" key={m.id} aria-pressed={active} aria-label={`${active?'移除':'添加'} ${m.label} 观测`} disabled={active&&observed.length===1} onClick={()=>toggle(m.id)} style={{'--modality':m.color} as CSSProperties}>{active?<Eye size={14}/>:<EyeOff size={14}/>}<span>{m.label}</span><i/></button>})}</div></fieldset>
      <div className="mm-config-bridge"><span>{observed.length} 个观测专家</span><ChevronRight size={16}/><span>7 层融合</span></div>
      <fieldset><legend>追踪输出 <span>模型同时生成四个模态</span></legend><div className="mm-output-buttons">{MODALITIES.map(m=><button type="button" key={m.id} aria-pressed={target===m.id} onClick={()=>{setTarget(m.id);select(`${m.id}-output`)}}>{m.label}</button>)}</div></fieldset>
    </div>
    <div className="mm-workbench"><div className="mm-scene-column">
      <div className="mm-view-toolbar"><div role="group" aria-label="相机预设">{([['orbit','轨道'],['front','正视'],['top','俯视']] as const).map(([v,label])=><button key={v} aria-pressed={view===v} onClick={()=>{setView(v);setReset(n=>n+1);setZoom(1)}}>{label}</button>)}</div><div>
        <button title="聚焦所选模块" aria-label="聚焦所选模块" onClick={()=>{setFocus(n=>n+1);setPlaying(false);setZoom(2.4)}}><Focus size={16}/></button>
        <button title="放大" aria-label="放大模型" onClick={()=>setZoom(z=>Math.min(z+0.2,2.4))}><ZoomIn size={16}/></button><button title="缩小" aria-label="缩小模型" onClick={()=>setZoom(z=>Math.max(z-0.2,0.6))}><ZoomOut size={16}/></button>
        <button title="重置视角" aria-label="重置视角" onClick={()=>{setView('orbit');setReset(n=>n+1);setZoom(1);setSpread(0);setIsolate(false)}}><RotateCcw size={16}/></button>
        <button title={expanded?'收起工作台':'展开工作台'} aria-label={expanded?'收起工作台':'展开工作台'} aria-pressed={expanded} onClick={()=>setExpanded(v=>!v)}>{expanded?<Minimize2 size={16}/>:<Maximize2 size={16}/>}</button>
      </div></div>
      <div className="mm-scene-wrap" ref={sceneWrap}>
        <div className="mm-scene-status"><span className={playing?'mm-live-dot is-playing':'mm-live-dot'}/>{playing?'信号流播放中':'自由探索'}<span>2D 网络 / 3D 拓扑</span></div>
        {near?<Suspense fallback={<div className="mm-loading"><Layers3 size={26}/><span>正在构建七层模型空间…</span></div>}><OrbitScene selected={selected} observed={observed} target={target} level={level} isolate={isolate} spread={spread} playing={playing&&inView} stage={stage} temperature={temperature} view={view} reset={reset} zoom={zoom} focus={focus} reducedMotion={reducedMotion} onSelect={select}/></Suspense>:<div className="mm-loading">三维模型将在进入视野后加载</div>}
        <div className="mm-scene-foot"><span>拖动旋转 · 点击模块 · 双指平移</span><span>Encoder ↑ <i/> 生成路径 ↓</span></div>
      </div>
      <div className="mm-level-bar"><span><Layers3 size={15}/> 潜变量层</span><div role="group" aria-label="选择潜变量层">{LEVELS.map(v=><button key={v.l} aria-pressed={level===v.l} onClick={()=>select(`poe-${v.l}`)}>z<sub>{v.l}</sub><small>{v.size===1?'global':`${v.size}²`}</small></button>)}</div><button className="mm-isolate" aria-pressed={isolate} onClick={()=>setIsolate(v=>!v)}><Focus size={15}/>{isolate?'显示全模型':'隔离本层'}</button></div>
      <div className="mm-playback"><button className="mm-play-button" aria-label={playing?'暂停信号流':'播放信号流'} onClick={()=>{setPlaying(v=>!v);setSpread(0);setIsolate(false)}}>{playing?<Pause size={16}/>:<Play size={16}/>}</button><div className="mm-stage-buttons" role="group" aria-label="信号流阶段">{STAGES.map((s,i)=><button key={s.title} aria-pressed={stage===i} onClick={()=>jumpStage(i)}><span>{i+1}</span>{s.title}</button>)}</div></div>
      <p className="mm-stage-description" aria-live="polite">{STAGES[stage].description}</p>
    </div><Inspector node={node} onSelect={select}/></div>
    <div className="mm-under-scene"><div className="mm-legend">{MODALITIES.map(m=><span key={m.id}><i style={{background:m.color}}/>{m.label}</span>)}<span><i style={{background:'#ebdfb9'}}/>PoE 融合</span><span><i className="mm-legend-shared"/>同层 Q / P 共享权重</span></div><label className="mm-spread-control">塔间展开<input aria-label="调整编码塔间距" type="range" min="0" max="1" step="0.1" value={spread} onChange={e=>{setSpread(Number(e.target.value));setPlaying(false)}}/><span>{spread>0?'模块检视 · 连线暂隐':'连接视图'}</span></label></div>

    <section className="mm-directory" id="mm-module-directory"><div className="mm-section-heading"><div><h4>每一个 block，都可追溯。</h4><p>选择模块展开真实算子顺序。外围 Q 副本表示不同模态调用，同层共享参数。</p></div><label className="mm-search"><Search size={15}/><input type="search" placeholder="查找模块或尺寸" aria-label="查找模块或尺寸" value={search} onChange={e=>setSearch(e.target.value)}/></label></div>
      <div className="mm-directory-tabs" role="group" aria-label="模块目录筛选">{[['layer',`当前层 z${level}`],['core','中央生成路径'],...MODALITIES.map(m=>[m.id,`${m.label} 编码塔`]),['outputs','输出与训练']].map(([id,label])=><button key={id} aria-pressed={directory===id} onClick={()=>setDirectory(id)}>{label}</button>)}</div>
      <div className="mm-node-grid">{modules.length?modules.map(n=><button key={n.id} aria-pressed={selected===n.id} onClick={()=>{select(n.id);if(window.innerWidth<1000)showInspector()}}><span className="mm-node-kind">{n.kind}</span><strong>{n.title}</strong><small>{n.shape}</small><ChevronRight size={14}/></button>):<p className="mm-no-results">当前分类没有匹配模块。尝试搜索 Encoder、Gaussian 或 128。</p>}</div>
      <details className="mm-shapes"><summary>查看完整张量尺寸与模块数量 <ChevronRight size={16}/></summary><div className="mm-table-scroll"><table><caption>train.py 默认配置 · base_features=16 · max_features=128 · pools=6</caption><thead><tr><th>潜变量</th><th>编码特征 C×H×W</th><th>Gaussian 参数</th><th>潜变量 C×H×W</th><th>连接</th></tr></thead><tbody>{LEVELS.map(v=><tr key={v.l}><th>z{v.l}</th><td>{v.feature} × {v.encoderSize} × {v.encoderSize}</td><td>{v.l===7?'Linear 1152 → 512':`BlockQ ${2*v.feature} → ${v.feature}`} → split</td><td>{v.channels} × {v.size} × {v.size}</td><td>{v.l===7?'FC → 128×3×3':v.l===1?'四个图像解码器':'Bilinear ×2 → Conv'}</td></tr>)}</tbody></table></div><p>独立编码塔：4 ×（1 stem + 7 BlockEncoder + 6 下采样 Conv）。共享核心：1 bottleneck_down、1 bottleneck_up、6 Upsample、6 BlockDecoder、6 BlockQ、6 可学习 prior head；z₇ 先验固定。输出：4 ×（6 ResnetBlock + 2 Conv 7×7）。</p></details>
    </section>

    <section className="mm-math-section"><div className="mm-section-heading"><div><h4>从分布到采样</h4><p>保留论文的数学含义，也精确标出公开实现的参数化。</p></div><BookOpen size={22}/></div><div className="mm-equation-layout"><div>
      <div className="mm-math-tabs" role="group" aria-label="公式版本"><button aria-pressed={formulaTab==='code'} onClick={()=>setFormulaTab('code')}>源码 · compute_full</button><button aria-pressed={formulaTab==='theory'} onClick={()=>setFormulaTab('theory')}>理论 · Gaussian PoE</button></div>
      <MathFormula formula={formulaTab==='code'?IMPLEMENTATION:THEORY}/>
      <p>{formulaTab==='code'?'a 表示 soft-clamp 后的 log-scale，s 是标准差。代码按 1/s 累加，并把 T·s 传给 torch.distributions.Normal。这里如实呈现代码，不把它写成倒数方差融合。':'Σ 表示协方差，Λ 表示精度。标准 Gaussian PoE 按逆协方差相加；对角情形即按 1/s² 加权。该式与公开代码的 1/s 加权存在差异。'}</p>
      <a className="mm-inline-source" href={source('network/mhvae.py',190)} target="_blank" rel="noreferrer">核对 compute_full <ArrowUpRight size={13}/></a>
    </div><div className="mm-sampling"><div className="mm-range-title"><label htmlFor="mm-temperature">采样温度 T</label><output htmlFor="mm-temperature">{temperature.toFixed(1)}</output></div><input id="mm-temperature" type="range" min="0.1" max="1" step="0.1" value={temperature} onChange={e=>setTemperature(Number(e.target.value))}/>
      <svg viewBox="0 0 252 120" role="img" aria-label={`标准化高斯曲线示意：温度 ${temperature.toFixed(1)}，均值保持不变，标准差随温度缩放。曲线按峰值归一化。`}><path className="mm-gaussian-axis" d="M12 100H240M126 10V105"/><path className="mm-gaussian-reference" d={gaussianPath(1)}/><path className="mm-gaussian-live" d={`${gaussianPath(temperature)} L240 100 L12 100 Z`}/><text x="122" y="118">μ</text><text x="204" y="18">T = 1</text></svg>
      <MathFormula formula={String.raw`z_l=\mu_l+${temperature.toFixed(1)}\,s_l\odot\epsilon`}/><p>温度仅缩放采样标准差。曲线按峰值归一化以比较宽度，不表示实测后验或生成质量。论文推理采用 T = 0.5。</p>
    </div></div>
    <details className="mm-equation-detail" open><summary>七层生成模型与子集混合后验 <ChevronRight size={16}/></summary><MathFormula formula={String.raw`\begin{aligned}p_\theta(z,x)&=p(z_7)\prod_{l=1}^{6}p_\theta(z_l\mid z_{>l})\prod_{j=1}^{4}p_{\theta_j}(x_j\mid z_1)\\q_\phi(z\mid x_r)&=\sum_{r'\in\mathcal S_r}\alpha_{r'}^{(r)}q_\phi(z\mid x_{r'})\end{aligned}`}/><p>混合发生在输入子集的后验分布上；每个子集内部逐层进行 PoE。论文训练将完整观测、仅 iUS、MRI 子集三类各赋 1/3 权重，再在 MRI 的非空子集中均匀分配。</p></details></section>

    <section className="mm-training"><div className="mm-section-heading"><div><h4>训练时，三条路线共同约束。</h4><p>以下对应公开 train.py；判别器只参与训练。</p></div><ArrowDown size={20}/></div><div className="mm-training-routes">{[['完整观测','全部实际存在的模态 → 模型 → 重建已知模态'],['仅 iUS','iUS → 模型 → 已知模态重建 + 对抗约束'],['MRI 子集','每个非空 MRI 子集 → 模型 → 重建 + 对抗约束']].map(([title,copy],i)=><div key={title}><span>{i+1}</span><h5>{title}</h5><p>{copy}</p></div>)}</div>
      <MathFormula formula={String.raw`\mathcal L=\mathcal L_{\rm img}+\lambda_{\rm KL}\frac{\sum_{b,l}(\log q_l(z_l)-\log p_l(z_l))}{N_{\rm voxels}\,M}+\lambda_{\rm adv}\mathcal L_{\rm adv}`}/>
      <div className="mm-training-notes"><p><strong>逐项对应源码</strong>每个 MRI 子集内执行完整观测、仅 iUS、MRI 子集三次前向。图像损失为已知模态上的平均 L1 / 2；KL 使用采样点的 log 概率差估计。权重默认在前 50 个 epoch 将 KL 升至 0.001。</p><p><strong>对抗分支</strong>四个 PatchGAN 配合模态独立样本池。判别器训练从 epoch 790 开始，生成器对抗权重从 800 开启；CLI 默认 0.05，README 示例与论文为 0.025。</p></div>
      <div className="mm-discriminators">{MODALITIES.map(m=><button key={m.id} onClick={()=>{select(`${m.id}-discriminator`);showInspector()}}>查看 {m.label} PatchGAN <ArrowUpRight size={14}/></button>)}</div>
    </section>
    <details className="mm-source-audit" open><summary><Code2 size={17}/> 论文与公开实现：四处需要区分的细节 <ChevronRight size={16}/></summary><div className="mm-table-scroll"><table><thead><tr><th>位置</th><th>论文 §4.2</th><th>本展示采用的公开实现</th></tr></thead><tbody><tr><th>残差单元</th><td>描述为 MobileNetV2 + SE + Swish</td><td>Encoder：两次 3×3 Conv；中央 Decoder：×6 扩展 + DW 5×5 + 1×1 投影。</td></tr><tr><th>图像解码器</th><td>5 个 ResNet block</td><td>train.py / pred.py 默认 nb_finalblocks=6；本页逐个展开 6 个。</td></tr><tr><th>高斯融合</th><td>按逆协方差相加</td><td>compute_full 使用倒数 scale；Normal 第二参数是标准差。</td></tr><tr><th>子集与权重</th><td>三类输入各 1/3，MRI 子集均匀</td><td>遍历 MRI 非空子集，每次三路前向；并非所有非空观测子集均匀混合。</td></tr></tbody></table></div><p>展示固定于源码提交 <a href={`https://github.com/ReubenDo/MMHVAE/tree/${COMMIT}`} target="_blank" rel="noreferrer">{COMMIT.slice(0,7)} <ArrowUpRight size={12}/></a>。张量尺寸由默认参数推导，未加载预训练权重。动画表达计算路径，不生成医学影像。</p><div className="mm-audit-links"><a href={source('train.py',163)} target="_blank" rel="noreferrer">训练循环 <ArrowUpRight size={13}/></a><a href={source('train.py',505)} target="_blank" rel="noreferrer">默认配置 <ArrowUpRight size={13}/></a><a href={source('network/blocks.py',54)} target="_blank" rel="noreferrer">全部基础模块 <ArrowUpRight size={13}/></a></div></details>
  </div>
}
