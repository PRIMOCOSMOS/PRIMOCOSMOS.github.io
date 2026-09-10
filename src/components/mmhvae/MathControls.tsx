import {ChevronLeft,ChevronRight,Pause,Play,MoveRight} from 'lucide-react'
import katex from 'katex'
import {useMemo} from 'react'
import {EditableText} from '../EditableContent'
import {activateValue,bilinear,convolutionDemo,demoValues,fusionValues,normalSamples,uniformSamples,type MathSpec} from './mathematics'

function readout(spec:MathSpec,p:number,t:number){
 const i=Math.min(47,Math.floor(p*48)),x=demoValues[i]
 if(spec.kind==='convolution'||spec.kind==='depthwise'){const d=convolutionDemo(spec),i=Math.min(d.outN*d.outN-1,Math.floor(p*d.outN*d.outN));return `输出 [${Math.floor(i/d.outN)}, ${i%d.outN}] · y = ${d.output[i].toFixed(3)} · 核 ${d.k}×${d.k} / 步幅 ${spec.stride} · 示例 ${d.outN}×${d.outN} 输出网格`}
 if(spec.kind==='linear')return `观察输出行 ${Math.min(3,Math.floor(p*4))} · 示例矩阵 4×6；对应 6 个输入坐标的点积`
 if(['addition','multiplication','mask'].includes(spec.kind)){const j=Math.min(15,Math.floor(p*16));return `对齐元素 [${Math.floor(j/4)}, ${j%4}] · 示例 4×4 网格；两路操作数与当前高亮输出一一对应`}
 if(spec.kind==='padding'){const j=Math.min(35,Math.floor(p*36)),r=Math.floor(j/6)-1,c=j%6-1,reflect=(v:number)=>v<0?-v:v>3?6-v:v;return `扩展网格 [${Math.floor(j/6)}, ${j%6}] → 原图 [${reflect(r)}, ${reflect(c)}] · x = ${demoValues[reflect(r)*4+reflect(c)].toFixed(3)}`}
 if(['concatenation','split'].includes(spec.kind)){const j=Math.min(35,Math.floor(p*36));return `元素 ${j} → 通道 ${Math.floor(j/9)} / 行 ${Math.floor(j/3)%3} / 列 ${j%3} · 示例 4×3×3，按通道分组`}
 if(spec.kind==='broadcast'){const j=Math.min(26,Math.floor(p*27)),c=Math.floor(j/9);return `通道 ${c} 的权重 ${(.25+c*.3).toFixed(2)} → 空间 [${Math.floor(j/3)%3}, ${j%3}] · 示例 3×3×3`}
 if(spec.kind==='activation'){const x=(p*2-1)*(spec.operation.includes('clamp')?20:4);return `x = ${x.toFixed(2)} → f(x) = ${activateValue(x,spec.operation).toPrecision(4)}`}
 if(spec.kind==='fusion'){const s=.45+p*1.35,v=fusionValues(s,t),w=v.weights.reduce((a,b)=>a+b),m=v.contributions.reduce((a,b)=>a+b);return spec.operation==='weight'?`专家 s = ${s.toFixed(2)} · 专家 w = ${(1/s).toFixed(3)} · 总权重 = ${w.toFixed(3)}`:spec.operation==='weighted-mean'?`贡献 [${v.contributions.map(x=>x.toFixed(3)).join(', ')}] · 加权和 m = ${m.toFixed(3)}`:`w = ${w.toFixed(3)} · m = ${m.toFixed(3)} → μ = ${v.mu.toFixed(3)} · s = ${v.baseScale.toFixed(3)}`}
 if(spec.kind==='distribution')return `示例 μ = 0 · s = ${(.45+p*1.35).toFixed(2)} · Var = ${((.45+p*1.35)**2).toFixed(3)} · 显示高度 = 8×密度`
 if(spec.kind==='sampling'){const i=Math.min(79,Math.floor(p*80)),u=uniformSamples[i],e=normalSamples[i];return `u₁ = ${u[0].toFixed(3)} · u₂ = ${u[1].toFixed(3)} → ε₁ = ${e[0].toFixed(3)} · ε₂ = ${e[1].toFixed(3)}`}
 if(spec.kind==='interpolation'){const j=Math.min(63,Math.floor(p*64)),v=bilinear(demoValues.slice(0,16),4,(j%8+.5)/2-.5,(Math.floor(j/8)+.5)/2-.5);return `目标 [${Math.floor(j/8)}, ${j%8}] · 四邻点权重 [${v.weights.map(x=>x.toFixed(2)).join(', ')}] · y = ${v.value.toFixed(3)}`}
 if(['normalization','pooling'].includes(spec.kind))return `观察通道 ${Math.min(2,Math.floor(p*3))} · 同一 batch 内独立空间统计`
 if(spec.kind==='tensor'&&['zeros','ones'].includes(spec.operation))return `索引 ${i} · 数值 ${spec.operation==='zeros'?0:1} · 通道与空间尺寸保持不变`
 return `元素 ${i} → 通道 ${Math.floor(i/16)} / 行 ${Math.floor(i/4)%4} / 列 ${i%4} · 原值 ${x.toFixed(3)} · 示例 3×4×4`
}
export function MathControls({spec,selected,onSelect,probe,onProbe,progress,onProgress,playing,onPlaying,temperature,onOverview,textKeyBase,readoutText}:{spec:MathSpec;selected:number;onSelect:(n:number)=>void;probe:number;onProbe:(v:number)=>void;progress:number;onProgress:(v:number)=>void;playing:boolean;onPlaying:()=>void;temperature:number;onOverview:()=>void;textKeyBase:string;readoutText?:string}){
 const current=spec.steps[selected]??spec.steps[0],html=useMemo(()=>katex.renderToString(current.formula,{displayMode:true,throwOnError:false,trust:false,output:'htmlAndMathml'}),[current.formula])
 return <section className="mm-math-controls" aria-label="原子运算数学实验">
  <div className="mm-math-steps" role="group" aria-label="数学推导步骤"><button onClick={onOverview}>完整演算</button>{spec.steps.map((s,i)=><button key={s.title} aria-pressed={i===selected} onClick={()=>onSelect(i)}><span>{i+1}</span>{s.title}</button>)}</div>
  <div className="mm-math-controls-body"><div className="mm-math-current"><div className="mm-math-stage-title"><button aria-label="上一个数学步骤" disabled={!selected} onClick={()=>onSelect(selected-1)}><ChevronLeft size={16}/></button><strong>{current.title}</strong><button aria-label="下一个数学步骤" disabled={selected===spec.steps.length-1} onClick={()=>onSelect(selected+1)}><ChevronRight size={16}/></button></div><div className="mm-math" tabIndex={0} dangerouslySetInnerHTML={{__html:html}}/><EditableText textKey={`mm-math-step-${textKeyBase}-${selected}`}>{current.explanation}</EditableText></div>
   <div className="mm-math-inputs"><label>{spec.control}<input type="range" aria-label={spec.control} min="0" max="1" step=".01" value={probe} onChange={e=>onProbe(Number(e.target.value))}/></label><output>{readoutText??readout(spec,probe,temperature)}</output><div><button onClick={onPlaying} aria-label={playing?'暂停数学演示':'播放数学演示'}>{playing?<Pause size={15}/>:<Play size={15}/>}<span>{playing?'暂停':'播放'}</span></button><label>演算进度<input type="range" aria-label="手动调整数学演算进度" min="0" max="1" step=".01" value={progress} onChange={e=>onProgress(Number(e.target.value))}/></label></div></div>
  </div><div className="mm-crystal-key" aria-label="三维演算图例"><span><i className="mm-crystal-key-focus"/>当前元素</span><span><i className="mm-crystal-key-active"/>参与计算</span><span><i className="mm-crystal-key-idle"/>其余张量</span><span><MoveRight size={20} aria-hidden="true"/>箭头：输入到结果</span></div><details><summary>示意比例与坐标含义</summary><EditableText textKey={`mm-math-legend-${textKeyBase}`}>{`${spec.legend} 张量单元使用等尺寸水晶块：冷蓝表示非负值，暖色表示负值，色彩强度反映幅值；浅白高亮是当前元素，低亮有色单元参与当前运算，透明单元保留上下文。箭头表示数据依赖方向，暂停后仍可读；重排与广播的路径表示索引映射，不表示内存复制。数值为小规模确定性教学示例，未加载训练权重。`}</EditableText></details>
 </section>
}
