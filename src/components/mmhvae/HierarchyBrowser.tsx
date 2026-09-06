import { useMemo, useState } from 'react'
import { ArrowLeft, ChevronRight, Folder, Box, FileText, ArrowUpRight, Search } from 'lucide-react'
import katex from 'katex'
import { MathLabel } from './MathLabel'
import { EditableText } from '../EditableContent'
import { NODE_MAP, source } from './model'
import { atomInfo, canonicalPath, childrenOf, graphFor, isFolder, navName, partName, principle, parameterLabel } from './navigation'

function Equation({value}:{value:string}){const html=useMemo(()=>katex.renderToString(value,{displayMode:true,throwOnError:false,trust:false,strict:'ignore',output:'htmlAndMathml'}),[value]);return <div className="mm-math" tabIndex={0} aria-label="当前层级数学原理" dangerouslySetInnerHTML={{__html:html}}/>}
export default function HierarchyBrowser({id,observed,onNavigate}:{id:string;observed:string[];onNavigate:(id:string)=>void}){
  const [query,setQuery]=useState('')
  const path=canonicalPath(id),graph=useMemo(()=>graphFor(id,observed),[id,observed]),atom=atomInfo(id,observed)
  const node=NODE_MAP.get((atom?.parent??id).split('/')[0]),children=childrenOf(id,observed),isSE=id.includes('/se/')&&!atom
  const explanation=atom?principle(atom.part):isSE?principle({id:'se',title:'通道注意力',glyph:'se',shape:'',detail:'',sourceName:'se'}):null
  const folderPurpose:Record<string,string>={root:'将可用模态的观测编码为七层共享潜变量，再由四个独立生成器重建目标影像。目录表示模块的组成关系；输入来源和输出去向单独列出。',encoders:'四座独立编码塔提取各模态的多尺度特征。每个尺度保留跳接，为对应潜变量层提供观测证据。',core:'从最粗尺度向细尺度逐层生成。每层用生成条件预测先验，将模态证据融合为后验，再采样送到下一层。',outputs:'四个独立影像生成器读取同一 z₁，通过六个残差块和影像映射尾部生成各模态。',training:'四个模态独立判别器仅在训练时约束生成影像的真实性。'}
  const purpose=explanation?.purpose??(isFolder(id)?folderPurpose[id]??'从输入切片出发，交替使用编码残差块与下采样，构建七个尺度的观测特征。':id.startsWith('layer-')?'把上层生成条件与本层可用观测汇合，获得当前尺度的潜变量样本。':node?.description??graph?.note??'')
  const formula=explanation?.formula??(id==='layer-7'?String.raw`p_7=\mathcal N(0,I),\quad q_7=\operatorname{PoE}(p_7,\{Q_7(e_{j,7})\}_j),\quad z_7\sim q_7`:id.startsWith('layer-')?String.raw`g_l=D_l(U_l(z_{l+1})),\quad q_l=\operatorname{PoE}(p_l,\{Q_l([g_l,e_{j,l}])\}_j),\quad z_l\sim q_l`:isFolder(id)?id==='outputs'?String.raw`\hat x_j=G_j(z_1)`:id==='training'?String.raw`\mathcal L_D=\mathbb E[(D(x)-1)^2+D(G(z))^2]`:id==='encoders'||id.startsWith('encoder:')?String.raw`e_{j,l+1}=E_{j,l+1}(\operatorname{Down}_l(e_{j,l}))`:String.raw`p(z,x)=p(z_7)\prod_{l=1}^6p(z_l\mid z_{>l})\prod_jp(x_j\mid z_1)`:node?.formula??'')
  const navigate=(next:string)=>{setQuery('');onNavigate(next)}
  const tree=(key:string,depth=0):React.ReactNode=><div key={key}><button style={{paddingLeft:12+depth*15}} aria-current={id===key?'location':undefined} onClick={()=>navigate(key)}>{childrenOf(key,observed).length?<Folder size={14}/>:<FileText size={14}/>}<span>{navName(key,observed)}</span></button>{path.includes(key)&&key!==id&&childrenOf(key,observed).map(child=>tree(child,depth+1))}</div>
  return <section className="mm-browser" id="mm-module-directory" aria-label="模型层级资源管理器" data-location={id}>
    <div className="mm-browser-toolbar"><button aria-label="返回上一级" disabled={path.length===1} onClick={()=>navigate(path.at(-2)!)}><ArrowLeft size={17}/></button><nav aria-label="模型层级路径">{path.map((key,i)=><span key={key}>{i>0&&<ChevronRight size={12}/>}<button aria-current={key===id?'location':undefined} onClick={()=>navigate(key)}>{navName(key,observed)}</button></span>)}</nav></div>
    <div className="mm-browser-layout"><aside className="mm-browser-tree" aria-label="模型组成目录">{tree('root')}</aside><div className="mm-browser-content">
      <header><div><h4><MathLabel value={navName(id,observed)}/></h4><span>{atom?'原子运算 · 最深层级':`${children.length} 个直接组成部分`}{node&&!atom?` · ${parameterLabel(node)}`:''}</span></div>{graph&&<strong className="mm-browser-shape"><MathLabel value={graph.tensor} math/></strong>}</header>
      <div className="mm-browser-explanation"><h5>原理与设计目的</h5><EditableText textKey={atom?`mm-purpose-${id}`:node&&!isSE?`mm-description-${node.id}`:`mm-purpose-${id}`}>{purpose}</EditableText><Equation value={formula}/>
      {atom?<><h5>运算与数据结构</h5><EditableText textKey={`mm-part-${atom.parent}-${atom.part.id}`}>{atom.part.detail}</EditableText><code>{atom.part.sourceName}</code></>:graph?<><h5>实现与数据流</h5><EditableText textKey={`mm-anatomy-note-${id}`}>{graph.note}</EditableText></>:null}
      {node&&<a className="mm-inline-source" href={source(node.file,node.line)} target="_blank" rel="noreferrer">{node.file}:{node.line}<ArrowUpRight size={13}/></a>}</div>
      {!!children.length&&<><div className="mm-browser-list-heading"><h5>内部组成</h5><label><Search size={14}/><input aria-label="筛选当前层子模块" placeholder="在这一层查找" value={query} onChange={e=>setQuery(e.target.value)}/></label></div><div className="mm-browser-files" aria-label="当前层子模块">
      {children.filter(key=>navName(key,observed).toLowerCase().includes(query.toLowerCase())).map(key=>{const a=atomInfo(key,observed),n=NODE_MAP.get(key),compound=childrenOf(key,observed).length>0;return <button key={key} onClick={()=>navigate(key)} data-entry={key}>{compound?<Folder size={17}/>:<Box size={17}/>}<span><strong><MathLabel value={navName(key,observed)}/></strong><small>{a?.part.sourceName??(n?`${n.kind} · ${parameterLabel(n)}`:'结构分组')}</small></span><span className="mm-file-shape"><MathLabel value={a?.part.shape??n?.shape??''} math/></span><ChevronRight size={15}/></button>})}
      {query&&!children.some(key=>navName(key,observed).toLowerCase().includes(query.toLowerCase()))&&<p>这一层没有匹配项。清空搜索可显示全部子模块。</p>}</div></>}
      {!!graph?.parts.some(p=>p.role)&&<div className="mm-browser-connections"><h5>边界外的连接</h5><p>连接表示数据去向，不属于当前模块的内部组成。</p>{graph.parts.filter(p=>p.role).map(p=><button key={p.id} disabled={!p.child} onClick={()=>p.child&&navigate(p.child)}><span>{p.role==='input'?'输入来自':'输出流向'}</span><strong><MathLabel value={partName(p)}/></strong><small><MathLabel value={p.shape} math/></small><ArrowUpRight size={14}/></button>)}</div>}
    </div></div>
  </section>
}
