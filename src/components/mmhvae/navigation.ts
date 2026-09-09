import { MODALITIES, LEVELS, NODE_MAP, type ModelNode } from './model'
import { anatomyFor, glyphFor, type AnatomyGraph, type AnatomyPart } from './anatomy'
import { modelTopology } from './topology'
import { mathematicalGraph } from './mathematics'

export const moduleName=(n:ModelNode)=>`${n.mod?`${MODALITIES.find(m=>m.id===n.mod)?.label} · `:''}${({input:'输入影像',stem:'初始特征提取',encoder:'编码残差块',down:'空间下采样',expert:'模态概率估计',prior:'条件先验分布',poe:'多专家融合',sample:'潜变量采样',up:'空间上采样',decoder:'生成残差块',lift:'向量还原为空间特征',output:'影像生成器',resnet:'影像残差块',image:'生成影像',discriminator:'真实性判别器',feature:'共享生成特征',concat:'条件与观测拼接',priorhead:'先验参数预测',factor:'模态残差分布',posterior:'融合后验分布'} as Record<string,string>)[n.kind]}${['input','stem','output','image','discriminator','lift'].includes(n.kind)?'':n.kind==='resnet'?` ${n.id.split('-').at(-1)}`:` · 第 ${n.l} 层`}`
export function partName(p:AnatomyPart):string {
  if(p.math)return p.title
  if(p.sourceName.endsWith('.convt2'))return '影像细节映射 · 7×7 卷积'
  if(p.child&&NODE_MAP.has(p.child))return moduleName(NODE_MAP.get(p.child)!)
  const specific:Record<string,string>={mean:'全局空间平均',flatten:'展平通道向量',fc1:'压缩通道描述',fc2:'恢复通道权重',view:'重排张量维度',gate:'施加通道权重',weight:'累加分布权重',weighted_mu:'累加加权均值',normalize:'归一化融合参数',chunk:'分离均值与尺度',split:'分离分布参数',eps:'生成标准正态噪声',se:'通道注意力',bn_0:'输入特征归一化',bn_1:'中间特征归一化',bn_2:'深度卷积后归一化'}
  if(specific[p.id])return specific[p.id]
  if(p.glyph==='conv')return `局部特征卷积 · ${p.title.replace('WeightNorm ','').replace('Conv ','')}`
  if(p.glyph==='norm')return '逐通道空间归一化'
  if(p.glyph==='depthwise')return '逐通道卷积 · 5×5'
  if(p.glyph==='activation')return `${p.title} · 非线性映射`
  if(p.glyph==='linear')return '全连接特征映射'
  if(p.glyph==='up')return '双线性空间放大'
  return p.title.replace('Squeeze & Excite','通道注意力').replace('ResnetBlock','影像残差块').replace('Concat','通道拼接')
}
const folderNames:Record<string,string>={root:'MMHVAE 模型',encoders:'四模态编码',core:'层次潜变量生成',outputs:'四模态影像生成',training:'训练判别网络'}
for(const m of MODALITIES)folderNames[`encoder:${m.id}`]=`${m.label} 编码塔`
export const isFolder=(id:string)=>id in folderNames
export const atomId=(graph:string,part:string)=>`atom:${encodeURIComponent(graph)}:${encodeURIComponent(part)}`
export function atomInfo(id:string,observed:string[]){if(!id.startsWith('atom:'))return null;const [,g,p]=id.split(':');const parent=decodeURIComponent(g);return {parent,part:anatomyFor(parent,observed).parts.find(v=>v.id===decodeURIComponent(p))!}}
export function navName(id:string,observed:string[]):string {if(isFolder(id))return folderNames[id];const atom=atomInfo(id,observed);if(atom)return partName(atom.part);if(id.startsWith('layer-'))return `第 ${id.slice(6)} 层 · 潜变量推导`;if(id.includes('/se/'))return '通道注意力';return NODE_MAP.has(id)?moduleName(NODE_MAP.get(id)!):id}
export function canonicalPath(id:string):string[]{
  if(id==='root')return ['root']
  if(isFolder(id))return id.startsWith('encoder:')?['root','encoders',id]:['root',id]
  if(id.startsWith('atom:')){const [,g]=id.split(':');return [...canonicalPath(decodeURIComponent(g)),id]}
  if(id.includes('/se/'))return [...canonicalPath(id.split('/')[0]),id]
  if(id.startsWith('layer-'))return ['root','core',id]
  const n=NODE_MAP.get(id);if(!n)return ['root']
  if(['input','stem','encoder','down'].includes(n.kind))return ['root','encoders',`encoder:${n.mod}`,id]
  if(n.kind==='discriminator')return ['root','training',id]
  if(n.kind==='resnet')return ['root','outputs',`${n.mod}-output`,id]
  if(['output','image'].includes(n.kind))return ['root','outputs',id]
  return ['root','core',`layer-${n.l}`,id]
}
export function childrenOf(id:string,observed:string[]):string[]{
  if(id==='root')return ['encoders','core','outputs','training']
  if(id==='encoders')return MODALITIES.map(m=>`encoder:${m.id}`)
  if(id==='core')return LEVELS.map(l=>`layer-${l.l}`)
  if(id==='outputs')return MODALITIES.flatMap(m=>[`${m.id}-output`,`${m.id}-image`])
  if(id==='training')return MODALITIES.map(m=>`${m.id}-discriminator`)
  if(id.startsWith('encoder:')){const mod=id.split(':')[1];return [`${mod}-input`,`${mod}-stem`,...Array.from({length:7},(_,i)=>[`${mod}-encoder-${i+1}`,...i<6?[`${mod}-down-${i+1}`]:[]]).flat()]}
  if(id.startsWith('atom:'))return []
  return anatomyFor(id,observed).parts.filter(p=>!p.role).map(p=>p.child??atomId(id,p.id))
}
export function graphFor(id:string,observed:string[]):AnatomyGraph|null {
  if(isFolder(id)) {
    if(['root','training'].includes(id))return null
    const top=modelTopology(),ids=[...top.positions.keys()].filter(key=>{
      const n=NODE_MAP.get(key)!
      if(id==='core')return !n.mod
      if(id==='outputs')return ['output','image'].includes(n.kind)
      return ['input','stem','encoder','down'].includes(n.kind)&&(id==='encoders'||n.mod===id.split(':')[1])
    })
    return {layout:'overview',origin:[0,0,0],title:navName(id,observed),tensor:id==='core'?'256 → 8 × 192²':id==='outputs'?'4 × (1 × 192²)':'1 × 192² → 128 × 3²',note:'保持总览中的相对位置、方向与连接；镜头聚焦当前结构子集。点击其中的模块进一步进入内部组成。',parts:ids.map(key=>{const n=NODE_MAP.get(key)!;return {id:key,child:key,title:moduleName(n),glyph:glyphFor(n.kind),shape:n.shape,detail:n.description,sourceName:key,mod:n.mod,position:top.positions.get(key)}}),edges:top.links.filter(e=>ids.includes(e.from)&&ids.includes(e.to))}
  }

  const atom=atomInfo(id,observed)
  if(!atom)return anatomyFor(id,observed)
  const parent=anatomyFor(atom.parent,observed),part={...atom.part,child:undefined,position:undefined}
  const edges=parent.edges.filter(e=>e.from===part.id||e.to===part.id)
  const neighbors=parent.parts.filter(p=>p.id!==part.id&&edges.some(e=>e.from===p.id||e.to===p.id)).map(p=>({...p,position:undefined,role:edges.some(e=>e.from===p.id&&e.to===part.id)?'input' as const:'output' as const,child:p.child??atomId(atom.parent,p.id)}))
  return mathematicalGraph(part,neighbors,edges,partName(part),principle(part).formula)
}
export function principle(p:AnatomyPart):{purpose:string;formula:string}{
  const table:Record<string,[string,string]>={
    tensor:['保留或重排数据的通道与空间结构，使相邻算子能够传递同一组特征。',String.raw`X\in\mathbb R^{B\times C\times H\times W}`],image:['以二维切片承载模态影像；体数据在网络入口组织为切片批次。',String.raw`x_j\in\mathbb R^{B\times1\times192\times192}`],
    conv:['通过局部感受野学习空间模式，并完成通道映射。',String.raw`Y_o=b_o+\sum_c W_{o,c}*X_c`],depthwise:['分别提取各通道的空间模式，减少通道混合卷积的计算量。',String.raw`Y_c=W_c*X_c,\quad\mathrm{groups}=C`],
    norm:['统一每个样本、每个通道的空间响应尺度，使后续特征变换更稳定。',String.raw`Y_{b,c}=\frac{X_{b,c}-\mu_{b,c}}{\sqrt{\sigma_{b,c}^2+10^{-5}}}`],
    linear:['学习向量维度间的线性关系，以压缩通道信息或恢复空间表示。',String.raw`y=Wx+b`],pool:['汇总全图信息，为每个通道生成一个全局描述值。',String.raw`u_c=\frac1{HW}\sum_{h,w}X_{c,h,w}`],
    se:['利用全局上下文调整各通道的重要性。',String.raw`Y=X\odot\sigma(W_2\operatorname{ReLU}(W_1\operatorname{GAP}(X)))`],
    concat:['保留生成条件与模态观测的独立信息，沿通道组合后交给概率估计。',String.raw`h_j=\operatorname{cat}(g_l,e_{j,l};\mathrm{dim}=1)`],split:['将预测结果分成均值与尺度两组参数。',String.raw`[\mu,a]=\operatorname{chunk}(h,2;\mathrm{dim}=1)`],
    gaussian:['用均值和标准差表达特征的不确定性，供融合或采样使用。',String.raw`q(z)=\mathcal N(\mu,\operatorname{diag}(s^2))`],
    poe:['合并条件先验与可用模态的证据；公开实现按倒数标准差累加。',String.raw`w=s_p^{-1}+\sum_j e^{-a_j},\quad s=w^{-1}`],
    sample:['通过标准正态噪声重参数化，让随机采样保留梯度路径。',String.raw`z=\mu+Ts\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I)`],
    up:['恢复空间分辨率，使生成路径与下一层编码跳接具有相同尺寸。',String.raw`X^{\uparrow}_{c,2h,2w}=\operatorname{Bilinear}(X)_{c,2h,2w}`],
    sum:['将主支路变换与旁路信息相加，保留直接的信息和梯度路径。',String.raw`Y=F(X)+X`],multiply:['通过逐元素乘法施加权重，权重按需要在空间维广播。',String.raw`Y=X\odot w`],network:['组合局部算子，在保持层级接口一致的同时学习特征变换。',String.raw`Y=f_n\circ\cdots\circ f_1(X)`],
    activation:['引入逐元素非线性，或将分布参数限制到稳定范围。',String.raw`\operatorname{SiLU}(x)=x\,\sigma(x)`],
  }
  let [purpose,formula]=table[p.glyph]
  const name=p.title.toLowerCase()
  if(p.glyph==='activation'){
    if(name.includes('clamp'))formula=name.includes('exp')?String.raw`s=\exp(10\tanh(a/10))`:String.raw`f(x)=10\tanh(x/10)`
    else if(name.includes('leaky'))formula=String.raw`f(x)=\max(x,0)+0.2\min(x,0)`
    else if(name.includes('relu'))formula=String.raw`f(x)=\max(0,x)`
    else if(name.includes('sigmoid'))formula=String.raw`\sigma(x)=(1+e^{-x})^{-1}`
    else if(name.includes('tanh'))formula=String.raw`f(x)=\tanh(x)`
  }
  if(p.id==='weighted_mu'){purpose='累加每个分布的均值贡献，随后除以总权重得到融合均值。';formula=String.raw`m=\mu_p/s_p+\sum_j\mu_j e^{-a_j}`}
  if(p.id==='normalize'){purpose='用累加权重统一均值和尺度，形成融合后验的参数。';formula=String.raw`\mu=m/w,\quad s=1/w`}
  if(p.id==='mean'&&p.glyph==='sum')formula=String.raw`z=\mu+Ts\odot\epsilon`
  if(p.id==='scale'&&p.glyph==='multiply')formula=String.raw`\eta=Ts\odot\epsilon`
  if(p.id==='mask')formula=String.raw`y=2\left(\frac{x+1}{2}\odot m\right)-1`
  return {purpose,formula}
}
export const parameterLabel=(n:ModelNode)=>['stem','encoder','down','expert','priorhead','up','decoder','lift','output','resnet','discriminator'].includes(n.kind)?n.shared?'共享权重':'独立权重':'张量 / 概率运算'
