import { LEVELS, MODALITIES, NODE_MAP } from './model'
import { modelTopology, levelY, type Point3 } from './topology'

export type Glyph = 'tensor' | 'image' | 'conv' | 'depthwise' | 'norm' | 'activation' | 'linear' | 'pool' | 'se' | 'concat' | 'split' | 'gaussian' | 'poe' | 'sample' | 'up' | 'sum' | 'multiply' | 'network'
export interface AnatomyPart {
  id: string; title: string; glyph: Glyph; shape: string; detail: string
  child?: string; sourceName: string; color?: string; position?: Point3; mod?: string; role?: 'input' | 'output'
}
export interface AnatomyEdge { from: string; to: string; label?: string; residual?: boolean }
export interface AnatomyGraph { layout?: 'overview'; origin?: Point3; title: string; parts: AnatomyPart[]; edges: AnatomyEdge[]; tensor: string; note: string }

const dims = (c: number, s: number) => `${c} × ${s} × ${s}`
export const glyphFor = (kind: string): Glyph => ({ input:'image', image:'image', stem:'conv', down:'conv', feature:'tensor', concat:'concat', priorhead:'conv', factor:'gaussian', posterior:'gaussian', encoder:'network', decoder:'network', resnet:'network', output:'network', expert:'network', prior:'gaussian', poe:'poe', sample:'sample', up:'up', lift:'linear', discriminator:'network' } as Record<string,Glyph>)[kind] ?? 'tensor'

function rawAnatomy(id: string, observed: string[] = MODALITIES.map(m=>m.id)): AnatomyGraph {
  if (id.startsWith('layer-')) return layerAnatomy(Number(id.slice(6)), observed)
  const [root, scope, channelOverride] = id.split('/')
  const node = NODE_MAP.get(root)!
  const level = LEVELS.find(v=>v.l===node.l)!, c = Number(channelOverride)||level.feature, s=level.encoderSize
  const parts:AnatomyPart[]=[], edges:AnatomyEdge[]=[]
  const add = (key: string, title: string, glyph: Glyph, shape: string, detail: string, sourceName=key, child?:string, after?: string|null) => {
    if(after===undefined&&parts.length) edges.push({from:parts.at(-1)!.id,to:key})
    else if(after)edges.push({from:after,to:key})
    if(node.kind==='resnet')sourceName=({conv0:'conv_block[1]',norm0:'conv_block[2]',relu:'conv_block[3]',conv1:'conv_block[5]',norm1:'conv_block[6]'} as Record<string,string>)[key]??sourceName
    parts.push({id:key,title,glyph,shape,detail,sourceName,child});return key
  }
  const link=(from:string,to:string,label?:string,residual=false)=>edges.push({from,to,label,residual})
  const tensor=(key:string,title:string,channels=c,size=s,after?:string|null)=>add(key,title,'tensor',dims(channels,size),'特征图沿通道堆叠；图中层片数量是示意，实际通道与空间尺寸以上方数字为准。',key,undefined,after)
  const norm=(key:string,ch=c)=>add(key,'InstanceNorm','norm',`${ch} ch`,`逐样本、逐通道做空间归一化；eps=1e−5，momentum=${['resnet','output','discriminator'].includes(node.kind)?'0.1':'0.05'}。默认 affine=False、track_running_stats=False。`,key)
  const act=(key:string,name='SiLU')=>add(key,name,'activation','逐元素 · 尺寸不变',name==='SiLU'?'SiLU / Swish: x·sigmoid(x)。':name==='LeakyReLU'?'负半轴斜率 0.2。':name==='Tanh'?'将输出限制到 [−1,1]。':'逐元素非线性变换。',key)
  const conv=(key:string,cin:number,cout:number,k=3,stride=1,padding=(k-1)/2)=>add(key,`Conv ${k}×${k}`,'conv',`${cin} → ${cout} ch`, `nn.Conv2d(${cin}, ${cout}, kernel_size=${k}, stride=${stride}, padding=${padding})；${node.kind==='output'&&k===7?"padding_mode='reflect'；":''}${key==='last_conv'?'WeightNorm，bias=False':'bias=True'}。`,key)
  const se=(ch=c)=>add('se','Squeeze & Excite','se',`${ch} → ${Math.max(Math.floor(ch/16),4)} → ${ch}`,'空间池化后学习每个通道的门控系数，再乘回原特征。点击继续拆解内部 Linear、激活与广播乘法。',node.kind==='decoder'?'se_2':'se',`${root}/se/${ch}`)
  const end=()=>tensor('output','输出特征')
  const residue=()=>{add('add','残差相加','sum',dims(c,s),'旁路输入与主支路输出逐元素相加；两者尺寸必须相同。','out + x');link('input','add','identity skip',true)}
  let title=node.title,note='每个节点对应代码中的算子或显式张量运算；点击小节点查看参数，带展开标记的节点可继续拆解。'
  if(scope==='se'){
    title=`SE · ${c} channels`
    tensor('input','通道特征',c,s,null)
    add('mean','Global average','pool',`${c} × H × W → ${c}`,'torch.mean(x, dim=[2,3])，为每个通道提取全局响应。','torch.mean')
    add('flatten','View (B, −1)','tensor',`B × ${c}`,'显式 view(se.size(0), −1)，形成 Linear 接收的通道向量。','view')
    add('fc1','Linear','linear',`${c} → ${Math.max(Math.floor(c/16),4)}`,'通道压缩；hidden=max(C//16,4)。','se[0]')
    act('relu','ReLU')
    add('fc2','Linear','linear',`${Math.max(Math.floor(c/16),4)} → ${c}`,'恢复通道数，产生通道门控 logits。','se[2]')
    act('sigmoid','Sigmoid')
    add('view','Broadcast','tensor',`${c} → ${c} × 1 × 1`,'reshape 成 B×C×1×1，广播到空间维度。','view')
    add('gate','通道加权','multiply',dims(c,s),'原特征 x 乘以广播后的门控系数。','x * se');link('input','gate','原特征',true);end()
  } else if(node.kind==='concat') {
    tensor('condition','生成特征 gₗ',c,s,null);tensor('input',`${node.mod} · Encoder skip`,c,s,null)
    add('cat','Concat · dim=1','concat',`${c} + ${c} → ${2*c} × ${s}²`,'上一级样本经过上采样和 BlockDecoder 后得到 gₗ，再与本模态 skip 沿通道拼接。','torch.cat',undefined,'input');link('condition','cat','gₗ')
  } else if(node.kind==='feature') {
    tensor('input','BlockDecoder 输出',c,s,null);add('out','条件特征 gₗ','tensor',dims(c,s),'同一张量分流到先验头 P 和各模态的 Concat。','out')
  } else if(node.kind==='priorhead') {
    tensor('input','生成特征 gₗ',c,s,null);add('pz','WeightNorm Conv 1×1','conv',`${c} → ${c} ch`,'共享 pz[i]，bias=False。','pz[i]');add('split','chunk(2)','split',`${c} → ${c/2} + ${c/2}`,'分离均值与 log-scale。','chunk');add('mu','clamp μ','activation',`${c/2} ch`,'10 tanh(μ/10)。','soft_clamp',undefined,'split');add('scale','clamp a → exp','activation',`${c/2} ch`,'10 tanh(a/10)，再 exp 得到标准差。','soft_clamp',undefined,'split')
  } else if(node.kind==='factor'||node.kind==='posterior') {
    add('mu','均值 μ','tensor',node.shape,'均值张量。','loc',undefined,null);add('scale',node.kind==='factor'?'log-scale a':'标准差 T·s','tensor',node.shape,node.kind==='factor'?'残差专家以 log-scale 保存，不在这里采样。':'融合后的标准差乘以温度。','scale',undefined,null)
    add('distribution',node.kind==='factor'?'残差 Expert':'融合后验 qₗ','gaussian',node.shape,node.description,'res_params / Normal',undefined,'mu');link('scale','distribution','scale')
  } else if(node.kind==='encoder') {
    tensor('input','输入特征',c,s,null);norm('bn_0');act('act_0');conv('conv_0',c,c);norm('bn_1');act('act_1');conv('conv_1',c,c);se();residue();end()
  } else if(node.kind==='decoder') {
    tensor('input','上采样特征',c,s,null);norm('bn_0');conv('conv_0',c,6*c);norm('bn_1',6*c);act('act_1')
    add('dw_conv_1','Depthwise 5×5','depthwise',`${6*c} → ${6*c} ch`,`groups=${6*c}，每个通道单独进行 5×5 卷积；stride=1，padding=2。`,'dw_conv_1')
    norm('bn_2',6*c);act('act_2');se(6*c);conv('conv_2',6*c,c,1);residue();end()
  } else if(node.kind==='expert') {
    tensor('input',node.l===7?'最粗编码特征':'模态 skip',c,s,null)
    if(node.l===7){add('flatten','Flatten','tensor','128 × 3 × 3 → 1152','x.view(B,−1)，保留所有特征，不是全局平均池化。','view');add('linear','WeightNorm Linear','linear','1152 → 512','bottleneck_down 在所有模态间共享权重。','bottleneck_down.linear')}
    else {
      tensor('condition','生成特征 gₗ',c,s,null)
      add('cat','Concat','concat',`${c} + ${c} = ${2*c} ch`,'torch.cat((top-down, skip), dim=1)。两路输入有相同空间尺寸。','torch.cat',undefined,'condition');link('input','cat','skip')
      norm('bn_0',2*c);act('act_0');conv('conv_0',2*c,c);norm('bn_1');act('act_1');conv('conv_1',c,c);se();conv('last_conv',c,c,1)
    }
    const out=node.l===7?256:c/2
    add('chunk','chunk(2)','split',`${2*out} → μ:${out} + a:${out}`,'沿通道维分离均值参数与 log-scale 参数。','chunk(2, dim=1)')
    add('mu','均值 soft-clamp','activation',`${out} ch`,'μ ← 10 tanh(μ/10)，控制数值范围。','soft_clamp',undefined,'chunk')
    add('scale','log-scale clamp','activation',`${out} ch`,'a ← 10 tanh(a/10)。残差专家参数保存为 loc 和 logscale_res。','soft_clamp',undefined,'chunk')
    add('factor','残差 Gaussian 因子','gaussian',`${out} × ${level.size}²`,'两个参数共同定义当前模态的残差专家因子，随后参与 compute_full。','res_params',undefined,'mu');link('scale','factor','a = log s')
  } else if(node.kind==='resnet') {
    tensor('input','z₁ / 残差特征',8,192,null)
    add('pad1','ReflectionPad','tensor','8 × 192² → 8 × 194²','镜像填充一圈边界。','conv_block[0]');conv('conv0',8,8,3,1,0);norm('norm0',8);act('relu','ReLU')
    add('pad2','ReflectionPad','tensor','8 × 192² → 8 × 194²','第二次镜像填充，避免零填充边缘。','conv_block[4]');conv('conv1',8,8,3,1,0);norm('norm1',8)
    add('add','残差相加','sum','8 × 192 × 192','x + conv_block(x)；不使用 dropout。','forward');link('input','add','identity',true);tensor('output','输出特征',8,192)
  } else if(node.kind==='output') {
    tensor('input','共享潜变量 z₁',8,192,null)
    for(let i=1;i<=6;i++)add(`res${i}`,`ResnetBlock ${i}`,'network','8 × 192 × 192','本模态独立的残差块。点击进一步展开两次镜像填充、卷积、IN、ReLU 与残差旁路。',`res[${i-1}]`,`${node.mod}-resnet-${i}`)
    conv('convt2',8,8,7);norm('norm2',8);act('act','LeakyReLU');conv('convt3',8,1,7);act('act_last','Tanh')
    add('mask','背景掩码','multiply','1 × 192 × 192','在 MHVAE2D.forward 中应用首个输入模态的背景 mask：2((x+1)/2·mask)−1。','2*((output_img+1)/2*mask)-1')
    add('output','生成影像','image','1 × 192 × 192','输出到本模态；四个 BlockFinalImg 均独立运行。','output_img')
  } else if(node.kind==='prior') {
    if(node.l===7){add('zeros','μ = 0','tensor','256-vector','固定标准正态先验的零均值。','zeros_like',undefined,null);add('ones','s = 1','tensor','256-vector','固定单位标准差。','ones_like',undefined,null)}
    else {add('zeros','先验均值 μp','tensor',dims(c/2,s),'先验概率头输出并 soft-clamp 后的均值。','mu_zi_p',undefined,null);add('ones','先验标准差 sp','tensor',dims(c/2,s),'先验概率头 log-scale 经 exp 得到的标准差。','exp(logvar_zi_p)',undefined,null)}
    add('prior','条件先验 pₗ','gaussian',node.shape,'torch.distributions.Normal 的第二参数为标准差。','Normal',undefined,'zeros');link('ones','prior','scale')
  } else if(node.kind==='poe') {
    add('prior','先验 pₗ','gaussian',node.shape,'来自上一级潜变量的条件先验，或顶层 N(0,I)。','prior',undefined,null)
    add('factors','观测残差专家','gaussian',`${observed.length} × (μⱼ, aⱼ)`,'仅将当前观测集合中的模态加入融合。','res_params',undefined,null)
    add('weight','倒数 scale 累加','poe',`1/sₚ + Σ exp(−aⱼ)`,'严格对应公开源码的 1/scale 权重，而不是理论 PoE 的 1/variance。','inv_sigma',undefined,'prior');link('factors','weight')
    add('weighted_mu','均值加权累加','sum','μₚ/sₚ + Σ μⱼexp(−aⱼ)','先验均值与各专家均值按倒数 scale 加权。','mu',undefined,'prior');link('factors','weighted_mu')
    add('normalize','归一化 μ / s','split',node.shape,'μ = 加权和 / 权重和；s = 1 / 权重和。','mu /= inv_sigma',undefined,'weight');link('weighted_mu','normalize')
    add('posterior','融合后验 qₗ','gaussian',node.shape,'Normal(μ, T·s)，温度缩放标准差。','Normal(mu, temp*sigma)')
  } else if(node.kind==='sample') {
    add('q','后验 qₗ','gaussian',node.shape,'由 compute_full 得到的融合正态分布。','full',undefined,null)
    add('eps','ε ~ N(0,I)','sample',node.shape,'重参数化中的随机噪声；点云仅表示随机性。','rsample noise',undefined,null)
    add('scale','T · s ⊙ ε','multiply',node.shape,'标准差和温度缩放噪声。','rsample',undefined,'eps');link('q','scale','scale')
    add('mean','μ + scaled ε','sum',node.shape,'加上均值，得到保留梯度路径的潜变量样本。','rsample',undefined,'scale');link('q','mean','loc')
    if(node.l===7)add('z','全局潜变量 z₇','tensor','B × 256','顶层样本是向量；经 bottleneck_up 后才恢复空间维度。','z_full[z7]')
    else tensor('z',`潜变量 z${node.l}`,level.channels,level.size)
  } else if(node.kind==='up') {
    const cin=node.l===6?128:LEVELS.find(v=>v.l===node.l+1)!.channels
    tensor('input','上层样本 / 特征',cin,s/2,null)
    add('resize','Bilinear ×2','up',`${cin} × ${s/2}² → ${cin} × ${s}²`,'nn.functional.interpolate(scale_factor=2, mode=bilinear, align_corners=False)。','interpolate')
    add('conv_1','Conv 3×3','conv',`${cin} → ${c} ch`,'通道映射；stride=1，padding=1，bias=False。','conv_1');end()
  } else if(node.kind==='lift') {
    add('input','全局 z₇','tensor','256-vector','顶层全局潜变量。','z7',undefined,null);add('linear','WeightNorm Linear','linear','256 → 1152','线性学习恢复空间特征。','bottleneck_up.linear');add('view','Reshape','tensor','128 × 3 × 3','view(B,128,3,3)，进入第一个上采样块。','view')
  } else if(node.kind==='stem'||node.kind==='down') {
    tensor('input','输入特征',node.kind==='stem'?1:c,s,null);conv(node.kind==='stem'?'first_conv':'td',node.kind==='stem'?1:c,node.kind==='stem'?16:Math.min(2*c,128),3,node.kind==='stem'?1:2);tensor('output','输出特征',node.kind==='stem'?16:Math.min(2*c,128),node.kind==='stem'?s:s/2)
  } else if(node.kind==='discriminator') {
    tensor('input','真实 / 生成影像',1,192,null)
    let ch=1,sz=192
    for(let i=0;i<6;i++){const out=64*Math.min(2**i,4);add(`conv${i}`,`Conv 4×4 / 2`,'conv',`${ch}→${out} · ${sz}²→${sz/2}²`,'kernel=4,stride=2,padding=1。',`model stage ${i}`);if(i)norm(`norm${i}`,out);act(`act${i}`,'LeakyReLU');ch=out;sz/=2}
    conv('conv6',256,256,4,1,1);norm('norm6',256);act('act6','LeakyReLU');conv('conv7',256,1,4,1,1);add('out','Patch score','tensor','1 × 1 × 1','LSGAN 分数；无 Sigmoid。','model output')
  } else {
    add('input',node.kind==='input'?'输入影像':'生成影像','image','1 × 192 × 192',node.description,node.kind==='input'?'input':'output',undefined,null)
    if(node.kind==='input')add('batch','切片 batch','tensor','B × 1 × 192 × 192','3D 数据在入口被展成二维切片，网络使用 Conv2d。','reshape')
    else {add('mask','背景 mask','multiply','1 × 192 × 192','m = (first input > −1)，使用 2((x+1)/2·m)−1 保留背景。','mask');add('output','模态输出','image','1 × 192 × 192','模型输出，不是浏览器实际推理结果。','output_img')}
  }
  return {title,parts,edges,tensor:scope==='se'?dims(c,s):node.shape,note}
}

function layerAnatomy(l:number,observed:string[]):AnatomyGraph {
  const topology=modelTopology(),selected=new Set<string>()
  for(const [id] of topology.positions){const n=NODE_MAP.get(id)!;if(n.l===l&&(!n.mod||observed.includes(n.mod))&&!['input','stem','down','output','image'].includes(n.kind))selected.add(id)}
  if(l<7){selected.add(`posterior-${l+1}`);selected.add(`sample-${l+1}`);if(l===6)selected.add('lift')}
  if(l>1)selected.add(l===7?'lift':`up-${l-1}`)
  else for(const m of MODALITIES){selected.add(`${m.id}-output`);selected.add(`${m.id}-image`)}
  const parts:AnatomyPart[]=[...selected].map(id=>{const n=NODE_MAP.get(id)!,p=topology.positions.get(id)!;return {id,title:n.title,glyph:glyphFor(n.kind),shape:n.shape,detail:n.description,sourceName:id,child:id,mod:n.mod,role:n.kind==='encoder'||n.l>l?'input':n.l<l||['output','image'].includes(n.kind)?'output':undefined,position:[p[0],p[1]-levelY(l),p[2]]}})
  const edges=topology.links.filter(e=>selected.has(e.from)&&selected.has(e.to)).map(e=>({from:e.from,to:e.to,label:e.label}))
  return {title:`z${l} · 三维层级推导`,parts,edges,tensor:NODE_MAP.get(`sample-${l}`)!.shape,note:'原位展示：上一级后验 → 采样 → 上采样 → Decoder → 生成特征。生成特征分为先验头与各模态 Concat / Q 两路，残差专家和先验在 PoE 汇合，再采样并进入下一级。'}
}

export function anatomyFor(id:string,observed:string[]=MODALITIES.map(m=>m.id)):AnatomyGraph {
  const graph=rawAnatomy(id,observed)
  if(id.startsWith('layer-'))return graph
  const [root,scope]=id.split('/'),node=NODE_MAP.get(root)!,l=node.l,mod=node.mod
  const topology=modelTopology()
  let inputs=topology.links.filter(e=>e.to===root).map(e=>({id:e.from,to:graph.parts.some(p=>p.id==='input')?'input':graph.parts[0].id,label:e.label}))
  let outputs=topology.links.filter(e=>e.from===root).map(e=>e.to)
  if(node.kind==='expert'&&l<7)inputs=[{id:`${mod}-encoder-${l}`,to:'input',label:'本模态 skip'},{id:`feature-${l}`,to:'condition',label:'中央生成特征 gₗ'}]
  if(node.kind==='concat')inputs=[{id:`${mod}-encoder-${l}`,to:'input',label:'本模态 skip'},{id:`feature-${l}`,to:'condition',label:'中央生成特征 gₗ'}]
  if(node.kind==='poe')inputs=[{id:`prior-${l}`,to:'prior',label:'条件先验'},...observed.map(m=>({id:`${m}-factor-${l}`,to:'factors',label:'残差专家参数'}))]
  if(node.kind==='sample')inputs=[{id:`posterior-${l}`,to:'q',label:'融合后验分布'}]
  if(node.kind==='resnet'){const index=Number(root.split('-').at(-1));inputs=[{id:index===1?'sample-1':`${mod}-resnet-${index-1}`,to:'input',label:index===1?'共享 z₁':'前一残差块'}];outputs=[index===6?`${mod}-output`:`${mod}-resnet-${index+1}`]}
  if(node.kind==='prior'&&l<7)inputs=[{id:`prior-head-${l}`,to:'zeros',label:'μp 参数'},{id:`prior-head-${l}`,to:'ones',label:'sp 参数'}]
  if(node.kind==='factor'||node.kind==='posterior')inputs=inputs.flatMap(v=>[{...v,to:'mu',label:'均值 μ'},{...v,to:'scale',label:node.kind==='factor'?'log-scale a':'标准差 T·s'}])
  if(scope==='se'){inputs=[{id:root,to:'input',label:node.kind==='decoder'?'来自 act_2':'来自 conv_1'}];outputs=[root]}
  const terminal=graph.parts.filter(p=>!graph.edges.some(e=>e.from===p.id)).map(p=>p.id)
  for(const [i,input] of inputs.entries()){
    const n=NODE_MAP.get(input.id);if(!n)continue
    const key=`external-in-${i}`
    graph.parts.push({id:key,title:`输入 · ${n.title}`,glyph:glyphFor(n.kind),shape:scope==='se'?graph.parts.find(p=>p.id==='input')!.shape:n.shape,detail:`${input.label}。来源：${n.title}；沿箭头进入当前模块。`,sourceName:input.id,child:input.id,role:'input',mod:n.mod})
    graph.edges.push({from:key,to:input.to,label:input.label})
  }
  for(const [i,out] of outputs.entries()){
    const n=NODE_MAP.get(out);if(!n)continue
    const key=`external-out-${i}`,finalResnet=node.kind==='resnet'&&root.endsWith('-6')
    graph.parts.push({id:key,title:finalResnet?`输出 → ${mod} · convt2 (7×7)`:`输出 → ${n.title}`,glyph:finalResnet?'conv':glyphFor(n.kind),shape:scope==='se'?graph.parts.find(p=>p.id==='input')!.shape:finalResnet?'8 × 192 × 192':n.shape,detail:`当前模块输出流向 ${n.title}。${scope==='se'?'返回原块，继续投影或残差相加。':finalResnet?'具体进入 convt2 (8→8, 7×7)，再经 IN、LeakyReLU、convt3 与 Tanh。':''}`,sourceName:finalResnet?`final_blocks.${mod}.convt2`:out,child:out,role:'output',mod:n.mod})
    for(const from of terminal)graph.edges.push({from,to:key,label:scope==='se'?'通道门控特征':'输出张量'})
  }
  return graph
}
