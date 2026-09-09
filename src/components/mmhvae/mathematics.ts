import type { AnatomyGraph, AnatomyPart } from './anatomy'

export type MathKind = 'convolution'|'depthwise'|'linear'|'normalization'|'activation'|'pooling'|'concatenation'|'split'|'distribution'|'fusion'|'sampling'|'interpolation'|'addition'|'multiplication'|'reshape'|'broadcast'|'padding'|'tensor'|'mask'|'attention'|'composition'
export interface MathStage {kind:MathKind; operation:string; stage:number; kernel:number; stride:number; padding:number; weightNorm:boolean;bias:boolean;reflect:boolean}
export interface MathStep {title:string;formula:string;explanation:string}
export interface MathSpec {kind:MathKind;operation:string;steps:MathStep[];control:string;legend:string;formula:string;source:string;kernel:number;stride:number;padding:number;weightNorm:boolean;bias:boolean;reflect:boolean}
const step=(title:string,formula:string,explanation:string):MathStep=>({title,formula,explanation})
const R=String.raw

export function classifyMath(p:AnatomyPart):{kind:MathKind;operation:string}{
 const text=`${p.id} ${p.title} ${p.sourceName} ${p.detail}`.toLowerCase()
 if(p.id==='mask')return {kind:'mask',operation:'background'}
 if(p.id==='weighted_mu')return {kind:'fusion',operation:'weighted-mean'}
 if(p.id==='normalize')return {kind:'fusion',operation:'normalize'}
 if(p.glyph==='tensor'||p.glyph==='image'){
  if(/reflectionpad|镜像填充/.test(text))return {kind:'padding',operation:'reflect'}
  if(/broadcast|广播/.test(text))return {kind:'broadcast',operation:'spatial'}
  if(/view|flatten|reshape|展成/.test(text))return {kind:'reshape',operation:p.id==='flatten'?'flatten':'reshape'}
  return {kind:'tensor',operation:/zeros|μ = 0/.test(text)?'zeros':/ones|s = 1/.test(text)?'ones':p.glyph==='image'?'image':'feature'}
 }
 if(p.glyph==='activation')return {kind:'activation',operation:/clamp/.test(text)?/exp/.test(text)?'clamp-exp':'clamp':/leaky/.test(text)?'leaky':/relu/.test(text)?'relu':/sigmoid/.test(text)?'sigmoid':/tanh/.test(text)?'tanh':'silu'}
 const types:Record<string,MathKind>={conv:'convolution',depthwise:'depthwise',linear:'linear',norm:'normalization',pool:'pooling',concat:'concatenation',split:'split',gaussian:'distribution',poe:'fusion',sample:'sampling',up:'interpolation',sum:'addition',multiply:'multiplication',se:'attention',network:'composition'}
 return {kind:types[p.glyph],operation:p.id==='mean'&&p.glyph==='sum'?'sample-mean':p.id==='scale'&&p.glyph==='multiply'?'sample-scale':p.glyph==='gaussian'&&/残差|res_params/.test(text)?'residual':p.glyph==='gaussian'&&/后验|full|qₗ/.test(text)?'posterior':p.glyph==='sample'?'noise':p.glyph==='poe'?'weight':p.glyph}
}

export function mathSpec(p:AnatomyPart,formula:string):MathSpec{
 const {kind,operation}=classifyMath(p),text=`${p.title} ${p.detail}`,kernel=Number(text.match(/(?:kernel_size=|kernel=|Conv |Depthwise )(\d+)/)?.[1]??(p.glyph==='depthwise'?5:3)),stride=Number(text.match(/stride=(\d+)/)?.[1]??1),padding=Number(text.match(/padding=(\d+)/)?.[1]??Math.floor(kernel/2)),weightNorm=/WeightNorm|weight_norm/.test(text)
 let steps:MathStep[]=[],control='观察位置',legend='水平面为 H×W，竖向层片为通道 C。颜色区分数值正负；白色单元是当前观察位置。'
 switch(kind){
 case 'convolution':case 'depthwise':
  control='卷积窗口位置';steps=[step('滑动感受野',R`\mathcal P_{h,w}=X[:,sh-p:sh-p+k,sw-p:sw-p+k]`,`核 ${kernel}×${kernel}，步幅 ${stride}，填充 ${padding}。白色框随位置移动；图形使用缩小的示例网格，模型真实尺寸保留在标题。`),step(kind==='depthwise'?'每个通道独立相乘':'核权重与输入逐项相乘',kind==='depthwise'?R`P_{c,u,v}=W_{c,u,v}X_{c,sh+u-p,sw+v-p}`:R`P_{o,c,u,v}=W_{o,c,u,v}X_{c,sh+u-p,sw+v-p}`,kind==='depthwise'?'每层仅连接自己的卷积核；groups=C，不跨通道求和。':'核不翻转，对应 PyTorch 的互相关运算。不同输入通道各有权重，随后向同一输出通道汇合。'),step('局部归约写入输出',kind==='depthwise'?R`Y_{c,h,w}=b_c+\sum_{u,v}P_{c,u,v}`:R`Y_{o,h,w}=b_o+\sum_{c,u,v}P_{o,c,u,v}`,'粒子沿连线汇入输出单元；各输出通道有独立卷积核。颜色和柱高来自示例数值的真实乘加结果。')];
  if(weightNorm)steps.splice(1,0,step('权重方向与尺度分离',R`W_o=g_o\frac{V_o}{\|V_o\|_2}`,'先将每个输出通道的权重方向归一化，再乘可学习长度 g；接着进行卷积。'))
  break
 case 'linear':control='观察输出神经元';steps=[step('输入向量与权重矩阵',R`x\in\mathbb R^n,\quad W\in\mathbb R^{m\times n}`,'水平矩阵的行对应输出、列对应输入；当前输出行被高亮，逐项读取同一输入向量。'),step(weightNorm?'归一化权重后点积':'逐项乘积与点积',weightNorm?R`W_i=g_iV_i/\|V_i\|_2,\quad q_{ij}=W_{ij}x_j`:R`q_{ij}=W_{ij}x_j`,'连线对应实际依赖；每个输出神经元连接全部输入坐标，不执行空间卷积。'),step('加偏置并组成新向量',R`y_i=\sum_jq_{ij}+b_i`,'各行求和形成输出向量。bottleneck 的空间恢复在后续 view 中进行；Linear 本身只变换特征维。')];break
 case 'normalization':control='观察通道';steps=[step('每个样本、通道独立统计',R`\mu_{bc}=\frac1{HW}\sum_{h,w}X_{bchw}`,'高亮层片中的所有空间位置汇入一个均值；统计不跨样本，也不跨通道。'),step('中心化与方差',R`v_{bc}=\frac1{HW}\sum_{h,w}(X_{bchw}-\mu_{bc})^2`,'先减均值，再逐元素平方。柱高展示非负的偏差平方；对它们求平均得到总体方差（unbiased=False）。'),step('缩放至统一响应尺度',R`Y_{bchw}=\frac{X_{bchw}-\mu_{bc}}{\sqrt{v_{bc}+10^{-5}}}`,'加入 eps 防止除零。此模型默认没有 affine 参数、不维护 running stats；不是 BatchNorm。')];break
 case 'activation':control=operation.includes('clamp')?'输入值 x / a（−20 至 20）':'输入值 x（−4 至 4）';legend='横轴为输入 x，竖轴为输出 f(x)，纵深方向分隔独立通道；点沿曲线移动，不是数据在空间中旋转。';steps=[step('独立读取每个张量元素',R`x=X_{b,c,h,w}`,'非线性作用于每个元素，保留 batch、通道和空间索引。选取的输入值同步出现在曲线和结果上。'),step('沿真实函数曲线映射',activationFormula(operation),'曲线由当前算子的真实函数求值绘制。轨迹投影同时展示输入与输出；负半轴行为与 SiLU、ReLU 等算子各自一致。'),step('回写同一张量位置',R`Y_{b,c,h,w}=f(X_{b,c,h,w})`,'数值改变，张量尺寸不变。soft-clamp 保留平滑梯度；含 exp 的先验尺度头会把结果转为正标准差。')];break
 case 'pooling':control='观察通道';steps=[step('收集一个通道的空间响应',R`X_c\in\mathbb R^{H\times W}`,'每层为一个通道，保留通道之间的独立性。'),step('空间求和再除以像素数',R`u_c=\frac{\sum_{h,w}X_{c,h,w}}{HW}`,'全部空间单元的连线汇入单个通道描述值；示例直接计算平均值。'),step('形成通道描述向量',R`u\in\mathbb R^{B\times C}`,'SE 随后的 Linear 使用每个通道的全局描述；这不是最大池化，也不混合不同通道。')];break
 case 'concatenation':control='观察拼接通道';steps=[step('两路特征保留空间对齐',R`A\in\mathbb R^{B\times C_a\times H\times W},\quad B\in\mathbb R^{B\times C_b\times H\times W}`,'两种颜色分别表示生成条件和模态 skip，来自边界外的两路输入。'),step('沿通道轴连接层片',R`Y=\operatorname{cat}(A,B;\mathrm{dim}=1)`,'层片沿竖直的通道轴排列；元素不做相加、不改变空间坐标。'),step('按通道区间寻址',R`Y_c=\begin{cases}A_c&c<C_a\\B_{c-C_a}&c\ge C_a\end{cases}`,'观察位置对应同一个元素在拼接前后的地址；新通道数为两路之和。')];break
 case 'split':control='观察参数通道';steps=[step('读取概率头的两组通道',R`h\in\mathbb R^{B\times2C_z\times H\times W}`,'前一半和后一半通道以两种颜色区分；来源是概率预测头。'),step('按通道维拆分',R`[\mu,a]=\operatorname{chunk}(h,2;\mathrm{dim}=1)`,'两组层片沿不同路径分开，原始元素值不变。这是参数拆分，不是抽样。'),step('均值与 log-scale 分路',R`\mu=h[:,:C_z],\quad a=h[:,C_z:]`,'a 是 log-scale，尚不是方差或标准差。后续 soft-clamp、exp 或融合由其真实下游执行。')];break
 case 'distribution':control='分布宽度 s';legend='曲面竖直高度为二维概率密度的 8 倍，仅为看清曲率；密度公式保持归一化。两条水平轴为独立坐标，示例不是训练后验。';steps=[step('参数给出中心与尺度',operation==='residual'?R`s=\exp(a),\quad(\mu,a)\in\mathrm{res\_params}`:R`(\mu,s),\quad s>0`,'十字标出均值，尺度控制宽度。残差专家在代码中保存 loc 与 logscale_res；曲面仅解释它代表的因子。'),step('构建对角 Gaussian 密度',R`q(z)=\prod_i\frac{\exp(-(z_i-\mu_i)^2/(2s_i^2))}{\sqrt{2\pi}s_i}`,'展示两个坐标的可旋转密度曲面和水平等高线。独立坐标无协方差交叉项。'),step('概率参数交给融合或采样',R`\operatorname{Var}(z_i)=s_i^2`,'Normal 的第二个参数是标准差 s；改变宽度时密度峰值也改变，并保持概率积分为 1。此阶段不生成样本。')];break
 case 'fusion':control='示例专家的尺度';legend='蓝色为先验，暖色为模态专家；条高是源码倒数 scale 权重，曲面是由这些权重得到的正态分布。';steps=[step('读取先验与残差专家',R`w_p=1/s_p,\quad w_j=e^{-a_j}`,'固定先验示例 μp=0、sp=1；两个观测专家示例均值 −0.8 和 1.2。只累加当前可用模态。'),step('累加权重和均值贡献',R`w=w_p+\sum_jw_j,\quad m=w_p\mu_p+\sum_jw_j\mu_j`,'各通道分别计算。权重条与贡献条来自相同示例数值；源码按 1/s，理论 Gaussian PoE 则按 1/s²。'),step('归一化并建立后验',R`\mu=m/w,\quad s=1/w,\quad q=\mathcal N(\mu,(Ts)^2)`,'改变一个专家的尺度，可观察其权重、融合中心和宽度同步变化。温度只缩放标准差。')];break
 case 'sampling':control='观察噪声样本';legend='固定种子的正态噪声样本用于可复现演示；点云移动展示重参数化，不能解释为模型真实推理。';steps=[step('标准正态噪声',R`\epsilon\sim\mathcal N(0,I)`,'点云由固定种子 Box–Muller 示例产生，便于暂停和重复对比。每一坐标独立抽样。'),step('缩放噪声到分布尺度',R`\eta=T s\odot\epsilon`,'中心仍在零；尺度与温度改变点云的离散程度。'),step('平移到均值，保留梯度路径',R`z=\mu+\eta,\quad\partial z/\partial\mu=1,\quad\partial z/\partial s=T\epsilon`,'白色轨迹连接同一个噪声点在缩放和平移后的坐标；随机性与可微参数路径分离。')];break
 case 'interpolation':control='目标网格采样位置';steps=[step('目标坐标映射回输入',R`u=(u'+1/2)/2-1/2,\quad v=(v'+1/2)/2-1/2`,'×2 放大，align_corners=False 使用半像素中心映射；边界坐标采用边界值处理。'),step('四邻点双线性加权',R`Y=(1-\alpha)(1-\beta)X_{00}+\alpha(1-\beta)X_{10}+(1-\alpha)\beta X_{01}+\alpha\beta X_{11}`,'四条连线的贡献由横纵距离决定。移动目标位置，四邻点、权重和插值结果一起更新。'),step('逐通道写入放大网格',R`C\times H\times W\longrightarrow C\times2H\times2W`,'插值没有可学习卷积核，通道数不变；后续 3×3 卷积是另一个可独立观察的原子模块。')];break
 case 'addition':case 'multiplication':case 'mask':
  control='观察元素位置';steps=[step(kind==='mask'?'影像与背景掩码':'两路对齐的操作数',kind==='mask'?R`m=\mathbf1[x_{\mathrm{first}}> -1]`:R`A,B\in\mathbb R^{C\times H\times W}`,'颜色区分两路输入；细线连接相同通道和空间位置。门控向量以 C×1×1 广播。'),step(kind==='addition'?'逐位置相加':kind==='mask'?'缩放后应用掩码':'逐位置相乘',kind==='addition'?operation==='sample-mean'?R`z_i=\mu_i+\eta_i`:R`Y_i=A_i+B_i`:kind==='mask'?R`Y_i=2((X_i+1)/2\cdot m_i)-1`:operation==='sample-scale'?R`\eta_i=Ts_i\epsilon_i`:R`Y_i=A_iB_i`,'展示实际示例操作数及计算结果，不跨位置求和。残差加法保留直接梯度路径；乘法以对应权重调节响应。'),step('回写与输入相同的结构',kind==='mask'?R`m_i=0\Rightarrow Y_i=-1,\quad m_i=1\Rightarrow Y_i=X_i`:R`\operatorname{shape}(Y)=\operatorname{shape}(A)`,'层片保持原有空间对齐；掩码为零时输出背景值 −1，而不是零。')];break
 case 'padding':control='观察边界位置';steps=[step('原始空间边界',R`X\in\mathbb R^{C\times H\times W}`,'中央单元是原特征；外围空位属于新边界，不改变通道数。'),step('镜像索引回原图',R`X_{-1,j}=X_{1,j},\quad X_{H,j}=X_{H-2,j}`,'边界外位置映射到镜像位置，端点不重复。白色连线显示原始元素与镜像副本的对应。'),step('交给无填充卷积',R`H\times W\to(H+2p)\times(W+2p)`,'先显式 ReflectionPad，再使用 padding=0 的卷积；影像尾部 reflect 卷积采用同样边界含义。')];break
 case 'reshape':case 'broadcast':case 'tensor':
  control='观察元素索引';steps=[step(kind==='broadcast'?'通道门控向量':'张量的有序元素',kind==='tensor'&&operation==='zeros'?R`X_i=0`:kind==='tensor'&&operation==='ones'?R`X_i=1`:R`X[b,c,h,w]`,'每个小单元是数值，水平两轴是空间，竖直层片是通道。只显示少量元素；完整真实尺寸标在模型数据结构中。'),step(kind==='reshape'?'线性索引保持不变':kind==='broadcast'?'单值沿空间维广播':'定位同一元素',kind==='reshape'?R`i=((bC+c)H+h)W+w`:kind==='broadcast'?R`G[b,c,h,w]=g[b,c,0,0]`:R`i\leftrightarrow(b,c,h,w)`,kind==='reshape'?'同一编号单元沿三维轨迹排成向量或恢复为层片，元素值与顺序不改变。':kind==='broadcast'?'一个通道权重连接所有空间位置；广播本身不生成新的可学习参数。':'高亮单元展示 batch、channel、row、column 的索引关系；纯数据张量没有额外隐藏网络。'),step(kind==='reshape'?'重组目标维度':kind==='broadcast'?'空间对齐的门控':'沿连接传递原数据',kind==='reshape'?R`\prod\mathrm{shape}_{in}=\prod\mathrm{shape}_{out}`:kind==='broadcast'?R`C\times1\times1\rightsquigarrow C\times H\times W`:R`Y[b,c,h,w]=X[b,c,h,w]`,'维度解释变化与数值运算明确区分；原始数值仍可沿高亮索引追踪。')];break
 case 'attention':case 'composition':control='观察通道';steps=[step('读取分组特征',R`X\in\mathbb R^{B\times C\times H\times W}`,'数据按通道组织，输入来自当前模块外部。'),step('按依赖组织映射',kind==='attention'?R`g=\sigma(W_2\operatorname{ReLU}(W_1\operatorname{GAP}(X)))`:R`Y=f_n\circ\cdots\circ f_1(X)`,'矩阵行列与连接表达函数组合，而非单一抽象图标；可返回上一级观察每个已列出的基础算子。'),step('形成输出响应',kind==='attention'?R`Y_c=g_cX_c`:R`Y=f(X)`,'层片展示输出数值结构。')];break
 }
 if(kind==='fusion'){
  if(operation==='weight')steps=[step('读取尺度参数',R`(s_p,\{a_j\})`,'先验提供标准差，残差专家提供 log-scale。分布曲面仅作为操作数的来源语义，不在此处构建新后验。'),step('转换为倒数尺度权重',R`w_p=1/s_p,\quad w_j=\exp(-a_j)`,'各条柱分别代表同一坐标的先验和专家权重。改变专家尺度可直接看到倒数关系。'),step('逐坐标累加权重',R`w=w_p+\sum_jw_j`,'此算子输出总权重 w，不输出后验分布；均值累加、归一化和 Normal 分别属于其下游模块。')]
  if(operation==='weighted-mean')steps=[step('读取各分布均值与尺度',R`(\mu_p,s_p),\quad(\mu_j,a_j)`,'参数来自先验与观测专家。蓝色对应先验，另外两条为教学示例专家。'),step('形成加权均值贡献',R`c_p=\mu_p/s_p,\quad c_j=\mu_j\exp(-a_j)`,'每个均值乘自己的倒数尺度；贡献可以为正或负，颜色和有符号柱高区分它们。'),step('累加贡献，等待归一化',R`m=c_p+\sum_jc_j`,'输出未归一化的加权和 m，而不是最终均值。下一模块再计算 μ=m/w。')]
  if(operation==='normalize')steps=[step('读取权重和与加权均值和',R`(w,m)`,'两路输入来自独立的累加算子；小规模示例按公开代码的倒数 scale 权重生成。'),step('分别执行除法和倒数',R`\mu=m/w,\quad s=1/w`,'共享同一个总权重 w。两组结果层片分别代表均值和标准差，不能把 s 当作方差。'),step('送出两组分布参数',R`(\mu,s)\longrightarrow\operatorname{Normal}(\mu,Ts)`,'这里只送出参数；温度缩放与正态分布构造由边界外的后验分布模块完成。')]
 }
 if(kind==='sampling'){
  control='观察噪声样本';legend='固定种子的 Box–Muller 数值例展示标准正态噪声的数学构造，不声称 PyTorch 内部随机数内核采用该算法。'
  steps=[step('独立均匀随机数示例',R`u_1,u_2\sim U(0,1)`,'用于解释正态噪声来源的确定性示例。源码 Normal.rsample 使用标准正态噪声；此处不假定底层随机数内核的具体算法。'),step('变换为正态坐标',R`\epsilon_1=\sqrt{-2\log u_1}\cos(2\pi u_2),\quad\epsilon_2=\sqrt{-2\log u_1}\sin(2\pi u_2)`,'Box–Muller 示例把均匀点转换为正态点，三维轨迹连接同一编号样本。'),step('组织为标准正态噪声张量',R`\epsilon_i\sim\mathcal N(0,1),\quad\epsilon\perp(\mu,s)`,'这里只输出噪声。乘以 T·s 和加上 μ 仍由边界外的乘法、加法节点展示，从而保留原算子的职责边界。')]
 }
 return {kind,operation,steps,control,legend,formula,source:p.sourceName,kernel,stride,padding,weightNorm,bias:!text.includes('bias=False'),reflect:text.includes('reflect')}
}

export function mathematicalGraph(part:AnatomyPart,neighbors:AnatomyPart[],edges:AnatomyGraph['edges'],title:string,formula:string):AnatomyGraph{
 const spec=mathSpec(part,formula)
 const internal=spec.steps.map((s,i):AnatomyPart=>({id:`math-${i}`,title:s.title,glyph:part.glyph,shape:s.formula,detail:s.explanation,sourceName:part.sourceName,math:{kind:spec.kind,operation:spec.operation,stage:i,kernel:spec.kernel,stride:spec.stride,padding:spec.padding,weightNorm:spec.weightNorm,bias:spec.bias,reflect:spec.reflect}}))
 const links=internal.slice(1).map((p,i)=>({from:internal[i].id,to:p.id,label:'计算依赖'}))
 for(const e of edges)links.push({...e,from:e.from===part.id?internal.at(-1)!.id:e.from,to:e.to===part.id?internal[0].id:e.to,label:e.label??'张量传递'})
 return {layout:'mathematics',mathematics:spec,title,tensor:part.shape,note:'数学内部视图 · 小规模确定性示例，未加载训练权重。每个空间单元、曲面和连线对应实际运算；模型真实尺寸与源码参数另列。',parts:[...internal,...neighbors],edges:links}
}

export const activationFormula=(op:string)=>({silu:R`f(x)=x/(1+e^{-x})`,relu:R`f(x)=\max(0,x)`,leaky:R`f(x)=\max(0,x)+0.2\min(0,x)`,sigmoid:R`f(x)=(1+e^{-x})^{-1}`,tanh:R`f(x)=\tanh(x)`,clamp:R`f(x)=10\tanh(x/10)`,'clamp-exp':R`f(a)=\exp(10\tanh(a/10))`}[op]??R`f(x)=x`)
export const activateValue=(x:number,op:string)=>op==='relu'?Math.max(0,x):op==='leaky'?Math.max(0,x)+.2*Math.min(0,x):op==='sigmoid'?1/(1+Math.exp(-x)):op==='tanh'?Math.tanh(x):op==='clamp'?10*Math.tanh(x/10):op==='clamp-exp'?Math.exp(10*Math.tanh(x/10)):x/(1+Math.exp(-x))
export const demoValues=Array.from({length:48},(_,i)=>Math.sin(i*1.7)*.8+Math.cos(i*.43)*.4)
export const mean=(a:number[])=>a.reduce((s,x)=>s+x,0)/a.length
export const normalizeValues=(a:number[])=>{const mu=mean(a),v=mean(a.map(x=>(x-mu)**2));return a.map(x=>(x-mu)/Math.sqrt(v+1e-5))}
export const fusionValues=(s:number,temp=.5)=>{const weights=[1,1/s,1/.7],mus=[0,-.8,1.2],w=weights.reduce((a,b)=>a+b),m=weights.reduce((a,b,i)=>a+b*mus[i],0);return {weights,mus,mu:m/w,scale:temp/w,baseScale:1/w,contributions:weights.map((v,i)=>v*mus[i])}}
export const bilinear=(grid:number[],n:number,x:number,y:number)=>{x=Math.max(0,Math.min(n-1,x));y=Math.max(0,Math.min(n-1,y));const x0=Math.floor(x),y0=Math.floor(y),a=x-x0,b=y-y0;const ids=[y0*n+x0,y0*n+Math.min(n-1,x0+1),Math.min(n-1,y0+1)*n+x0,Math.min(n-1,y0+1)*n+Math.min(n-1,x0+1)],weights=[(1-a)*(1-b),a*(1-b),(1-a)*b,a*b];return {ids,weights,value:ids.reduce((s,id,i)=>s+grid[id]*weights[i],0)}}
export function convolutionDemo(spec:Pick<MathSpec,'kernel'|'kind'|'weightNorm'|'stride'|'padding'|'bias'|'reflect'>){
 const k=spec.kernel,n=k+3,channels=spec.kind==='depthwise'?3:2,values=Array.from({length:n*n*channels},(_,i)=>demoValues[i%demoValues.length]),rawWeights=Array.from({length:k*k*channels},(_,i)=>Math.cos(i*1.2)*.2),weights=spec.weightNorm?rawWeights.map(v=>v/Math.hypot(...rawWeights)):rawWeights,outN=Math.floor((n+2*spec.padding-k)/spec.stride)+1
 const reflect=(i:number)=>i<0?-i:i>=n?2*n-2-i:i
 const read=(c:number,y:number,x:number)=>spec.reflect?values[c*n*n+reflect(y)*n+reflect(x)]:y<0||x<0||y>=n||x>=n?0:values[c*n*n+y*n+x]
 const output=Array.from({length:outN*outN*(spec.kind==='depthwise'?channels:1)},(_,i)=>{const outChannel=Math.floor(i/(outN*outN)),oy=Math.floor(i/outN)%outN,ox=i%outN;let result=spec.bias?.1:0;for(let c=0;c<channels;c++)if(spec.kind!=='depthwise'||outChannel===c)for(let y=0;y<k;y++)for(let x=0;x<k;x++)result+=read(c,oy*spec.stride+y-spec.padding,ox*spec.stride+x-spec.padding)*weights[c*k*k+y*k+x];return result})
 return {k,n,channels,values,weights,outN,output}
}
const random=(()=>{let seed=923417;return()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed+.5)/4294967296}})()
export const uniformSamples=Array.from({length:80},()=>[random(),random(),random(),random()])
export const normalSamples=uniformSamples.map(([u,v,u2,v2])=>[Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v),Math.sqrt(-2*Math.log(u))*Math.sin(2*Math.PI*v),Math.sqrt(-2*Math.log(u2))*Math.cos(2*Math.PI*v2)])
