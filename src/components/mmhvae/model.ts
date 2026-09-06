export const COMMIT = '31a988e77adc42ff786bcbed2f066be051c47c66'
export const source = (file: string, line: number) => `https://github.com/ReubenDo/MMHVAE/blob/${COMMIT}/${file}#L${line}`
export const MODALITIES = [
  { id: 'us', label: 'iUS', color: '#d8ff45', x: -9, z: 2.8 },
  { id: 't2', label: 'T2', color: '#78b8ff', x: -5.6, z: -3.2 },
  { id: 'cet1', label: 'ceT1', color: '#ff9e7a', x: 5.6, z: -3.2 },
  { id: 'flair', label: 'FLAIR', color: '#c6a6ff', x: 9, z: 2.8 },
] as const
export const LEVELS = [
  { l: 7, size: 1, channels: 256, feature: 128, encoderSize: 3 },
  { l: 6, size: 6, channels: 64, feature: 128, encoderSize: 6 },
  { l: 5, size: 12, channels: 64, feature: 128, encoderSize: 12 },
  { l: 4, size: 24, channels: 64, feature: 128, encoderSize: 24 },
  { l: 3, size: 48, channels: 32, feature: 64, encoderSize: 48 },
  { l: 2, size: 96, channels: 16, feature: 32, encoderSize: 96 },
  { l: 1, size: 192, channels: 8, feature: 16, encoderSize: 192 },
]
export type Kind = 'input' | 'stem' | 'encoder' | 'down' | 'expert' | 'prior' | 'poe' | 'sample' | 'up' | 'decoder' | 'lift' | 'output' | 'resnet' | 'image' | 'discriminator' | 'feature' | 'concat' | 'priorhead' | 'factor' | 'posterior'
export interface ModelNode {
  id: string; kind: Kind; title: string; short: string; l: number; mod?: string
  shape: string; description: string; operations: string[]; formula: string
  file: string; line: number; shared?: boolean
}
const shape = (c: number, s: number) => `${c} × ${s} × ${s}`
export function createNodes(): ModelNode[] {
  const nodes: ModelNode[] = []
  const add = (node: ModelNode) => nodes.push(node)
  for (const m of MODALITIES) {
    const base = { mod: m.id, file: 'network/mhvae.py', l: 1 }
    add({ ...base, id: `${m.id}-input`, kind: 'input', title: `${m.label} · 输入切片`, short: m.label, shape: shape(1,192), description: '模型逐张处理二维切片。体数据在训练入口被展开为切片批次；三维展示表达网络拓扑，不代表 Conv3d。取消观测后整座编码塔退出前向计算。', operations: ['体数据 → 二维切片批次', '单通道输入 · 192 × 192', '已观测模态进入 create_encodings'], formula: String.raw`x_j\in\mathbb{R}^{B\times1\times192\times192}`, line: 165 })
    add({ ...base, id: `${m.id}-stem`, kind: 'stem', title: `${m.label} · 输入卷积`, short: 'Conv 3×3', shape: '1 → 16 · 192²', description: '每个模态独立拥有 first_conv，保留空间尺寸并把灰度输入映射为 16 通道特征。', operations: ['Conv2d · 1 → 16', 'kernel 3 · stride 1 · padding 1', 'bias = True'], formula: String.raw`h^j_0=\operatorname{Conv}_{3\times3}(x_j)`, line: 56 })
    for (const v of LEVELS) {
      const l = v.l, c = v.feature, s = v.encoderSize
      add({ ...base, l, id: `${m.id}-encoder-${l}`, kind: 'encoder', title: `${m.label} · BlockEncoder ${l}`, short: `E${l} · ${c}ch`, shape: shape(c,s), description: `第 ${l} 级模态独立编码块。${l < 7 ? '输出保存为 skip，供对应层 BlockQ 与生成特征拼接。' : '最粗 3×3 特征被展平后送入共享线性 Gaussian head。'}默认启用残差和 SE，归一化实际为 InstanceNorm2d。`, operations: [`InstanceNorm(${c}) → SiLU → Conv 3×3`, `InstanceNorm(${c}) → SiLU → Conv 3×3`, `SE: GAP → Linear ${c}→${Math.max(c/16,4)} → ReLU`, `Linear → ${c} → Sigmoid → 通道乘法`, '残差相加 · out + input'], formula: String.raw`h^j_l=h+\operatorname{SE}\!\left(C_3\,s\,\mathrm{IN}\!\left(C_3\,s\,\mathrm{IN}(h)\right)\right)`, file: 'network/blocks.py', line: 54 })
      if (l < 7) add({ ...base, l, id: `${m.id}-down-${l}`, kind: 'down', title: `${m.label} · 下采样 ${l} → ${l+1}`, short: '↓ Conv s2', shape: `${c} → ${Math.min(2*c,128)} · ${s}² → ${s/2}²`, description: '下采样是独立的可学习步长卷积，位于两个 BlockEncoder 之间。通道数逐级翻倍，最多 128。', operations: [`Conv2d · ${c} → ${Math.min(2*c,128)}`, 'kernel 3 · stride 2 · padding 1', '先缓存 skip，再下采样'], formula: String.raw`h_{l+1}^{\mathrm{in}}=\operatorname{Conv}_{3\times3,s=2}(h_l)`, line: 78 })
      add({ ...base, l, id: `${m.id}-expert-${l}`, kind: 'expert', title: `${m.label} · Gaussian head ${l}`, short: l === 7 ? 'FC · μ / log s' : `Q${l} · μ / log s`, shared: true, shape: l === 7 ? '1152 → 512 → (256, 256)' : `${2*c} → ${c} → (${c/2}, ${c/2}) · ${s}²`, description: l === 7 ? '同一个 bottleneck_down 被用于每个观测模态。128×3×3 展平为 1152，经 weight-normalized Linear 输出 512，再沿通道分成均值与 log-scale。' : '将本模态 skip 与中央 BlockDecoder 特征沿通道拼接，进入 BlockQ。qz 是按层组织的 ModuleList：同一层在各模态间共享参数，但输入及 Gaussian 输出各不相同。', operations: l === 7 ? ['Flatten · 128×3×3 → 1152', 'WeightNorm Linear · 1152 → 512', 'chunk(2) → μ, log-scale', '两组参数均经 10 tanh(a/10)'] : [`Concat [top-down, skip] · ${2*c}ch`, `IN → SiLU → Conv 3×3 · ${2*c}→${c}`, `IN → SiLU → Conv 3×3 · ${c}→${c}`, 'SE → WeightNorm Conv 1×1 · bias=False', `chunk(2) → ${c/2}ch μ + ${c/2}ch log-scale`, '两组参数均经 soft_clamp'], formula: l === 7 ? String.raw`(\mu^j_7,a^j_7)=\operatorname{split}\!\left(W_{\rm down}\operatorname{vec}(h^j_7)\right)` : String.raw`(\mu^j_l,a^j_l)=\operatorname{split}\!\left(Q_l([g_l,h^j_l])\right)`, file: l===7?'network/blocks.py':'network/blocks.py', line: l===7?226:177 })
    }
    for (let i=1;i<=6;i++) add({ ...base, id: `${m.id}-resnet-${i}`, kind: 'resnet', title: `${m.label} · 输出 ResNet ${i}/6`, short: `R${i}`, shape: shape(8,192), description: '各模态拥有独立的输出解码器。源码默认配置六个 ResnetBlock；每一个保持 8 通道与 192² 尺寸，使用 ReLU，无 dropout。', operations: ['ReflectionPad(1) → Conv 3×3 → IN → ReLU', 'ReflectionPad(1) → Conv 3×3 → IN', '残差相加 · input + conv_block(input)'], formula: String.raw`r_{k+1}=r_k+F_k(r_k),\quad k=0,\ldots,5`, file: 'network/blocks.py', line: 242 })
    add({ ...base, id: `${m.id}-output`, kind: 'output', title: `${m.label} · BlockFinalImg`, short: `${m.label} decoder`, shape: '8 × 192² → 1 × 192²', description: '从共享 z₁ 分叉为四条模态独立输出路径。默认 nfeat_finalblock=8，与 z₁ 通道相同，因此不插入额外 1×1 投影。六个残差块之后是两个 7×7 卷积。', operations: ['ResnetBlock × 6 · 8ch', 'Conv 7×7 · 8→8 · reflect padding=3', 'InstanceNorm → LeakyReLU(0.2)', 'Conv 7×7 · 8→1 · reflect padding=3', 'Tanh → 背景 mask'], formula: String.raw`\hat x_j=G_j(z_1),\qquad \hat x_j\in[-1,1]^{B\times1\times192\times192}`, file: 'network/blocks.py', line: 299 })
    add({ ...base, id: `${m.id}-image`, kind: 'image', title: `${m.label} · 输出与背景掩码`, short: `x̂ ${m.label}`, shape: shape(1,192), description: 'forward 同时生成全部模态。界面的输出选择只高亮相应解码支路，不改变网络输出数量。背景掩码取输入字典中第一个模态的 x > −1。', operations: ['所有 final_blocks 同时运行', 'mask = (first observed input > −1)', '2 × ((output + 1)/2 × mask) − 1'], formula: String.raw`\hat x_j\leftarrow2\left(\frac{\hat x_j+1}{2}\odot m\right)-1`, line: 288 })
    add({ ...base, id: `${m.id}-discriminator`, kind: 'discriminator', title: `${m.label} · PatchGAN（训练）`, short: `D · ${m.label}`, shape: '1×192² → 1×1²', description: '每个模态单独配置 NLayerDiscriminator，ndf=64、n_layers=6。只参与训练；不在推理生成路径中。', operations: ['Conv 4×4 s2 · 1→64 → LeakyReLU', '5× Conv 4×4 s2 → IN → LeakyReLU', 'channels: 64 → 128 → 256 → 256 → 256 → 256', 'Conv 4×4 s1 · 256→256 → IN → LeakyReLU', 'Conv 4×4 s1 · 256→1 · 无 Sigmoid', 'LSGAN · 模态独立 real-image pool'], formula: String.raw`\mathcal L_{\rm adv}^{G}=\sum_j\mathbb E[(D_j(\hat x_j)-1)^2]`, file: 'network/discriminator_pathgan2D.py', line: 30 })
  }
  add({ id:'lift', kind:'lift', l:7, title:'AsFeatureMap_up · 全局到空间', short:'FC → 3×3', shared:true, shape:'256 → 1152 → 128 × 3 × 3', description:'z₇ 不是直接插值到 6×6。先经过共享的 weight-normalized Linear 映射到 1152 维，再 reshape 为 128×3×3，作为第一层 top-down 输入。', operations:['WeightNorm Linear · 256 → 1152','Reshape · B × 128 × 3 × 3'], formula:String.raw`g_7=\operatorname{reshape}(W_{\rm up}z_7)`, file:'network/blocks.py',line:208 })
  for (const v of LEVELS) {
    const l=v.l,c=v.feature,s=v.size, base={l, file:'network/mhvae.py', shared:true}
    if(l<7) {
      add({...base,id:`feature-${l}`,kind:'feature',title:`生成特征 g${l}`,short:`g${l}`,shape:shape(c,s),description:'上一级潜变量样本经 Upsample 和 BlockDecoder 后的特征。它分流至先验头，并与各模态的 skip 沿通道拼接；不是直接把原始潜变量与 skip 拼接。',operations:['BlockDecoder 输出','分流至 pz 与各模态 torch.cat'],formula:String.raw`g_l=D_l(U_l(z_{l+1}))`,line:253})
      add({...base,id:`prior-head-${l}`,kind:'priorhead',title:`先验概率头 P${l}`,short:`P${l} · Conv 1×1`,shape:`${c} → ${v.channels}+${v.channels} · ${s}²`,description:'生成特征通过共享的 WeightNorm 1×1 卷积，再分离并 soft-clamp 均值和 log-scale。',operations:[`WeightNorm Conv 1×1 · ${c}→${c} · bias=False`,'chunk(2) → μp / ap','soft_clamp → exp(ap)'],formula:String.raw`(\mu_{p,l},a_{p,l})=\operatorname{split}(P_l(g_l))`,line:258})
      for(const m of MODALITIES)add({...base,mod:m.id,id:`${m.id}-concat-${l}`,kind:'concat',title:`${m.label} · 拼接 g${l} 与 skip`,short:'Concat · dim 1',shape:`${c}+${c} → ${2*c} · ${s}²`,description:'两路张量空间尺寸相同，沿通道维拼接。生成特征来自中央 Decoder，skip 来自该模态同层 Encoder。',operations:[`g${l}: ${shape(c,s)}`,`${m.label} skip: ${shape(c,s)}`,'torch.cat((out, skip), dim=1)'],formula:String.raw`c^j_l=[g_l,h^j_l]_{C}`,line:267})
      add({...base,id:`up-${l}`,kind:'up',title:`Upsample · z${l+1} → level ${l}`,short:'↑ ×2 + Conv',shape:`${l===6?128:LEVELS.find(v=>v.l===l+1)!.channels} → ${c} · ${s/2}² → ${s}²`,description:'空间尺寸先用双线性插值翻倍，再用 3×3 卷积映射通道。align_corners=False。',operations:['Bilinear interpolate ×2 · align_corners=False',`Conv 3×3 · → ${c}ch · bias=False`],formula:String.raw`u_l=C_3\operatorname{Bilinear}_{\times2}(z_{l+1})`,file:'network/blocks.py',line:40})
      add({...base,id:`decoder-${l}`,kind:'decoder',title:`BlockDecoder · level ${l}`,short:`D${l} · expand ×6`,shape:shape(c,s),description:'中央生成路径的倒残差块。实现采用 3×3 扩展卷积、5×5 depthwise 卷积与 1×1 投影；扩展倍率为 6。它的输出同时提供先验参数和各模态 Gaussian head 的 top-down 条件。',operations:[`IN → Conv 3×3 · ${c}→${6*c}`,'IN → SiLU → Depthwise Conv 5×5',`groups = ${6*c} · padding 2`,'IN → SiLU → SE',`Conv 1×1 · ${6*c}→${c} → 残差相加`],formula:String.raw`g_l=u_l+C_1\operatorname{SE}\!\left(s\,\mathrm{IN}\!\left(\operatorname{DW}_5(s\,\mathrm{IN}(C_3\mathrm{IN}(u_l)))\right)\right)`,file:'network/blocks.py',line:92})
    }
    for(const m of MODALITIES)add({...base,mod:m.id,id:`${m.id}-factor-${l}`,kind:'factor',title:`${m.label} · 残差专家参数 z${l}`,short:`${m.label} · μ / log s`,shape:l===7?'B × 256':shape(v.channels,s),description:'Gaussian head 输出的均值与 log-scale 残差因子。它不是潜变量样本；仅实际观测到的模态参与本层 PoE。',operations:['μ 与 a 两个同形张量','存入 res_params','传入 compute_full'],formula:String.raw`r^j_l=(\mu^j_l,a^j_l)`,line:l===7?221:271})
    add({...base,id:`posterior-${l}`,kind:'posterior',title:`融合后验 q${l}`,short:`q${l} · Normal`,shape:l===7?'B × 256':shape(v.channels,s),description:'compute_full 返回的融合正态分布。均值与温度缩放后的标准差流向 rsample，随后得到本层潜变量。',operations:['Normal(μ, T·s)','将分布传入 rsample'],formula:String.raw`q_l=\mathcal N(\mu_l,\operatorname{diag}((Ts_l)^2))`,line:198})
    add({...base,id:`prior-${l}`,kind:'prior',title:`Prior Gaussian · z${l}`,short:l===7?'N(0, I)':`P${l} · μ / log s`,shape:l===7?'B × 256':shape(v.channels,s),description:l===7?'最顶层先验为固定标准正态，不包含可学习的 Gaussian head。':'共享 pz[i] 用带 WeightNorm 的 1×1 卷积生成均值与 log-scale，两者均 soft-clamp。Normal 的第二参数为 exp(log-scale)，即标准差。',operations:l===7?['μ = zeros_like(expert μ)','scale = ones_like(expert log-scale)','Normal(0, 1)']:[`WeightNorm Conv 1×1 · ${c}→${c} · bias=False`,'chunk(2) → μp, log-scale','soft_clamp · 10 tanh(a/10)','Normal(μp, exp(log-scale))'],formula:l===7?String.raw`p(z_7)=\mathcal N(0,I)`:String.raw`p_\theta(z_l\mid z_{>l})=\mathcal N(\mu_{p,l},\operatorname{diag}(s_{p,l}^{2}))`,line:l===7?224:258})
    add({...base,id:`poe-${l}`,kind:'poe',title:`Product of Experts · z${l}`,short:`PoE · z${l}`,shape:l===7?'B × 256':shape(v.channels,s),description:'已观测模态的 Gaussian 因子与条件先验在这里融合。此处严格展示 compute_full 的实现：使用倒数 scale 加权，而不是标准高斯 PoE 的倒数方差；理论与实现的区别见下方公式页。',operations:['初始化 μ_sum = prior.loc / prior.scale','初始化 w_sum = 1 / prior.scale','逐模态累加 μj exp(−aj) 和 exp(−aj)','μ = μ_sum / w_sum · s = 1 / w_sum','Normal(μ, temperature × s)'],formula:String.raw`s=\left(s_p^{-1}+\sum_{j\in r}s_j^{-1}\right)^{-1},\quad\mu=s\left(\frac{\mu_p}{s_p}+\sum_{j\in r}\frac{\mu_j}{s_j}\right)`,line:190})
    add({...base,id:`sample-${l}`,kind:'sample',title:`Reparameterize · z${l}`,short:`z${l} · sample`,shape:l===7?'B × 256':shape(v.channels,s),description:`调用融合分布的 rsample()，通过重参数化保留梯度。${l===1?'z₁ 送往四个模态独立解码器。':l===7?'z₇ 经线性映射恢复空间特征。':'该层样本用于下一层的 top-down 路径。'}温度缩放标准差，页面动效仅解释采样过程，不执行医学影像推理。`,operations:['ε ~ Normal(0, I)', 'z = μ + temperature × scale × ε','rsample() · 可微采样','累计 log q_full(z) − log p(z)'],formula:String.raw`z_l=\mu_l+T\,s_l\odot\epsilon,\qquad\epsilon\sim\mathcal N(0,I)`,line:l===7?234:278})
  }
  return nodes
}
export const NODES = createNodes()
export const NODE_MAP = new Map(NODES.map(n=>[n.id,n]))
export const STAGES = [
  {title:'编码观测',description:'信号沿外围竖直塔自下而上。七级 BlockEncoder 缓存特征，六次步长卷积压缩空间。'},
  {title:'融合全局',description:'共享线性 head 把每个最粗特征转为 Gaussian 因子，与标准正态先验融合，采样 z₇。'},
  {title:'逐层推断',description:'z₇ → z₁：生成特征构造先验，并与各塔 skip 拼接成 Gaussian 专家；每层重新融合、采样。'},
  {title:'生成模态',description:'z₁ 分流至四个独立图像解码器，每条路径经过六个 ResNet block 和两个 7×7 卷积。'},
]
