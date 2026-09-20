import {MODULES} from './models'

/** Learning/task hierarchy, independent of the implementation's execution groups. */
export const CATALOG = [
 {name:'概率编码与 VAE',branches:[{name:'编码器与后验参数',ids:['vae-head','vae-mlp','vae-conv']},{name:'采样、融合与正则',ids:['vae-reparameter','vae-poe','vae-kl']}]},
 {name:'生成与扩散',branches:[{name:'CUT · 非配对图像翻译',ids:['cut-generator','cut-patchgan','cut-projector','cut-nce','cut-objective']},{name:'Diffusion · 概率与采样',ids:['diffusion-forward','diffusion-reverse','diffusion-ddim']},{name:'Diffusion · 条件网络与训练',ids:['diffusion-time','diffusion-unet','diffusion-train']} ]},
 {name:'基础与张量', branches:[
  {name:'线性与数据组织',ids:['linear','mlp','add','concat','flatten']},
  {name:'激活函数',ids:['ReLU','ReLU6','LeakyReLU','Sigmoid','Tanh','SiLU','GELU','Hardswish','Hardsigmoid','Softplus']},
  {name:'归一化与正则',ids:['softmax','BatchNorm2d','InstanceNorm2d','GroupNorm','LayerNorm','RMSNorm','dropout']},
 ]},
 {name:'空间与视觉', branches:[
  {name:'卷积与转置卷积',ids:['conv1d','conv2d','conv3d','transpose1d','transpose2d','transpose3d','depthwise']},
  {name:'池化与重采样',ids:['maxpool','avgpool','gap','nearest','bilinear']},
  {name:'视觉骨干与通道门控',ids:['cnn','residual','se']},
  {name:'MobileNet 家族',ids:['mobilev1','mobilev2','mobilev3']},
 ]},
 {name:'序列与注意力', branches:[
  {name:'注意力与前馈单元',ids:['attention','cross','swiglu']},
  {name:'Transformer 架构',ids:['encoder','decoder','vit','gpt']},
  {name:'循环与记忆单元',ids:['rnn','gru','lstm']},
 ]},
]
export const catalogPath=(id:string)=>{const root=CATALOG.find(r=>r.branches.some(b=>b.ids.includes(id)))!;return [root.name,root.branches.find(b=>b.ids.includes(id))!.name]}
export const matchesModule=(id:string,query:string)=>{const m=MODULES.find(m=>m.id===id)!;return `${m.id} ${m.name} ${m.description} ${catalogPath(id).join(' ')}`.toLowerCase().includes(query.trim().toLowerCase())}
