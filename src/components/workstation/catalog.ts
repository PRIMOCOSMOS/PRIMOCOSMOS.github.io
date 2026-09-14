import {MODULES} from './models'

/** Learning/task hierarchy, independent of the implementation's execution groups. */
export const CATALOG = [
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
