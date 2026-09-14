import {Engine,type Tensor,type Run,size} from './engine'
export type Family='基础运算'|'卷积与空间'|'归一化与门控'|'MobileNet'|'Transformer'|'循环网络'
export interface ModuleDef {id:string;name:string;family:Family;description:string;format:'image'|'sequence'|'vector';source:string}
const torch='https://docs.pytorch.org/docs/stable/nn.html',vision='https://docs.pytorch.org/vision/stable/_modules/torchvision/models/'
export const MODULES:ModuleDef[]=[
 ['transpose1d','一维转置卷积','卷积与空间','插零、核转置与边界贡献','image'],['transpose2d','二维转置卷积','卷积与空间','完整转置卷积与output_padding','image'],['transpose3d','三维转置卷积','卷积与空间','体积坐标上的转置映射','image'],['nearest','最近邻插值','卷积与空间','输入坐标与复制单元逐项对应','image'],['bilinear','双线性插值','卷积与空间','四邻点、距离权重与角点对齐','image'],
 ['linear','Linear 全连接','基础运算','每个输出的完整权重行与乘加归约','vector'],['mlp','多层感知机 MLP','基础运算','逐层 Linear 与非线性，不省略中间激活','vector'],['add','残差相加','基础运算','两路同形张量逐元素相加','image'],['concat','通道拼接','基础运算','维持空间索引，沿通道连接','image'],['flatten','Flatten 展平','基础运算','存储索引与目标维度逐项对应','image'],['softmax','Softmax','基础运算','稳定指数与逐行归一化','sequence'],
 ['conv1d','一维卷积','卷积与空间','序列上的分组互相关','image'],['conv2d','二维卷积','卷积与空间','卷积核、步幅、填充、膨胀和分组','image'],['conv3d','三维卷积','卷积与空间','完整体积感受野与通道归约','image'],['depthwise','深度可分离卷积','卷积与空间','逐通道空间卷积，再用1×1混合通道','image'],['cnn','卷积网络 CNN','卷积与空间','卷积、归一化、激活、池化与分类头','image'],['residual','ResNet BasicBlock','卷积与空间','两次3×3卷积；尺寸改变时投影旁路','image'],['maxpool','最大池化','卷积与空间','窗口逐项比较与最大值输出','image'],['avgpool','平均池化','卷积与空间','完整窗口求和再除元素数','image'],['gap','全局平均池化','卷积与空间','每通道全部空间坐标归约','image'],
 ...['ReLU','ReLU6','LeakyReLU','Sigmoid','Tanh','SiLU','GELU','Hardswish','Hardsigmoid','Softplus'].map(k=>[k,k,'归一化与门控','逐元素实际函数求值','vector']),
 ...['BatchNorm2d','InstanceNorm2d','GroupNorm'].map(k=>[k,k,'归一化与门控','显式统计轴、总体方差与仿射参数','image']),
 ...['LayerNorm','RMSNorm'].map(k=>[k,k,'归一化与门控','沿最后一维计算归一化','sequence']),['dropout','Dropout','归一化与门控','固定种子可复查的 Bernoulli 掩码','sequence'],['se','Squeeze-and-Excitation','归一化与门控','空间压缩、两次投影、门控与广播','image'],
 ['mobilev1','MobileNet V1 单元','MobileNet','3×3 depthwise +1×1 pointwise；BN与ReLU','image'],['mobilev2','MobileNet V2 倒残差','MobileNet','扩展、depthwise、线性瓶颈与条件残差','image'],['mobilev3','MobileNet V3 单元','MobileNet','可选 SE、硬激活、倒残差与线性投影','image'],
 ['attention','多头自注意力','Transformer','Q/K/V、head 拆分、QKᵀ、Softmax、AV与输出投影','sequence'],['cross','多头交叉注意力','Transformer','query 与 memory 独立来源','sequence'],['encoder','Transformer Encoder','Transformer','多层注意力、前馈与可选 pre/post norm','sequence'],['decoder','Transformer Decoder','Transformer','因果自注意力、交叉注意力与前馈','sequence'],['vit','ViT 编码器','Transformer','真实 patch 卷积、CLS、位置向量、Encoder与分类头','image'],['gpt','GPT 风格解码器','Transformer','学习位置向量、因果 pre-norm 层与最终读出','sequence'],['swiglu','SwiGLU 前馈','Transformer','SiLU门控与独立value投影，逐项相乘后读出','sequence'],
 ['rnn','RNN 展开','循环网络','完整时间展开与隐状态传递','sequence'],['gru','GRU 展开','循环网络','reset/update/new三门与PyTorch的候选隐状态顺序','sequence'],['lstm','LSTM 展开','循环网络','四门、cell state与hidden state逐步更新','sequence'],
].map(([id,name,family,description,format])=>({id,name,family:family as Family,description,format:format as ModuleDef['format'],source:id==='mobilev2'?vision+'mobilenetv2.html':id==='mobilev3'?vision+'mobilenetv3.html':id==='residual'?vision+'resnet.html':id==='mobilev1'?'https://arxiv.org/abs/1704.04861':torch})).sort((a,b)=>['基础运算','卷积与空间','归一化与门控','MobileNet','Transformer','循环网络'].indexOf(a.family)-['基础运算','卷积与空间','归一化与门控','MobileNet','Transformer','循环网络'].indexOf(b.family))
export interface Config {batch:number;channels:number;out:number;spatial:number;depth:number;kernel:number;stride:number;padding:number;groups:number;dilation:number;tokens:number;dim:number;heads:number;hidden:number;layers:number;expansion:number;activation:string;seed:number;training:boolean;preNorm:boolean;causal:boolean;se:boolean;pattern:string;alignCorners:boolean;outputPadding:number}
export const DEFAULT:Config={batch:1,channels:2,out:4,spatial:5,depth:3,kernel:3,stride:1,padding:1,groups:1,dilation:1,tokens:4,dim:4,heads:2,hidden:8,layers:2,expansion:3,activation:'ReLU',seed:17,training:false,preNorm:true,causal:false,se:true,pattern:'random',alignCorners:false,outputPadding:0}
export function execute(id:string,c:Config,customInput?:number[]):Run{
 const def=MODULES.find(v=>v.id===id)!;if(!def)throw Error('未找到模块')
 if(['attention','cross','encoder','decoder','gpt','vit'].includes(id)&&c.dim%c.heads)throw Error('embedding 维度必须能被注意力 head 数整除')
 const e=new Engine(c.seed),shape=def.format==='vector'?[c.batch,c.dim]:def.format==='sequence'?[c.batch,c.tokens,c.dim]:['conv1d','transpose1d'].includes(id)?[c.batch,c.channels,c.spatial]:['conv3d','transpose3d'].includes(id)?[c.batch,c.channels,c.depth,c.spatial,c.spatial]:[c.batch,c.channels,c.spatial,c.spatial]
 const data=customInput??(c.pattern==='ramp'?Array.from({length:size(shape)},(_,i)=>2*i/Math.max(1,size(shape)-1)-1):c.pattern==='impulse'?Array.from({length:size(shape)},(_,i)=>i===Math.floor(size(shape)/2)?1:0):undefined)
 let x=e.input=e.tensor('输入 X',shape,data)
 const group=(s:string)=>{e.group=s},act=(t:Tensor,k=c.activation)=>e.activation(t,k),bn=(t:Tensor)=>e.norm(t,'BatchNorm2d',1,id==='mobilev3'?1e-3:1e-5,c.training),drop=(t:Tensor)=>e.dropout(t,c.training?.1:0)
 const conv=(t:Tensor,out:number,k=3,stride=1,g=1,title='卷积',padding=Math.floor(k/2))=>e.conv(t,out,k,stride,padding,g,1,title,false)
 const se=(t:Tensor,hard=false)=>{const channels=t.shape[1],squeeze=hard?Math.max(8,Math.floor((channels/4+4)/8)*8):Math.max(1,Math.floor(channels/4));let z=e.pool(t,'global');z=e.conv(z,squeeze,1,1,0,1,1,'压缩通道');z=act(z,'ReLU');z=e.conv(z,channels,1,1,0,1,1,'恢复门控通道');z=act(z,hard?'Hardsigmoid':'Sigmoid');return e.binary(t,z,'mul','SE 广播调制')}
 function attention(query:Tensor,memory:Tensor,causal=false){
  const [batch,n,d]=query.shape,m=memory.shape[1],h=c.heads,dh=d/h
  const project=(t:Tensor,name:string)=>{const y=e.linear(t,d,true,name);return e.permute(e.reshape(y,[batch,t.shape[1],h,dh],name+' · 拆出 head'),[0,2,1,3],name+' · head 轴前置')}
  const q=project(query,'Q 查询'),k=project(memory,'K 键'),v=project(memory,'V 值')
  let logits=e.matmul(q,k,true,1/Math.sqrt(dh));if(causal){if(n!==m)throw Error('因果自注意力要求序列长度一致');logits=e.causal(logits)}
  const prob=drop(e.softmax(logits)),result=e.matmul(prob,v),merged=e.reshape(e.permute(result,[0,2,1,3],'合并 head 前重排'),[batch,n,d],'合并全部 head')
  return e.linear(merged,d,true,'注意力输出投影')
 }
 function encoder(t:Tensor,index:number,causal=false,memory?:Tensor){
  const sub=(name:string,fn:(z:Tensor)=>Tensor)=>{group(`层 ${index+1} / ${name}`);const base=t;if(c.preNorm)t=e.norm(t,'LayerNorm');t=drop(fn(t));t=e.binary(base,t,'add');if(!c.preNorm)t=e.norm(t,'LayerNorm')}
  sub('自注意力',z=>attention(z,z,causal));if(memory)sub('交叉注意力',z=>attention(z,memory))
  sub('前馈网络',z=>e.linear(drop(act(e.linear(z,c.hidden),id==='gpt'?'GELU':c.activation)),c.dim));return t
 }
 group(def.name)
 if(id==='linear')x=e.linear(x,c.out)
 else if(id==='mlp'){for(let i=0;i<c.layers;i++){group(`隐藏层 ${i+1}`);x=act(e.linear(x,c.hidden))}group('输出层');x=e.linear(x,c.out)}
 else if(['conv1d','conv2d','conv3d'].includes(id))x=e.conv(x,c.out,c.kernel,c.stride,c.padding,c.groups,c.dilation)
 else if(id.startsWith('transpose'))x=e.transposeConv(x,c.out,c.kernel,c.stride,c.padding,c.groups,c.dilation,c.outputPadding)
 else if(id==='nearest'||id==='bilinear')x=e.interpolate(x,id,2,c.alignCorners)
 else if(id==='add'||id==='concat'){const y=e.tensor('独立输入 Y',shape);x=id==='add'?e.binary(x,y,'add'):e.concat([x,y],1)}
 else if(id==='flatten')x=e.reshape(x,[c.batch,size(shape.slice(1))])
 else if(['depthwise','mobilev1'].includes(id)){x=conv(x,c.channels,3,c.stride,c.channels,'逐通道空间卷积');if(id==='mobilev1')x=act(bn(x),'ReLU');x=conv(x,c.out,1,1,1,'逐点通道混合');if(id==='mobilev1')x=act(bn(x),'ReLU')}
 else if(id==='cnn'){for(let i=0;i<c.layers;i++){group(`卷积层 ${i+1}`);x=act(bn(conv(x,c.out,3)))}group('空间归约与分类');x=e.pool(x,'global');x=e.linear(e.reshape(x,[c.batch,c.out]),c.out)}
 else if(id==='residual'){const input=x;group('主分支');x=act(bn(conv(x,c.out,3,c.stride)),'ReLU');x=bn(conv(x,c.out,3));let shortcut=input;if(c.channels!==c.out||c.stride!==1){group('投影旁路');shortcut=bn(conv(input,c.out,1,c.stride))}group('残差汇合');x=act(e.binary(x,shortcut,'add'),'ReLU')}
 else if(['maxpool','avgpool','gap'].includes(id))x=e.pool(x,id==='gap'?'global':id==='maxpool'?'max':'avg',Math.min(c.kernel,c.spatial),c.stride)
 else if(['mobilev2','mobilev3'].includes(id)){
  const input=x,expanded=c.channels*c.expansion,is3=id==='mobilev3',activation=is3?'Hardswish':'ReLU6';group('通道扩展')
  if(c.expansion!==1)x=act(bn(conv(x,expanded,1)),activation)
  group('深度空间卷积');x=act(bn(conv(x,expanded,is3?c.kernel:3,c.stride,expanded,'Depthwise')),activation)
  if(is3&&c.se){group('压缩与激励 SE');x=se(x,true)}group('线性瓶颈');x=bn(conv(x,c.out,1))
  if(c.channels===c.out&&c.stride===1){group('恒等旁路');x=e.binary(x,input,'add')}
 }
 else if(id==='se')x=se(x)
 else if(['BatchNorm2d','InstanceNorm2d','GroupNorm','LayerNorm','RMSNorm'].includes(id))x=e.norm(x,id as Parameters<Engine['norm']>[1],c.groups,1e-5,c.training)
 else if(id==='dropout')x=e.dropout(x,c.training?.25:0)
 else if(id==='softmax')x=e.softmax(x)
 else if(id==='attention'||id==='cross')x=attention(x,id==='cross'?e.tensor('外部 memory',[c.batch,c.tokens,c.dim]):x,c.causal&&id==='attention')
 else if(['encoder','decoder','gpt','vit'].includes(id)){
  if(id==='vit'){group('Patch 与位置编码');x=e.conv(x,c.dim,c.kernel,c.stride,0,1,1,'无填充 patch 投影');const [,d,h,w]=x.shape;x=e.permute(e.reshape(x,[c.batch,d,h*w]),[0,2,1],'空间 patch 序列');const cls=e.param('CLS',[c.batch,1,d],.02);x=e.concat([cls,x],1,'前置 CLS')}
  if(id==='vit'||id==='gpt'){group('位置编码');x=e.binary(x,e.param('可学习位置嵌入',[1,x.shape[1],c.dim],.02),'add')}
  const memory=id==='decoder'?e.tensor('编码器 memory',[c.batch,c.tokens,c.dim]):undefined
  for(let i=0;i<c.layers;i++)x=encoder(x,i,id==='decoder'||id==='gpt'||c.causal,memory)
  if(id==='gpt'||id==='vit'){group('最终读出');x=e.norm(x,'LayerNorm');if(id==='vit')x=e.reshape(e.slice(x,1,0,1,'读取 CLS'),[c.batch,c.dim]);x=e.linear(x,c.out,true,id==='gpt'?'逐 token logits':'分类 logits')}
 }
 else if(id==='swiglu'){const gate=act(e.linear(x,c.hidden,false,'门控投影'),'SiLU'),v=e.linear(x,c.hidden,false,'值投影');x=e.linear(e.binary(gate,v,'mul'),c.dim,false,'输出投影')}
 else if(['rnn','gru','lstm'].includes(id)){
  // Gate weights are shared by all time steps, including separate input/recurrent biases.
  const hidden=c.hidden,gates=id==='lstm'?4:id==='gru'?3:1,wi=e.param('W_ih',[gates*hidden,c.dim],1/Math.sqrt(hidden)),wh=e.param('W_hh',[gates*hidden,hidden],1/Math.sqrt(hidden)),bi=e.param('b_ih',[gates*hidden],1/Math.sqrt(hidden)),bh=e.param('b_hh',[gates*hidden],1/Math.sqrt(hidden))
  const affine=(t:Tensor,w:Tensor,b:Tensor,name:string)=>{const n=t.shape.at(-1)!,m=b.shape[0],y=e.tensor(name,[c.batch,m],Array.from({length:c.batch*m},(_,i)=>{const r=Math.floor(i/m),o=i%m;return b.values[o]+Array.from({length:n},(_,j)=>t.values[r*n+j]*w.values[o*n+j]).reduce((a,b)=>a+b,0)}));return e.record(name,'Linear',[t,w,b],y,String.raw`y=xW^T+b`,'时间维共享同一套门控权重，输入与递归偏置分别保留。',i=>[...Array.from({length:n},(_,j)=>e.term(t,Math.floor(i/m)*n+j,w.values[(i%m)*n+j])),e.term(b,i%m,1)],{})}
  let h=e.tensor('初始 hidden',[c.batch,hidden],Array(c.batch*hidden).fill(0)),cell=e.tensor('初始 cell',[c.batch,hidden],Array(c.batch*hidden).fill(0));const outputs:Tensor[]=[]
  const oneMinus=(t:Tensor)=>e.record('1 − gate','Identity',[t],e.tensor('反向门控',t.shape,t.values.map(v=>1-v)),String.raw`1-g`,'互补门控。',i=>[e.term(t,i)],{operation:'one-minus'})
  for(let time=0;time<c.tokens;time++){
   group(`时间步 ${time+1}`);const xt=e.reshape(e.slice(x,1,time,time+1,'当前输入'),[c.batch,c.dim]),a=affine(xt,wi,bi,'输入门控'),b=affine(h,wh,bh,'递归门控'),gate=(v:Tensor,j:number)=>e.slice(v,1,j*hidden,(j+1)*hidden,'门控分组')
   if(id==='rnn')h=act(e.binary(a,b,'add'),'Tanh')
   else if(id==='gru'){const r=act(e.binary(gate(a,0),gate(b,0),'add'),'Sigmoid'),z=act(e.binary(gate(a,1),gate(b,1),'add'),'Sigmoid'),n=act(e.binary(gate(a,2),e.binary(r,gate(b,2),'mul'),'add'),'Tanh');h=e.binary(e.binary(oneMinus(z),n,'mul'),e.binary(z,h,'mul'),'add')}
   else {const all=e.binary(a,b,'add'),i=act(gate(all,0),'Sigmoid'),f=act(gate(all,1),'Sigmoid'),g=act(gate(all,2),'Tanh'),o=act(gate(all,3),'Sigmoid');cell=e.binary(e.binary(f,cell,'mul'),e.binary(i,g,'mul'),'add');h=e.binary(o,act(cell,'Tanh'),'mul')}
   outputs.push(e.reshape(h,[c.batch,1,hidden],'当前 hidden 输出'))
  }group('时间序列输出');x=e.concat(outputs,1)
 }else x=act(x,id)
 return e.run(x)
}
