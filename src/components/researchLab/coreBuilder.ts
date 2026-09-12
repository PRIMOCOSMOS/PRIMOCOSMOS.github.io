import {ModelBuilder,type Entry,type PaperModel} from './catalog'
import {operatorSpec,type Op} from './operators'
import {coreSpec,type CoreOp} from './coreOperators'
import type {Glyph} from '../mmhvae/anatomy'
const R=String.raw
export class CoreBuilder extends ModelBuilder{
 file:string
 constructor(file:string){super();this.file=file}
 folder(id:string,parent:string|undefined,title:string,shape:string,purpose:string,implementation:string,formula='',line=1,file=this.file){return this.add(id,parent,title,'network',shape,purpose,implementation,formula,file,line)}
 atom(id:string,parent:string,title:string,op:Op|CoreOp,shape:string,detail:string,formula='',line=1,glyph:Glyph='tensor',file=this.file){
  const core=['layernorm','gelu','scores','softmax','weighted','threshold','l1norm','dropout','vq','noise','ddim','tensor-signal','angular','log-signal','rotation'].includes(op)
  const spec=core?coreSpec(op as CoreOp,formula):operatorSpec(op as Op,detail,formula)
  return this.add(id,parent,title,glyph,shape,spec.steps[0].explanation,detail,formula||spec.steps[1].formula,file,line,id,spec)
 }
 dropout(id:string,parent:string,prob:number,line:number){
  const e=this.atom(id,parent,`Dropout · p=${prob}`,'dropout','形状不变',`训练按p=${prob}独立丢弃并倒置缩放；eval时恒等。`,'',line,'activation')
  e.math!.operation=`core:dropout:${prob}`
  e.math!.steps[0].explanation=`读取当前特征，训练丢弃率为 ${prob}；作者推理配置关闭 dropout。`
  e.math!.steps[2].explanation='保留项除以1−p；p=0或eval时原样输出。教学固定掩码仅说明训练运算。'
  return e
 }
 transformer(id:string,parent:string,title:string,dim:number,heads:number,hidden:number,line:number,extraResidual=false){
  const e=this.folder(id,parent,title,`B × N × ${dim}`, '让 token 交换信息，再独立变换每个 token 的特征。',`PreNorm 注意力残差 + PreNorm 前馈残差${extraResidual?' + 层输入额外旁路':''}。${heads} heads，每头 ${dim/heads} 维；MLP 隐藏宽 ${hidden}。`,extraResidual?R`y=x+A(\mathrm{LN}_1x),\quad z=y+F(\mathrm{LN}_2y)+x`:R`y=x+A(\mathrm{LN}_1x),\quad z=y+F(\mathrm{LN}_2y)`,line)
  this.atom(id+'/input',id,'层输入与残差分路','tensor',e.shape,'原始 token 同时流入注意力归一化与残差相加。',R`x`,line)
  this.atom(id+'/norm1',id,'注意力前 LayerNorm','layernorm',e.shape,'每个 token 沿 embedding 轴做 LayerNorm，含 affine。','',line)
  const att=this.folder(id+'/attention',id,'多头自注意力',`B × ${heads} × N × ${dim/heads}`,'通过全体 key 对当前 query 汇聚上下文。','QKV 无偏置联合投影；按 head 拆分；缩放点积、softmax、AV；合并 head 后线性输出。',R`A(X)=\operatorname{Concat}_h\!\left(\operatorname{softmax}(Q_hK_h^T/\sqrt{d_h})V_h\right)W_O`,line)
  this.atom(att.id+'/qkv',att.id,'查询／键／值联合投影','linear',`${dim} → ${3*dim}`,'bias=False；每个 token 的三套投影矩阵。',R`[Q,K,V]=XW_{QKV}`,line,'linear').math!.bias=false
  this.atom(att.id+'/split',att.id,'拆出 head 与 Q/K/V','split',`3 × B × ${heads} × N × ${dim/heads}`,'reshape / rearrange；把联合投影拆为三个张量，再组织 head。',R`[Q,K,V]\in\mathbb R^{3\times B\times H\times N\times d_h}`,line,'split').math!.operation='lab:split:3'
  this.atom(att.id+'/scores',att.id,'查询与键的缩放点积','scores','B × heads × N × N','示例关系矩阵为 4×4，计算实际点积并除 sqrt(head_dim)。','',line,'linear')
  this.atom(att.id+'/softmax',att.id,'沿 key 归一化','softmax','B × heads × N × N','softmax(dim=-1)，每个 query 行的权重总和为 1。','',line,'activation')
  if(!extraResidual)this.dropout(att.id+'/attn-drop',att.id,0,line)
  this.atom(att.id+'/values',att.id,'加权汇聚值向量','weighted','B × heads × N × head_dim','各 head 独立计算 attention @ V。','',line,'linear')
  this.atom(att.id+'/merge',att.id,'合并多头特征','reshape',`B × N × ${dim}`,'重排 head 维到 embedding，元素顺序按实现恢复。',R`[B,H,N,d_h]\to[B,N,Hd_h]`,line)
  this.atom(att.id+'/output',att.id,'注意力输出投影','linear',`${dim} → ${dim}`,'含偏置的输出 Linear。',R`O=\operatorname{Concat}(O_h)W_O+b_O`,line,'linear');this.dropout(att.id+'/out-drop',att.id,extraResidual?.1:0,line);this.chain(att.id);this.link(att.id,att.id+'/split',att.id+'/values','V 分支',true)
  this.atom(id+'/add1',id,'注意力残差相加','add',e.shape,'原输入加注意力输出。',R`y=x+A(\mathrm{LN}_1x)`,line,'sum')
  this.atom(id+'/norm2',id,'前馈前 LayerNorm','layernorm',e.shape,'每个 token 独立归一化后送入 MLP。','',line)
  const ff=this.folder(id+'/mlp',id,'逐 token 前馈网络',`${dim} → ${hidden} → ${dim}`,'在特征维构建非线性组合，不跨 token 混合。','Linear → GELU → Dropout → Linear → Dropout。',R`F(x)=W_2\operatorname{GELU}(W_1x+b_1)+b_2`,line)
  this.atom(ff.id+'/fc1',ff.id,'前馈输入投影','linear',`${dim} → ${hidden}`,'含偏置 Linear。','',line,'linear')
  this.atom(ff.id+'/gelu',ff.id,'GELU 平滑门控','gelu',String(hidden),'nn.GELU()。','',line,'activation')
  this.dropout(ff.id+'/drop1',ff.id,extraResidual?.1:0,line)
  this.atom(ff.id+'/fc2',ff.id,'前馈输出投影','linear',`${hidden} → ${dim}`,'含偏置 Linear。','',line,'linear');this.dropout(ff.id+'/drop2',ff.id,extraResidual?.1:0,line);this.chain(ff.id)
  this.atom(id+'/add2',id,'前馈残差相加','add',e.shape,'前馈输出与 y 相加。',R`z=y+F(\mathrm{LN}_2y)`,line,'sum')
  if(extraResidual)this.atom(id+'/outer',id,'额外层输入旁路','add',e.shape,'vit_pytorch.py 中 ff(x) 已是 Residual，随后仍 +x1。公开实现包含第三条旁路。',R`z\leftarrow z+x`,line,'sum')
  this.chain(id);this.link(id,id+'/input',id+'/add1','原始 token',true);this.link(id,id+'/add1',id+'/add2','注意力后 token',true);if(extraResidual)this.link(id,id+'/input',id+'/outer','层输入 x1',true)
  return e
 }
 vertical(id:string,x:number,top:number,step=5,z=0){const e=this.entries[id];e.overview=true;e.children.forEach((key,i)=>this.entries[key].position=[x,top-i*step,z])}
 overview(){const keys:string[]=[],edges:PaperModel['overviewEdges']=[];const flatten=(e:Entry)=>{if(e.overview){for(const id of e.children)flatten(this.entries[id]);edges.push(...e.edges)}else keys.push(e.id)};for(const id of this.entries.root.children)flatten(this.entries[id]);const end=(id:string,from:boolean):string=>{const e=this.entries[id];return e?.overview&&e.children.length?end(from?e.children.at(-1)!:e.children[0],from):id};return {keys,edges:[...edges,...this.entries.root.edges].map(e=>({...e,from:end(e.from,true),to:end(e.to,false)}))}}
}
