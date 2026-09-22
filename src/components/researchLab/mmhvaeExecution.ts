/** Forward graph transcribed from pinned MHVAE2D.forward / network.blocks.
 * Independent of the former icon scene and its conceptual Gaussian markers. */
import type {SourcePlan} from './sourceExecution'
import type {MathKind,MathSpec} from '../mmhvae/mathematics'
import {mathSpec} from '../mmhvae/mathematics'
import {MODALITIES,LEVELS,NODE_MAP} from '../mmhvae/model'
import {atomId,canonicalPath} from '../mmhvae/navigation'
import type {Glyph} from '../mmhvae/anatomy'
const R=String.raw;
export function concreteMmhvaePlan(observed:string[],temperature=.5):SourcePlan{
 const plan:SourcePlan={leaves:[],links:[],paths:{root:[]},titles:{root:'MMHVAE'}};
 const shapes=new Map<string,number[]>();
 const add=(root:string,id:string,title:string,kind:MathKind,operation:string,shape:number[],inputs:string[],formula='',options:Partial<MathSpec>={},settings:Record<string,unknown>={})=>{
  const full=atomId(root,id),glyph:Glyph=kind==='convolution'?'conv':kind==='normalization'?'norm':kind==='linear'?'linear':kind==='activation'?'activation':'tensor';
  const spec={...mathSpec({id,title,glyph,shape:shape.join(' × '),detail:'',sourceName:id},formula),kind,operation,...options};spec.formula=formula||spec.formula;
  const node=NODE_MAP.get(root.split('/')[0]),path=[...canonicalPath(root),full],group=root;
  plan.titles[root]=root.includes('/se/')?`${node?.title??root} · 通道注意力`:node?.title??root;plan.titles[full]=title;
  plan.leaves.push({module:{id:full,title,shape:shape.join(' × '),dimensions:shape,glyph,formula:spec.formula,source:`${node?.file??'network/mhvae.py'}:${node?.line??190} · ${id}`,operator:spec,children:[],edges:[],settings},path,group,lane:root.split('/')[0]});
  for(const input of inputs)plan.links.push({from:input,to:full});
  for(const p of path){plan.paths[p]??=[];plan.paths[p].push(full)}shapes.set(full,shape);return full;
 };
 const unary=(r:string,id:string,title:string,x:string,op:string,formula:string)=>add(r,id,title,'activation',op,shapes.get(x)!,[x],formula);
 const norm=(r:string,id:string,x:string)=>add(r,id,'InstanceNorm · 逐通道空间统计','normalization','instance',shapes.get(x)!,[x],R`Y=(X-mu)/sqrt{v+10^{-5}}`);
 const act=(r:string,id:string,x:string,type='silu')=>unary(r,id,type==='silu'?'SiLU':type==='relu'?'ReLU':type==='leaky'?'LeakyReLU(0.2)':type==='tanh'?'Tanh':'Sigmoid',x,type,type==='silu'?R`Y=Xsigma(X)`:type==='relu'?R`Y=max(0,X)`:type==='leaky'?R`Y=max(X,0.2X)`:type==='tanh'?R`Y=	anh X`:R`Y=(1+e^{-X})^{-1}`);
 const conv=(r:string,id:string,x:string,out:number,k=3,stride=1,pad=(k-1)/2,bias=true,dw=false,wn=false,reflect=false)=>{
  const d=shapes.get(x)!,shape=[out,...d.slice(-2).map(v=>Math.floor((v+2*pad-k)/stride)+1)];return add(r,id,`${dw?'逐通道':'卷积'} ${k}×${k} · ${d[0]}→${out}`,dw?'depthwise':'convolution',dw?'depthwise':'conv',shape,[x],R`Y_{o,h,w}=b_o+sum_{c,u,v}X_{c,sh-p+u,sw-p+v}W_{o,c,u,v}`,{kernel:k,stride,padding:pad,bias,weightNorm:wn,reflect});
 };
 const linear=(r:string,id:string,x:string,out:number,wn=false)=>add(r,id,`全连接 ${shapes.get(x)!.at(-1)}→${out}`,'linear','linear',[...shapes.get(x)!.slice(0,-1),out],[x],R`Y=XW^	op+b`,{weightNorm:wn,bias:true});
 const plus=(r:string,id:string,a:string,b:string)=>add(r,id,'残差逐元素相加','addition','add',shapes.get(a)!,[a,b],R`Y=A+B`);
 const se=(r:string,x:string)=>{
  const d=shapes.get(x)!,c=d[0],scope=`${r}/se/${c}`;
  let a=add(scope,'mean','每通道全局平均','pooling','pool',[c],[x],R`u_c=rac1{HW}sum_{h,w}X_{c,h,w}`);
  a=linear(scope,'fc1',a,Math.max(Math.floor(c/16),4));a=act(scope,'relu',a,'relu');a=linear(scope,'fc2',a,c);a=act(scope,'sigmoid',a,'sigmoid');
  a=add(scope,'view','门控重排 C×1×1','reshape','reshape',[c,1,1],[a],R`gleftarrowoperatorname{view}(g,C,1,1)`);
  return add(scope,'gate','逐通道广播门控','multiplication','multiply',d,[x,a],R`Y_{c,h,w}=X_{c,h,w}g_{c,0,0}`);
 };
 const encoder=(r:string,x:string)=>{const c=shapes.get(x)![0];let a=norm(r,'bn_0',x);a=act(r,'act_0',a);a=conv(r,'conv_0',a,c);a=norm(r,'bn_1',a);a=act(r,'act_1',a);a=conv(r,'conv_1',a,c);return plus(r,'add',se(r,a),x)};
 const split=(r:string,x:string)=>{
  const dims=shapes.get(x)!,half=dims[0]/2,shape=[half,...dims.slice(1)];
  const mu=add(r,'mu','均值通道切片','split','slice',shape,[x],R`mu=h[:C/2]`,{},{axis:0,start:0});
  const a=add(r,'scale','log-scale 通道切片','split','slice',shape,[x],R`a=h[C/2:]`,{},{axis:0,start:half});
  return [unary(r,'mu-clamp','均值 soft-clamp',mu,'clamp',R`mu=10	anh(mu/10)`),unary(r,'scale-clamp','log-scale soft-clamp',a,'clamp',R`a=10	anh(a/10)`)] as const;
 };
 const skips=new Map<string,string>(),inputs=new Map<string,string>();
 for(const mod of observed){let x=add(`${mod}-input`,'input',`${MODALITIES.find(m=>m.id===mod)?.label} 输入切片`,'tensor','feature',[1,192,192],[],R`Xinmathbb R^{B	imes1	imes192	imes192}`);inputs.set(mod,x);x=conv(`${mod}-stem`,'first_conv',x,16);
  for(let l=1;l<=7;l++){x=encoder(`${mod}-encoder-${l}`,x);skips.set(`${mod}:${l}`,x);if(l<7)x=conv(`${mod}-down-${l}`,'td',x,Math.min(shapes.get(x)![0]*2,128),3,2)}
 }
 let z='';
 for(let l=7;l>=1;l--){const level=LEVELS.find(v=>v.l===l)!,shape=l===7?[256]:[level.channels,level.size,level.size];let feature='',muP='',scaleP='';
  if(l===7){muP=add('prior-7','zeros','先验均值 0','tensor','zeros',shape,[]);scaleP=add('prior-7','ones','先验标准差 1','tensor','ones',shape,[])}
  else{
   const r=`up-${l}`,old=shapes.get(z)!;let x=add(r,'resize','双线性 ×2 · 半像素坐标','interpolation','bilinear',[old[0],old[1]*2,old[2]*2],[z],R`u=(u'+1/2)/2-1/2`);x=conv(r,'conv_1',x,level.feature,3,1,1,false);
   const d=`decoder-${l}`,c=level.feature;let a=norm(d,'bn_0',x);a=conv(d,'conv_0',a,c*6);a=norm(d,'bn_1',a);a=act(d,'act_1',a);a=conv(d,'dw_conv_1',a,c*6,5,1,2,true,true);a=norm(d,'bn_2',a);a=act(d,'act_2',a);a=se(d,a);a=conv(d,'conv_2',a,c,1);feature=plus(d,'add',a,x);
   const head=conv(`prior-head-${l}`,'pz',feature,c,1,1,0,false,false,true),[mu,aP]=split(`prior-head-${l}`,head);muP=mu;scaleP=unary(`prior-${l}`,'exp','先验标准差 exp(a)',aP,'exp',R`s_p=exp(a_p)`);
  }
  const experts:{mu:string;a:string}[]=[];
  for(const mod of observed){const r=`${mod}-expert-${l}`,skip=skips.get(`${mod}:${l}`)!;let a:string;
   if(l===7){a=add(r,'flatten','展平 C×H×W','reshape','reshape',[1152],[skip],R`h=operatorname{vec}(X)`);a=linear(r,'linear',a,512,true)}
   else {const c=level.feature;a=add(r,'cat','生成特征与模态 skip 拼接','concatenation','concat',[2*c,level.size,level.size],[feature,skip],R`h=operatorname{cat}(g,h^{mod};C)`);a=norm(r,'bn_0',a);a=act(r,'act_0',a);a=conv(r,'conv_0',a,c);a=norm(r,'bn_1',a);a=act(r,'act_1',a);a=conv(r,'conv_1',a,c);a=se(r,a);a=conv(r,'last_conv',a,c,1,1,0,false,false,true)}
   const [mu,logscale]=split(r,a);experts.push({mu,a:logscale});
  }
  const poe=`poe-${l}`;let w=unary(poe,'prior-weight','先验倒数尺度',scaleP,'reciprocal',R`w_p=1/s_p`),m=add(poe,'prior-mean','先验加权均值','multiplication','multiply',shape,[muP,w],R`m_p=mu_p w_p`);
  for(const [j,e] of experts.entries()){const weight=unary(poe,`expert-weight-${j}`,'专家倒数尺度 exp(−a)',e.a,'exp-negative',R`w_j=exp(-a_j)`),contribution=add(poe,`contribution-${j}`,'专家均值贡献','multiplication','multiply',shape,[e.mu,weight],R`m_j=mu_jw_j`);w=add(poe,`weight-sum-${j}`,'累加倒数尺度','addition','add',shape,[w,weight],R`wleftarrow w+w_j`);m=add(poe,`mean-sum-${j}`,'累加加权均值','addition','add',shape,[m,contribution],R`mleftarrow m+m_j`)}
  const mu=add(poe,'normalize','融合均值 m / w','multiplication','divide',shape,[m,w],R`mu=m/w`),sigma=unary(poe,'sigma','融合标准差 1 / w',w,'reciprocal',R`s=1/w`),scale=unary(poe,'temperature',`温度缩放 T=${temperature}`,sigma,'temperature',`s_T=${temperature}s`);
  const sample=`sample-${l}`,eps=add(sample,'eps','标准正态噪声 ε','tensor','normal-noise',shape,[],R`epsilonsimmathcal N(0,I)`),noise=add(sample,'scale','标准差 × 噪声','multiplication','multiply',shape,[scale,eps],R`eta=s_Todotepsilon`);z=add(sample,'mean',`潜变量 z${l} · 重参数化`,'addition','add',shape,[mu,noise],R`z=mu+eta`);
  if(l===7){z=linear('lift','linear',z,1152,true);z=add('lift','view','恢复空间特征 128×3×3','reshape','reshape',[128,3,3],[z],R`g_7=operatorname{view}(Wz_7,128,3,3)`)}
 }
 for(const mod of MODALITIES){let x=z;
  for(let i=1;i<=6;i++){const r=`${mod.id}-resnet-${i}`;let a=add(r,'pad1','镜像边界填充','padding','reflect',[8,194,194],[x],R`X'=operatorname{ReflectionPad}_1(X)`,{padding:1});a=conv(r,'conv0',a,8,3,1,0);a=norm(r,'norm0',a);a=act(r,'relu',a,'relu');a=add(r,'pad2','镜像边界填充','padding','reflect',[8,194,194],[a],R`X'=operatorname{ReflectionPad}_1(X)`,{padding:1});a=conv(r,'conv1',a,8,3,1,0);a=norm(r,'norm1',a);x=plus(r,'add',a,x)}
  const r=`${mod.id}-output`;x=conv(r,'convt2',x,8,7,1,3,true,false,false,true);x=norm(r,'norm2',x);x=act(r,'act',x,'leaky');x=conv(r,'convt3',x,1,7,1,3,true,false,false,true);x=act(r,'act_last',x,'tanh');add(`${mod.id}-image`,'mask','首个输入的背景掩码','mask','background',[1,192,192],[x,inputs.get(observed[0])!],R`Y=2((X+1)/2cdotmathbf1[X_{first}>-1])-1`);
 }
 return plan;
}
