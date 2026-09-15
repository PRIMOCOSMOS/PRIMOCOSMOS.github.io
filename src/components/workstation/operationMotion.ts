import * as T from 'three'
import type {Step} from './engine'
import type {Position3} from '../mmhvae/crystalPrimitives'

export interface Connection {from:Position3;to:Position3;weight?:number;output?:number;term?:number}
/** A connection is a weighted dependency, not a proxy neuron. GPU pulses travel
 * through all edges; output/term selection picks out the actual arithmetic. */
export function connectionFabric(parent:T.Group,links:Connection[]){
 const geometry=new T.BufferGeometry(),positions:number[]=[],along:number[]=[],weights:number[]=[],outputs:number[]=[],terms:number[]=[];
 for(const l of links){positions.push(...l.from,...l.to);along.push(0,1);weights.push(l.weight??1,l.weight??1);outputs.push(l.output??-1,l.output??-1);terms.push(l.term??-1,l.term??-1)}
 geometry.setAttribute('position',new T.Float32BufferAttribute(positions,3));geometry.setAttribute('along',new T.Float32BufferAttribute(along,1));geometry.setAttribute('weight',new T.Float32BufferAttribute(weights,1));geometry.setAttribute('outputIndex',new T.Float32BufferAttribute(outputs,1));geometry.setAttribute('termIndex',new T.Float32BufferAttribute(terms,1));
 const material=new T.ShaderMaterial({transparent:true,depthWrite:false,uniforms:{phase:{value:0},selectedOutput:{value:-1},selectedTerm:{value:-1},strength:{value:1}},vertexShader:`attribute float along,weight,outputIndex,termIndex; varying float vAlong,vWeight,vOutput,vTerm; void main(){vAlong=along;vWeight=weight;vOutput=outputIndex;vTerm=termIndex;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,fragmentShader:`uniform float phase,selectedOutput,selectedTerm,strength;varying float vAlong,vWeight,vOutput,vTerm;void main(){float chosen=1.-step(.1,abs(vOutput-selectedOutput));float current=chosen*(1.-step(.1,abs(vTerm-selectedTerm)));float pulse=pow(max(0.,1.-abs(fract(vAlong-phase*1.6)-.5)*8.),2.);float magnitude=.22+.78*min(1.,abs(vWeight));vec3 tint=vWeight<0.?vec3(.94,.62,.39):vec3(.34,.78,.93);tint=mix(tint,vec3(.88,.98,1.),current*.8);float opacity=(.2*magnitude+chosen*.32+current*.2+pulse*(.12+chosen*.4))*strength;gl_FragColor=vec4(tint,opacity);}`});
 const mesh=new T.LineSegments(geometry,material);parent.add(mesh);
 return {mesh,update:(phase:number,output=-1,term=-1,strength=1)=>{material.uniforms.phase.value=phase;material.uniforms.selectedOutput.value=output;material.uniforms.selectedTerm.value=term;material.uniforms.strength.value=strength},dispose:()=>{parent.remove(mesh);geometry.dispose();material.dispose()}};
}

export function operationFamily(s:Step){
 if(s.kind==='Linear')return 'dense';
 if(s.kind.startsWith('Conv'))return 'convolution';
 if(['ReLU','ReLU6','LeakyReLU','GELU','Sigmoid','SiLU','Tanh','Hardsigmoid','Hardswish','Softplus'].includes(s.kind))return 'activation';
 if(s.kind.includes('Norm')||String(s.settings.operation).startsWith('norm-'))return 'normalize';
 if(s.kind.includes('Pool')||s.kind==='Softmax'||['row-max','row-sum','reduce-all','l2-norm','cross-entropy-zero','cumprod'].includes(String(s.settings.operation)))return 'reduce';
 if(s.settings.operation==='add')return 'merge';
 if(s.settings.operation==='mul'||s.kind==='Dropout')return 'gate';
 if(s.settings.operation==='box-muller')return 'noise';
 if('transposeB' in s.settings)return 'contract';
 if(s.settings.axes||s.settings.axis!==undefined||s.kind==='Flatten'||s.settings.shape)return 'reindex';
 return 'transform';
}
export const motionNames:Record<string,string>={dense:'全连接 · 每条线是一项权重',convolution:'卷积 · 感受野扫描与局部汇聚',activation:'激活 · 函数曲线中的输入映射',normalize:'归一化 · 中心化与尺度收缩',reduce:'归约 · 多个输入汇入一个结果',merge:'相加 · 对齐坐标后合并',gate:'门控 · 逐坐标调制',noise:'采样 · 均匀变量变换为高斯噪声',contract:'矩阵乘法 · 沿共享维加权归约',reindex:'重排 · 坐标移动，数值保持',transform:'逐元素变换'};

/** Abstract operator diagrams live inside the inter-layer corridor. Only tensor
 * fields represent data; these curves/frames express the operation's function. */
export function operatorBridge(parent:T.Group,s:Step,from:Position3,to:Position3,width:number){
 const family=operationFamily(s),group=new T.Group();parent.add(group);
 const a=new T.Vector3(...from),b=new T.Vector3(...to),mid=a.clone().lerp(b,.5),span=Math.min(2,Math.max(.8,width*.35)),gap=a.distanceTo(b),links:Connection[]=[];
 const point=(x:number,y:number,z=0)=>mid.clone().add(new T.Vector3(x,y,z)).toArray() as Position3;
 let path:Position3[]=[];
 if(family==='activation'){
  const erf=(z:number)=>{const sign=z<0?-1:1,t=1/(1+.3275911*Math.abs(z));return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-z*z))};
  const fn=(x:number)=>s.kind==='ReLU'?Math.max(0,x):s.kind==='ReLU6'?Math.min(6,Math.max(0,x)):s.kind==='LeakyReLU'?(x<0?.2*x:x):s.kind==='Sigmoid'?1/(1+Math.exp(-x)):s.kind==='Tanh'?Math.tanh(x):s.kind==='SiLU'?x/(1+Math.exp(-x)):s.kind==='Hardsigmoid'?Math.max(0,Math.min(6,x+3))/6:s.kind==='Hardswish'?x*Math.max(0,Math.min(6,x+3))/6:s.kind==='GELU'?.5*x*(1+erf(x/Math.SQRT2)):Math.log1p(Math.exp(x));
  path=Array.from({length:49},(_,i)=>{const x=-3+i/8;return point(x*span/3,fn(x)*Math.min(span,gap*.16)/3)});
  links.push({from:point(-span,0),to:point(span,0)},{from:point(0,-span*.3),to:point(0,span)});
 }else{
  for(let i=0;i<7;i++){const x=(i-3)*span/3,z=(i%2)*.15;
   if(family==='reduce'||family==='contract'||family==='merge')links.push({from:point(x,gap*.19,z),to:point(0,-gap*.19)});
   else if(family==='reindex')links.push({from:point(x,gap*.19,z),to:point(-x,-gap*.19,.15-z)});
   else if(family==='normalize')links.push({from:point(x,gap*.19,z),to:point(x*.4,-gap*.19,z)});
   else if(family==='gate')links.push({from:point(x,gap*.19,z),to:point(x,-gap*.19,z),weight:i%3?1:-.4});
   else links.push({from:point(x,gap*.19,z),to:point(x*.7,-gap*.19,z)});
  }
 }
 for(let i=1;i<path.length;i++)links.push({from:path[i-1],to:path[i]});
 const fabric=connectionFabric(group,links);
 const frame=family==='convolution'?new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(span*1.3,.35,span*1.3)),new T.LineBasicMaterial({color:'#dceef7',transparent:true,opacity:.6})):undefined;
 if(frame){group.add(frame);frame.position.copy(mid)}
 return {group,family,update:(phase:number,active=false)=>{fabric.update(phase,-1,-1,active?1:.55);if(frame)frame.position.x=mid.x+Math.sin(phase*Math.PI*2)*span*.6}};
}
