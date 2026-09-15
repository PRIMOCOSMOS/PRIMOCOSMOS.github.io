import type {Step,Tensor,Term} from './engine'
interface Computation {terms:Term[];products?:Tensor;prefix?:Tensor;sums?:Tensor;bias?:Tensor;width?:number;productOffset?:number;elementwise?:boolean}

/** Derived intermediates of the actual selected computation; never proxy neurons. */
export function computation(step:Step,index:number):Computation{
 const terms=step.trace(index),weighted=terms.length>0&&terms.every(t=>t.factor!==undefined)
 const stage=(id:string,name:string,values:number[],shape=step.output.shape):Tensor=>({id:`calc-${id}`,name,shape,values});
 const sequence=(names:string[],values:number[][])=>({terms,products:stage('products',names[0],values[0]),prefix:stage('prefix',names[1],values[1]),sums:values[2]?stage('sums',names[2],values[2]):undefined,width:values[0].length,productOffset:0,elementwise:true});
 if(step.kind==='MaxPool2d'||step.settings.operation==='row-max'){
  const values=terms.map(t=>t.value);let maximum=-Infinity;
  return {terms,products:stage('products','窗口中全部比较候选',values,[1,values.length]),prefix:stage('prefix','逐项比较 · 当前最大值',values.map(v=>maximum=Math.max(maximum,v)),[1,values.length]),width:values.length,productOffset:0};
 }
 if(['ReLU','ReLU6','LeakyReLU','Sigmoid','Tanh','SiLU','GELU','Hardswish','Hardsigmoid','Softplus'].includes(step.kind)){
  const x=step.inputs[0].values,k=step.kind;
  if(['Sigmoid','SiLU'].includes(k)){const ex=x.map(v=>Math.exp(-v)),den=ex.map(v=>1+v);return sequence(['指数 exp(−x)','分母 1 + exp(−x)','Sigmoid 门控 1 / 分母'],[ex,den,den.map(v=>1/v)])}
  if(k==='GELU'){const cdf=x.map((v,i)=>v===0?.5:step.output.values[i]/v);return sequence(['标准化 x / √2','正态累计概率 Φ(x)'],[x.map(v=>v/Math.SQRT2),cdf])}
  if(k==='Tanh')return sequence(['绝对值指数 exp(−2|x|)','幅度 (1−e) / (1+e)'],[x.map(v=>Math.exp(-2*Math.abs(v))),x.map(v=>Math.tanh(Math.abs(v)))]);
  if(k==='Softplus')return sequence(['稳定项 exp(−|x|)','log1p(exp(−|x|))','正半轴 max(0,x)'],[x.map(v=>Math.exp(-Math.abs(v))),x.map(v=>Math.log1p(Math.exp(-Math.abs(v)))),x.map(v=>Math.max(0,v))]);
  if(k==='Hardsigmoid'||k==='Hardswish')return sequence(['平移 x + 3','截断 min(6,max(0,x+3))','线性门控 / 6'],[x.map(v=>v+3),x.map(v=>Math.min(6,Math.max(0,v+3))),x.map(v=>Math.min(6,Math.max(0,v+3))/6)]);
  return sequence(['分支判定 x > 0','分支选值'],[x.map(v=>v>0?1:0),x.map(v=>k==='LeakyReLU'&&v<0?v*.2:Math.max(0,v))]);
 }
 if(['BatchNorm2d','InstanceNorm2d','LayerNorm','GroupNorm','RMSNorm'].includes(step.kind)){
  const ts=step.output.values.map((_,i)=>step.trace(i)),eps=Number(step.settings.eps);
  return sequence(['中心化 x − μ','标准差 √(v + ε)','缩放 γ · (x−μ) / √(v+ε)'],[ts.map(t=>t[0].value-t[1].value),ts.map(t=>Math.sqrt(t[2].value+eps)),ts.map(t=>(t[0].value-t[1].value)/Math.sqrt(t[2].value+eps)*t[3].value)]);
 }
 if(step.settings.operation==='add'||step.settings.operation==='mul'){
  const ts=step.output.values.map((_,i)=>step.trace(i));return sequence(['左操作数 · 对齐输出坐标','右操作数 · 按真实单例维广播'],[ts.map(t=>t[0].value),ts.map(t=>t[1].value)]);
 }
 if(!weighted)return {terms}
 if(step.kind==='Linear'){
  const [x,w,b]=step.inputs,n=x.shape.at(-1)!,out=w.shape[0],rows=x.values.length/n
  const values=Array.from({length:rows*out*n},(_,i)=>{const j=i%n,o=Math.floor(i/n)%out,r=Math.floor(i/(n*out));return x.values[r*n+j]*w.values[o*n+j]})
  const products:Tensor={id:'calc-products',name:'全部逐项乘积 xᵢWₒᵢ',shape:[...x.shape.slice(0,-1),out,n],values}
  let total=0;const prefix:Tensor={id:'calc-prefix',name:'所选输出的逐项累加',shape:[1,n],values:values.slice(index*n,(index+1)*n).map(v=>total+=v)}
  const sums:Tensor={id:'calc-sums',name:'各输出的点积（加偏置之前）',shape:step.output.shape,values:Array.from({length:rows*out},(_,i)=>values.slice(i*n,(i+1)*n).reduce((a,b)=>a+b,0))}
  return {terms,products,prefix,sums,bias:b,width:n,productOffset:index*n}
 }
 const values=terms.map(t=>t.value*t.factor!);let total=0
 return {terms,products:{id:'calc-products',name:'所选输出的全部乘积贡献',shape:[1,values.length],values} as Tensor,prefix:{id:'calc-prefix',name:'顺序累加结果',shape:[1,values.length],values:values.map(v=>total+=v)} as Tensor,width:values.length,productOffset:0}
}

export function scalarEquation(step:Step,index:number,progress:number){
 const terms=step.trace(index),at=Math.min(terms.length-1,Math.floor(progress*terms.length)),t=terms[at],format=(v:number)=>Number.isFinite(v)?v.toPrecision(4):String(v)
 if(!t)return `${step.title}：${format(step.output.values[index])}`
 if(step.kind==='MaxPool2d'||step.settings.operation==='row-max')return `max(${terms.slice(0,at+1).map(t=>format(t.value)).join(', ')}) = ${format(Math.max(...terms.slice(0,at+1).map(t=>t.value)))}`;
 if(step.settings.operation==='add'||step.settings.operation==='mul')return `${format(terms[0].value)} ${step.settings.operation==='add'?'+':'×'} ${format(terms[1].value)} = ${format(step.output.values[index])}`;
 if(['ReLU','ReLU6','LeakyReLU','Sigmoid','Tanh','SiLU','GELU','Hardswish','Hardsigmoid','Softplus'].includes(step.kind))return `${step.kind}(${format(terms[0].value)}) = ${format(step.output.values[index])}`;
 if(['BatchNorm2d','InstanceNorm2d','LayerNorm','GroupNorm','RMSNorm'].includes(step.kind))return `${format(terms[3].value)} × (${format(terms[0].value)} − ${format(terms[1].value)}) / √(${format(terms[2].value)} + ${step.settings.eps}) + ${format(terms[4]?.value??0)} = ${format(step.output.values[index])}`;
 if(t.factor!==undefined){const before=terms.slice(0,at).reduce((s,t)=>s+t.value*(t.factor??1),0),product=t.value*t.factor;return `${format(before)} + (${format(t.value)} × ${format(t.factor)}) = ${format(before+product)}`}
 return `${step.title} · 读取 ${at+1}/${terms.length}：${format(t.value)} → 输出 ${format(step.output.values[index])}`
}
