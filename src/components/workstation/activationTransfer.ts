import * as T from 'three'
import type {Step} from './engine'
import {createCrystalTensor,type Position3} from '../mmhvae/crystalPrimitives'
import {valueExtent,numericPalette} from '../mmhvae/numericPalette'
import {activationKinds} from './execution'

export const hasTransferPlot=(step:Step)=>step.kind==='SourceFunction'||activationKinds.has(step.kind)||['vae-exp','vae-square','vae-minus-one','vae-negative-log'].includes(String(step.settings.operation));
const transferValue=(step:Step,x:number)=>{
 if(step.settings.symbolic){if(String(step.settings.activation).includes('leaky01'))return x<0?.01*x:x;switch(step.settings.activation){case 'clamp':return 10*Math.tanh(x/10);case 'exp':return Math.exp(x);case 'exp-negative':return Math.exp(-x);case 'reciprocal':return 1/x;case 'temperature':return .5*x}}

 switch(step.settings.operation){case 'vae-exp':return Math.exp(x);case 'vae-square':return x*x;case 'vae-minus-one':return x-1;case 'vae-negative-log':return -Math.log(x);default:return activationValue(step.kind,x)}
};

export function activationValue(kind:string,x:number){
 const erf=(z:number)=>{const sign=z<0?-1:1,t=1/(1+.3275911*Math.abs(z));return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-z*z))}
 switch(kind){case 'ReLU':return Math.max(0,x);case 'ReLU6':return Math.min(6,Math.max(0,x));case 'LeakyReLU':return x<0?.2*x:x;case 'Sigmoid':return 1/(1+Math.exp(-x));case 'Tanh':return Math.tanh(x);case 'SiLU':return x/(1+Math.exp(-x));case 'GELU':return .5*x*(1+erf(x/Math.SQRT2));case 'Hardsigmoid':return Math.min(6,Math.max(0,x+3))/6;case 'Hardswish':return x*Math.min(6,Math.max(0,x+3))/6;case 'Softplus':return Math.max(0,x)+Math.log1p(Math.exp(-Math.abs(x)));default:return x}
}

/** An actual transfer plot in the inter-layer space: every data point is
 * (x_i, f(x_i)); incoming/outgoing paths preserve that neuron's index. */
export function activationTransfer(parent:T.Group,step:Step,inputs:Position3[],outputs:Position3[]){
 const group=new T.Group();parent.add(group);const x=step.inputs[0].values,y=step.output.values;
 // Show the characteristic bend even when this batch contains only small
 // values. The data point and curve share the same honest coordinate map.
 const positiveDomain=step.settings.operation==='vae-negative-log'||step.settings.activation==='reciprocal';
 const extent=Math.max(step.kind==='ReLU6'?8:step.settings.operation?3:6,...x.map(Math.abs));
 const low=positiveDomain?.05: -extent,high=extent,xmid=(low+high)/2,xrange=(high-low)/2;
 const samples=Array.from({length:65},(_,i)=>transferValue(step,low+(high-low)*i/64));
 const ymin=Math.min(...samples),ymax=Math.max(...samples),ymid=(ymin+ymax)/2,yrange=Math.max(1e-6,(ymax-ymin)/2);
 const pitch=inputs.length>1?Math.min(...inputs.slice(1).map((p,i)=>Math.abs(p[0]-inputs[i][0])).filter(v=>v>.01)):1;
 const width=Math.max(.12,Math.min(.86,Number.isFinite(pitch)?pitch*.42:.86));
 const height=Math.min(1.18,...inputs.map((p,i)=>Math.abs(p[1]-outputs[i][1])*.3));
 const centers=inputs.map((p,i)=>p.map((v,j)=>(v+outputs[i][j])/2) as Position3);
 const map=(center:Position3,value:number,result:number):Position3=>[center[0]+(value-xmid)/xrange*width,center[1]+(result-ymid)/yrange*height,center[2]];
 const positions:Position3[]=x.map((v,i)=>map(centers[i],v,y[i]));
 const points=createCrystalTensor(group,x.length,.24,{valueEdges:true,bodyOpacity:.32,edgeOpacity:.8,valueScale:valueExtent(y)});points.update(y,positions,{focus:-1});points.body.userData.tensor=step.output.id;points.body.userData.scope=step.id;
 const segments:T.Vector3[]=[],axes:T.Vector3[]=[];for(const center of centers){
  for(let i=0;i<64;i++){for(const j of [i,i+1]){const value=low+(high-low)*j/64;segments.push(new T.Vector3(...map(center,value,samples[j])))}}
  axes.push(new T.Vector3(...map(center,low,0)),new T.Vector3(...map(center,high,0)),new T.Vector3(...map(center,positiveDomain?low:0,ymin)),new T.Vector3(...map(center,positiveDomain?low:0,ymax)));
 }
 group.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(axes),new T.LineBasicMaterial({color:'#8fb8c7',transparent:true,opacity:.12})));
 group.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(segments),new T.LineBasicMaterial({color:numericPalette.relation,transparent:true,opacity:.68})));
 if(step.settings.symbolic)points.group.visible=false;
 const inputScale=valueExtent(x),outputScale=valueExtent(y);
 const moving=createCrystalTensor(group,x.length,.22,{valueEdges:true,bodyOpacity:.36,edgeOpacity:.8,valueScale:1});
 return {group,positions,body:points.body,motion:moving.group,update:(phase:number)=>{
  const transformed:number[]=[],travellers:Position3[]=x.map((value,i)=>{const t=(phase+i*.075)%1;if(step.settings.symbolic){const v=low+(high-low)*t;transformed.push(.8);return map(centers[i],v,transferValue(step,v))}const a=t<.5?inputs[i]:positions[i],b=t<.5?positions[i]:outputs[i],u=t<.5?t*2:(t-.5)*2,e=u*u*(3-2*u);transformed.push(t<.5?value/inputScale:y[i]/outputScale);return [a[0]+(b[0]-a[0])*e,a[1]+(b[1]-a[1])*e,a[2]+(b[2]-a[2])*e]});moving.update(transformed,travellers,{focus:-1});
 }}
}
