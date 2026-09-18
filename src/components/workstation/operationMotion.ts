import * as T from 'three'
import type {Position3} from '../mmhvae/crystalPrimitives'

export interface Connection {from:Position3;to:Position3;weight?:number;output?:number;term?:number}
/** A connection is a weighted dependency, not a proxy neuron. GPU pulses travel
 * only through the current weighted dependency; output/term selection picks out the actual arithmetic. */
export function connectionFabric(parent:T.Group,links:Connection[]){
 const geometry=new T.BufferGeometry(),positions:number[]=[],along:number[]=[],weights:number[]=[],outputs:number[]=[],terms:number[]=[];
 for(const l of links){positions.push(...l.from,...l.to);along.push(0,1);weights.push(l.weight??1,l.weight??1);outputs.push(l.output??-1,l.output??-1);terms.push(l.term??-1,l.term??-1)}
 geometry.setAttribute('position',new T.Float32BufferAttribute(positions,3));geometry.setAttribute('along',new T.Float32BufferAttribute(along,1));geometry.setAttribute('weight',new T.Float32BufferAttribute(weights,1));geometry.setAttribute('outputIndex',new T.Float32BufferAttribute(outputs,1));geometry.setAttribute('termIndex',new T.Float32BufferAttribute(terms,1));
 const material=new T.ShaderMaterial({transparent:true,depthWrite:false,uniforms:{phase:{value:0},selectedOutput:{value:-1},selectedTerm:{value:-1},strength:{value:1},density:{value:1/Math.sqrt(Math.max(1,links.length/64))}},vertexShader:`attribute float along,weight,outputIndex,termIndex; varying float vAlong,vWeight,vOutput,vTerm; void main(){vAlong=along;vWeight=weight;vOutput=outputIndex;vTerm=termIndex;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,fragmentShader:`uniform float phase,selectedOutput,selectedTerm,strength,density;varying float vAlong,vWeight,vOutput,vTerm;void main(){float chosen=1.-step(.1,abs(vOutput-selectedOutput));float current=chosen*(1.-step(.1,abs(vTerm-selectedTerm)));float pulse=pow(max(0.,1.-abs(fract(vAlong-phase*1.6)-.5)*8.),2.);float magnitude=.22+.78*min(1.,abs(vWeight));vec3 tint=vWeight<0.?vec3(.94,.62,.39):vec3(.34,.78,.93);tint=mix(tint,vec3(.88,.98,1.),current*.8);float opacity=(.02*magnitude*density+chosen*.07+current*.5+pulse*current*.4)*strength;gl_FragColor=vec4(tint,opacity);}`});
 const mesh=new T.LineSegments(geometry,material);parent.add(mesh);
 return {mesh,update:(phase:number,output=-1,term=-1,strength=1)=>{material.uniforms.phase.value=phase;material.uniforms.selectedOutput.value=output;material.uniforms.selectedTerm.value=term;material.uniforms.strength.value=strength},dispose:()=>{parent.remove(mesh);geometry.dispose();material.dispose()}};
}
