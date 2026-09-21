import * as T from 'three'
import type {Step} from './engine'
import {coords} from './engine'
import type {Position3} from '../mmhvae/crystalPrimitives'

/** Actual trace indices account for stride, padding, dilation, groups and transpose. */
export function receptiveField(step:Step,outputIndex:number){
 const input=step.inputs[0],indices=[...new Set(step.trace(outputIndex).filter(t=>t.tensor===input.id&&t.index>=0).map(t=>t.index))];
 const channels=new Map<string,number[]>();for(const i of indices){const c=coords(i,input.shape),key=c.slice(0,2).join(':');channels.set(key,[...channels.get(key)??[],i])}
 return {indices,channels,output:coords(outputIndex,step.output.shape),padding:step.trace(outputIndex).filter(t=>t.tensor===input.id&&t.index<0).length};
}

export function receptiveFieldOverlay(parent:T.Group){
 const group=new T.Group();parent.add(group);group.visible=false;
 const boxes:T.LineSegments[]=[],faces:T.Mesh[]=[];
 const boxGeometry=new T.EdgesGeometry(new T.BoxGeometry(1,1,1)),fillGeometry=new T.BoxGeometry(1,1,1);
 let key='';
 const output=new T.LineSegments(boxGeometry,new T.LineBasicMaterial({color:'#f4e1ae',transparent:true,opacity:.95,depthTest:false}));group.add(output);
 return {group,update:(id:string,patches:Position3[][],target:Position3,phase:number)=>{
  group.visible=true;
  if(key!==id){key=id;for(let i=0;i<patches.length;i++){
   if(!boxes[i]){boxes[i]=new T.LineSegments(boxGeometry,new T.LineBasicMaterial({color:'#82eddf',transparent:true,opacity:.9}));faces[i]=new T.Mesh(fillGeometry,new T.MeshBasicMaterial({color:'#75dccd',transparent:true,opacity:.045,depthWrite:false}));group.add(boxes[i],faces[i])}
   const bounds=new T.Box3().setFromPoints(patches[i].map(p=>new T.Vector3(...p))).expandByScalar(.26),size=bounds.getSize(new T.Vector3()),center=bounds.getCenter(new T.Vector3());
   boxes[i].position.copy(center);boxes[i].scale.copy(size);faces[i].position.copy(center);faces[i].scale.copy(size);
  }boxes.forEach((b,i)=>{b.visible=i<patches.length;faces[i].visible=b.visible})}
  output.position.set(...target);output.scale.setScalar(.62);
  for(const b of boxes)(b.material as T.LineBasicMaterial).opacity=.65+.25*(.5+.5*Math.sin(phase*Math.PI*2));
 }};
}
