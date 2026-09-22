import * as T from 'three'
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js'
import {sourceTensor} from '../mmhvae/sourceTensor'
import {disposeGroup} from '../mmhvae/glyphs'
import {createArrowStream} from '../mmhvae/crystalPrimitives'
import {makeScientificVisual} from './scientificVisuals'
import type {MathFrame,MathVisual} from '../mmhvae/mathVisuals'
import type {SourceModule} from './sourceArchitecture'

export interface AssemblyCell {module:SourceModule;position:T.Vector3;hit:T.Mesh;static:T.Group}
/** Same source composition at every depth. Frozen source tensor sections are
 * cheap; only the active function allocates its existing numerical visual. */
export function moduleAssembly(module:SourceModule,color:string,collapsed=false,page=0){
 if(collapsed&&module.children.length){const original=module;module={...original,children:[],expand:()=>original}}
 const group=new T.Group(),structure=new T.Group();group.add(structure);
 const cells:AssemblyCell[]=[],tracks:{from:string;to:string;curve:T.CatmullRomCurve3}[]=[],centers=new Map<string,T.Vector3>(),ports=new Map<string,{input:T.Vector3;output:T.Vector3}>();
 const draw=(node:SourceModule,x:number,y:number,z:number):number=>{
  if(!node.children.length){
   const position=new T.Vector3(x,y,z),staticGroup=sourceTensor(node.glyph,node.shape,color,page);staticGroup.position.copy(position);staticGroup.scale.setScalar(1.4);structure.add(staticGroup);
   const hit=new T.Mesh(new T.BoxGeometry(11,10,9),new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));hit.position.copy(position);hit.userData.moduleId=node.id;group.add(hit);cells.push({module:node,position,hit,static:staticGroup});centers.set(node.id,position);ports.set(node.id,{input:position.clone().add(new T.Vector3(0,2,0)),output:position.clone().add(new T.Vector3(0,-2,0))});return 10;
  }
  // Real child order is refined by the dependency DAG. Independent functions
  // are distinct tiers; shortcut edges never become spurious sequential edges.
  const pending=new Set(node.children.map(c=>c.id)),ordered:SourceModule[]=[];
  for(let pass=0;pass<node.children.length;pass++)for(const child of node.children)if(pending.has(child.id)&&node.edges.filter(e=>e.to===child.id&&!e.residual&&pending.has(e.from)).length===0){ordered.push(child);pending.delete(child.id)}
  ordered.push(...node.children.filter(c=>pending.has(c.id)));let offset=0;
  for(const child of ordered){const used=draw(child,x,y-offset,z);centers.set(child.id,new T.Vector3(x,y-offset-used/2+5,z));offset+=used}
  for(const edge of node.edges){const a=ports.get(edge.from)?.output,b=ports.get(edge.to)?.input;if(!a||!b)continue;const mid=a.clone().lerp(b,.5);if(edge.residual){mid.x+=7;mid.z+=2}const curve=new T.CatmullRomCurve3([a,mid,b]);tracks.push({from:edge.from,to:edge.to,curve});const line=new T.Line(new T.BufferGeometry().setFromPoints(curve.getPoints(16)),new T.LineBasicMaterial({color:edge.residual?'#bea2cc':color,transparent:true,opacity:.24}));structure.add(line)}
  const frame=new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(12,Math.max(8,offset-1),10)),new T.LineBasicMaterial({color,transparent:true,opacity:.13}));frame.position.set(x,y-offset/2+5,z);structure.add(frame);centers.set(node.id,frame.position.clone());if(ordered.length)ports.set(node.id,{input:ports.get(ordered[0].id)!.input,output:ports.get(ordered.at(-1)!.id)!.output});return offset;
 };
 const batches:T.Object3D[]=[];const height=draw(module,0,0,0);structure.position.y=0;
 // Merge static lines by material. Hundreds of reusable substructures no
 // longer require hundreds of individual line submissions.
 structure.updateMatrixWorld(true);const lines=new Map<string,{geometries:T.BufferGeometry[];material:T.LineBasicMaterial}>();
 structure.traverse(o=>{if(!(o instanceof T.Line)||!(o.material instanceof T.LineBasicMaterial))return;const material=o.material,key=`${material.color.getHex()}:${material.opacity}`;const item=lines.get(key)??{geometries:[],material:material.clone()},g=o.geometry.clone().applyMatrix4(o.matrixWorld);if(o instanceof T.LineSegments)item.geometries.push(g);else{const p=g.getAttribute('position'),points:T.Vector3[]=[];for(let i=1;i<p.count;i++)points.push(new T.Vector3().fromBufferAttribute(p,i-1),new T.Vector3().fromBufferAttribute(p,i));item.geometries.push(new T.BufferGeometry().setFromPoints(points));g.dispose()}lines.set(key,item);o.visible=false});
 for(const {geometries,material} of lines.values()){const merged=mergeGeometries(geometries.map(g=>g.index?g.toNonIndexed():g));const batch=new T.LineSegments(merged,material);group.add(batch);batches.push(batch);geometries.forEach(g=>g.dispose())}
 const solids=new Map<string,{geometries:T.BufferGeometry[];material:T.Material}>();structure.traverse(o=>{if(!(o instanceof T.Mesh)||Array.isArray(o.material)||o instanceof T.InstancedMesh)return;const m=o.material as T.MeshPhysicalMaterial,key=[m.type,m.color?.getHex(),m.opacity,m.roughness,m.metalness,Object.keys(o.geometry.attributes).sort().join(',')].join(':');const batch=solids.get(key)??{geometries:[],material:m.clone()},g=o.geometry.clone().applyMatrix4(o.matrixWorld);batch.geometries.push(g.index?g.toNonIndexed():g);if(g.index)g.dispose();solids.set(key,batch);o.visible=false});for(const {geometries,material} of solids.values()){const batch=new T.Mesh(mergeGeometries(geometries),material);group.add(batch);batches.push(batch);geometries.forEach(g=>g.dispose())}
 let active:string='',math:MathVisual|undefined,flow:ReturnType<typeof createArrowStream>|undefined,nested:ReturnType<typeof moduleAssembly>|undefined;
 const setActive=(id:string,frame:MathFrame)=>{
  const cell=cells.find(c=>c.module.id===id),operator=cell?.module.operator;
  if(active!==id){if(math){group.remove(math.group);disposeGroup(math.group);math=undefined}if(nested){group.remove(nested.group);disposeGroup(nested.group);nested=undefined}if(flow){group.remove(flow.group);disposeGroup(flow.group);flow=undefined}active=id;
   if(cell&&operator){math=makeScientificVisual({...operator,stage:operator.steps.length-1});math.group.position.copy(cell.position);group.add(math.group)}
   else if(cell?.module.expand){nested=moduleAssembly(cell.module.expand(),color,false,page);nested.group.scale.setScalar(8/nested.height);nested.group.position.copy(cell.position).add(new T.Vector3(0,(nested.height/2-5)*nested.group.scale.y,0));group.add(nested.group)}
   if(cell){flow=createArrowStream(group,'#b9e6e8',.15,2)}
  }
  if(math)math.update(frame);
  if(nested){const cursor=frame.phase*nested.cells.length,child=nested.cells[Math.min(nested.cells.length-1,Math.floor(cursor))];if(child)nested.setActive(child.module.id,{...frame,phase:cursor%1})}
  if(flow&&cell){const incoming=tracks.find(t=>t.to===id)||tracks.find(t=>t.from===id);if(incoming)flow.update(incoming.curve,frame.phase);else flow.group.visible=false}
  if(cells.length===1)for(const batch of batches)batch.visible=!math&&!nested;group.userData.activeFunction=cell?.module.id??'';group.userData.activeMath=Number(!!math)+(nested?.group.userData.activeMath??0);
  return cell;
 };
 return {group,cells,height,width:18,depth:12,setActive,clear:()=>setActive('',{phase:0,probe:.42,temperature:1})};
}
