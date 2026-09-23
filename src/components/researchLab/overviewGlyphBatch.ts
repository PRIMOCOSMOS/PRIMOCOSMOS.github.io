import * as T from 'three'
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js'
import {makeGlyph,disposeGroup} from '../mmhvae/glyphs'
import type {OverviewAppearance,OverviewGlyph} from './overviewAppearance'
import type {Position3} from '../mmhvae/crystalPrimitives'

export interface OverviewItem extends OverviewAppearance {id:string;position:Position3}
const scales:Partial<Record<OverviewGlyph,number>>={encoder:4.5,image:4.5,conv:4,linear:3.5,gaussian:3.2,poe:3,sample:3,decoder:2.8}
const glyphScale=(kind:OverviewGlyph)=>scales[kind]??2.8
function glyph(kind:OverviewGlyph,color:string){
 if(kind==='encoder')return makeGlyph('network',color);
 if(kind==='decoder'){
  const g=makeGlyph('up',color),stack=makeGlyph('tensor',color);stack.scale.set(.7,.7,.7);stack.position.y=.72;g.add(stack);return g;
 }
 if(kind==='attention'){
  const g=makeGlyph('linear',color),ring=makeGlyph('poe',color);ring.scale.setScalar(.65);ring.position.z=.55;g.add(ring);return g;
 }
 if(kind==='fourier'){
  const g=new T.Group(),mat=new T.LineBasicMaterial({color,transparent:true,opacity:.9});
  for(const z of [-.25,.25])g.add(new T.Line(new T.BufferGeometry().setFromPoints(Array.from({length:49},(_,i)=>new T.Vector3(i/24-1,Math.sin(i/48*Math.PI*4)*.42,z))),mat));return g;
 }
 if(kind==='loss')return makeGlyph('sum',color);
 return makeGlyph(kind,color);
}

/** Reuse the original semantic miniatures, baked into two geometry batches.
 * Only one active miniature is drawn separately, regardless of model size. */
export function overviewGlyphBatch(parent:T.Group,items:OverviewItem[]){
 const meshes:T.BufferGeometry[]=[],lines:number[]=[],lineColors:number[]=[];
 for(const item of items){const g=glyph(item.glyph,item.color);g.scale.setScalar(glyphScale(item.glyph));g.position.set(...item.position);g.updateMatrixWorld(true);
  g.traverse(o=>{
   if(o instanceof T.Mesh){const geo=o.geometry.index?o.geometry.toNonIndexed():o.geometry.clone();geo.applyMatrix4(o.matrixWorld);for(const name of Object.keys(geo.attributes))if(!['position','normal'].includes(name))geo.deleteAttribute(name);if(!geo.getAttribute('normal'))geo.computeVertexNormals();const color=(o.material as T.MeshStandardMaterial).color??new T.Color(item.color),colors=Array.from({length:geo.getAttribute('position').count},()=>color.toArray()).flat();geo.setAttribute('color',new T.Float32BufferAttribute(colors,3));meshes.push(geo)}
   else if(o instanceof T.Line){const a=o.geometry.getAttribute('position'),stride=o instanceof T.LineSegments?2:1,color=new T.Color(item.color);for(let i=0;i<a.count-1;i+=stride){for(const j of [i,i+1]){const p=new T.Vector3().fromBufferAttribute(a,j).applyMatrix4(o.matrixWorld);lines.push(...p.toArray());lineColors.push(...color.toArray())}}}
  });disposeGroup(g);
 }
 const group=new T.Group();parent.add(group);
 if(meshes.length){const combined=mergeGeometries(meshes);meshes.forEach(g=>g.dispose());group.add(new T.Mesh(combined,new T.MeshStandardMaterial({vertexColors:true,roughness:.3,metalness:.25,transparent:true,opacity:.66,depthWrite:false,side:T.DoubleSide})))}
 const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(lines,3));geo.setAttribute('color',new T.Float32BufferAttribute(lineColors,3));group.add(new T.LineSegments(geo,new T.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.85,depthWrite:false})));
 let active:T.Group|undefined,activeId='',activeScale=1;
 return {group,update:(id:string,phase:number)=>{
  if(id!==activeId){if(active){group.remove(active);disposeGroup(active)}activeId=id;const item=items.find(i=>i.id===id);active=item?glyph(item.glyph,new T.Color(item.color).lerp(new T.Color('#ffffff'),.22).getStyle()):undefined;if(active&&item){active.position.set(...item.position);activeScale=glyphScale(item.glyph);group.add(active)}}
  if(active){active.scale.setScalar(activeScale*(1+.055*Math.sin(phase*Math.PI*2)));}
 }};
}
