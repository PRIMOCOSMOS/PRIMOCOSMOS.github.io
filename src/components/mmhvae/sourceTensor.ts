import * as T from 'three'
import {makeGlyph} from './glyphs'
import type {Glyph} from './anatomy'

/** Axis structure, not invented activations. Large/symbolic extents stay
 * continuous and their exact source dimensions remain in the adjacent label. */
export function sourceTensor(glyph:Glyph,shape:string,color:string){
 const output=shape.split(/→|⇒/).at(-1)!.trim().replace(/(\d+|[DHW])³/g,'$1 × $1 × $1').replace(/(\d+|[HW])²/g,'$1 × $1')
 const match=output.match(/(?:\([^)]{1,20}\)|[A-Za-z]\w*|\d+)(?:\s*[×x]\s*(?:\([^)]{1,20}\)|[A-Za-z]\w*|\d+)){1,4}/)
 if(!match||!['tensor','image','network','conv','depthwise','linear','up','pool'].includes(glyph))return makeGlyph(glyph,color)
 const axes=match[0].split(/\s*[×x]\s*/),volume=axes.length===5||/³|\bD\s*×\s*H\s*×\s*W\b/.test(shape.split(/→|⇒/).at(-1)!)
 const sequence=/^[BN]\b/.test(axes[0])&&axes.length<=3
 const group=new T.Group();group.userData.tensorAxes=axes;group.userData.volume=volume;group.userData.extentOnly=true
 const plate=(x:number,y:number,z:number,w:number,h:number,d:number)=>{
  const geo=new T.BoxGeometry(w,h,d),mesh=new T.Mesh(geo,new T.MeshPhysicalMaterial({color,transparent:true,opacity:.11,roughness:.12,metalness:.06,clearcoat:1,depthWrite:false}))
  const wire=new T.LineSegments(new T.EdgesGeometry(geo),new T.LineBasicMaterial({color,transparent:true,opacity:.65,depthWrite:false}));mesh.position.set(x,y,z);wire.position.copy(mesh.position);group.add(mesh,wire)
 }
 if(volume){
  plate(0,0,0,1.8,1.6,1.8)
  // Section planes describe the depth axis, not a reduced number of voxels.
  for(const y of [-.55,0,.55])plate(0,y,0,1.8,.018,1.8)
 }else if(sequence){plate(0,0,0,2.2,.07,1.4);plate(0,-.24,0,2.2,.035,1.4)}
 else if(axes.length>=3){for(const y of [-.3,0,.3])plate(0,y,0,1.9,.06,1.9)}
 else plate(0,0,0,2.2,.09,axes.length===2?1.15:.35)
 return group
}
