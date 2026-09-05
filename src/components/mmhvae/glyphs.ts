import * as T from 'three'
import type { Glyph } from './anatomy'

/** Semantic miniatures; layer counts are illustrative, dimensions are labelled separately. */
export function makeGlyph(kind: Glyph, color: string) {
  const g=new T.Group(), ink=new T.Color(color)
  const solid=new T.MeshStandardMaterial({color:ink,metalness:.35,roughness:.38,transparent:true,opacity:.65})
  const wire=new T.LineBasicMaterial({color,transparent:true,opacity:.85})
  const mesh=(geo:T.BufferGeometry,x=0,y=0,z=0)=>{const m=new T.Mesh(geo,solid);m.position.set(x,y,z);g.add(m);return m}
  const line=(points:number[][])=>g.add(new T.Line(new T.BufferGeometry().setFromPoints(points.map(p=>new T.Vector3(...p as [number,number,number]))),wire))
  const ring=(r:number,z=0)=>{const m=mesh(new T.TorusGeometry(r,.025,6,40),0,0,z);return m}
  const plate=(x:number,y:number,z:number,w=1,h=.9)=>{const geo=new T.PlaneGeometry(w,h);const m=mesh(geo,x,y,z);m.material=new T.MeshStandardMaterial({color,side:T.DoubleSide,transparent:true,opacity:.22});const e=new T.LineSegments(new T.EdgesGeometry(geo),wire);e.position.copy(m.position);g.add(e)}
  if(kind==='gaussian'){
    const points=Array.from({length:25},(_,i)=>{const r=i/24*.8;return new T.Vector2(r,Math.exp(-r*r*7)*1.15-.45)})
    const bell=mesh(new T.LatheGeometry(points,32));bell.rotation.x=.32;ring(.78).rotation.x=Math.PI/2;line([[0,-.5,0],[0,.85,0]])
    for(const r of [.28,.5,.7]){const contour=ring(r);contour.rotation.x=Math.PI/2;contour.position.y=Math.exp(-r*r*7)*1.15-.45}
  }else if(kind==='poe'){
    [0,Math.PI/3,-Math.PI/3].forEach(a=>{const r=ring(.67);r.rotation.y=a;r.rotation.x=.4})
    mesh(new T.IcosahedronGeometry(.22,1));line([[-1,.6,0],[0,0,0],[1,0,0]]);line([[-1,-.6,0],[0,0,0]])
  }else if(kind==='sample'){
    mesh(new T.OctahedronGeometry(.47));ring(.78).rotation.y=.55
    for(let i=0;i<13;i++){const a=i*2.399;mesh(new T.SphereGeometry(.035,5,5),Math.cos(a)*(.55+i*.02),Math.sin(a)*(.55+i*.02),Math.sin(i)*.3)}
  }else if(kind==='norm'||kind==='sum'||kind==='multiply'){
    ring(.57);if(kind==='norm'){ring(.4,.22);ring(.23,.44)}
    else {line([[-.28,0,0],[.28,0,0]]);line([[0,-.28,0],[0,.28,0]]);if(kind==='multiply')g.rotation.z=Math.PI/4}
  }else if(kind==='activation'){
    line([[-.8,-.5,0],[.8,-.5,0]]);line([[0,-.65,0],[0,.8,0]])
    line(Array.from({length:33},(_,i)=>{const x=i/32*1.6-.8;return[x,1.2*x/(1+Math.exp(-5*x))-.15,0]}))
  }else if(kind==='linear'){
    for(let i=0;i<4;i++){mesh(new T.SphereGeometry(.085,8,8),-.6,(i-1.5)*.36);for(let j=0;j<3;j++)line([[-.6,(i-1.5)*.36,0],[.6,(j-1)*.48,0]])}
    for(let j=0;j<3;j++)mesh(new T.SphereGeometry(.1,8,8),.6,(j-1)*.48)
  }else if(kind==='depthwise'){
    for(let i=0;i<3;i++){
      const x=(i-1)*.47;mesh(new T.CylinderGeometry(.13,.13,.82,12),x,0,0)
      line([[x,-.8,0],[x,.8,0]])
      for(const y of [-.58,.58]){const disc=ring(.19);disc.position.set(x,y,0);disc.rotation.x=Math.PI/2.5}
    }
  }else if(kind==='up'||kind==='pool'||kind==='se'){
    const up=kind==='up';plate(0,-.48,0,up?.55:1.4,.3);plate(0,.48,0,up?1.4:.55,.3)
    for(const x of [-1,1])line([[x*(up?.275:.7),-.48,0],[x*(up?.7:.275),.48,0]])
    if(kind==='se'){ring(.25, .2);line([[0,.6,0],[.8,.6,0],[.8,-.6,0]])}
  }else if(kind==='split'||kind==='concat'){
    const s=kind==='split'?1:-1;line([[-.8*s,0,0],[0,0,0],[.7*s,.5,0]]);line([[0,0,0],[.7*s,-.5,0]])
    mesh(new T.SphereGeometry(.13,8,8));plate(.7*s,.5,0,.3,.3);plate(.7*s,-.5,0,.3,.3)
  }else {
    const count=kind==='image'?1:kind==='network'?4:3
    for(let i=0;i<count;i++){const z=(i-(count-1)/2)*.22;plate(i*.08,0,z,1.05,.85)
      if(kind==='conv'||kind==='image')for(let j=1;j<4;j++){line([[-.525+i*.08,-.425+j*.2125,z],[.525+i*.08,-.425+j*.2125,z]]);line([[-.525+j*.2625+i*.08,-.425,z],[-.525+j*.2625+i*.08,.425,z]])}}
    if(kind==='conv')mesh(new T.BoxGeometry(.3,.3,.07),.2,.1,.32)
    if(kind==='network')line([[-.7,0,0],[-.7,.7,0],[.85,.7,0],[.85,0,.3]])
  }
  return g
}

export function disposeGroup(group:T.Object3D){
  const geometries=new Set<T.BufferGeometry>(),materials=new Set<T.Material>(),textures=new Set<T.Texture>()
  group.traverse(o=>{const m=o as T.Mesh;if(m.geometry)geometries.add(m.geometry);if(m.material)(Array.isArray(m.material)?m.material:[m.material]).forEach(mat=>{materials.add(mat);const map=(mat as T.SpriteMaterial).map;if(map)textures.add(map)})})
  geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());textures.forEach(t=>t.dispose())
}
