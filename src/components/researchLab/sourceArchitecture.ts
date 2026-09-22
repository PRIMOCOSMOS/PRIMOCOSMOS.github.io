import type {AnatomyEdge,AnatomyPart,Glyph} from '../mmhvae/anatomy'
import {LEVELS,NODE_MAP} from '../mmhvae/model'
import {anatomyFor} from '../mmhvae/anatomy'
import {mathSpec,type MathSpec} from '../mmhvae/mathematics'
import {atomId,partName} from '../mmhvae/navigation'
import {sourceWindow} from '../mmhvae/sourceTensor'
import type {PaperModel} from './catalog'

export interface SourceModule {
 id:string;title:string;shape:string;glyph:Glyph;formula:string;source:string;
 operator?:MathSpec;children:SourceModule[];edges:AnatomyEdge[];expand?:()=>SourceModule;
}
/** The tree is compiled from source-audited entries. Repeated source blocks stay
 * separate instances; parameter sharing remains in the source edges/notes. */
export function paperModule(model:PaperModel,id:string,depth=2):SourceModule{
 const e=model.entries[id];
 return {id:e.id,title:e.title,shape:e.shape,glyph:e.glyph,formula:e.formula,source:`${e.file}:${e.line} · ${e.symbol}`,operator:e.math,children:depth>0?e.children.map(id=>paperModule(model,id,depth-1)):[],edges:e.edges,expand:depth===0&&e.children.length?()=>paperModule(model,id,1):undefined};
}
export function mmhvaeModule(part:AnatomyPart,parent:string,observed:string[],depth=2):SourceModule{
 const id=part.child??atomId(parent,part.id),module:SourceModule={id,title:partName(part),shape:part.shape,glyph:part.glyph,formula:'',source:part.sourceName,children:[],edges:[]};
 if(part.child&&depth>0){const graph=anatomyFor(part.child,observed);module.children=graph.parts.filter(p=>!p.role).map(p=>mmhvaeModule(p,part.child!,observed,depth-1));const internal=graph.parts.filter(p=>!p.role);for(let i=0;i<internal.length;i++){const p=internal[i];if(/尺寸不变/.test(p.shape)||p.glyph==='se'){const edge=graph.edges.find(e=>e.to===p.id&&!e.residual),sourceIndex=internal.findIndex(v=>v.id===edge?.from);if(sourceIndex>=0)module.children[i].shape=module.children[sourceIndex].shape}}module.edges=graph.edges.filter(e=>graph.parts.some(p=>p.id===e.from&&!p.role)&&graph.parts.some(p=>p.id===e.to&&!p.role)).map(e=>({...e,from:module.children[graph.parts.filter(p=>!p.role).findIndex(p=>p.id===e.from)]?.id,to:module.children[graph.parts.filter(p=>!p.role).findIndex(p=>p.id===e.to)]?.id})).filter(e=>e.from&&e.to)}
 else if(!part.child){module.operator=mathSpec(part,'');const node=NODE_MAP.get(parent.split('/')[0]),level=LEVELS.find(l=>l.l===node?.l),s=level?.encoderSize;const ch=part.shape.match(/(?:^|→\s*)(\d+)\s*ch/);if(/尺寸不变/.test(part.shape)){const graph=anatomyFor(parent,observed),edge=graph.edges.find(e=>e.to===part.id&&!e.residual),previous=graph.parts.find(p=>p.id===edge?.from);if(previous&&previous.id!==part.id)module.shape=mmhvaeModule(previous,parent,observed,0).shape}if(ch&&s&&!parent.includes('/se/')){const spatial=node?.kind==='down'?Math.floor((s+2*module.operator.padding-module.operator.kernel)/module.operator.stride)+1:s;module.shape=`${ch[1]} × ${spatial} × ${spatial}`}
  if((part.glyph==='conv'||part.glyph==='depthwise'||part.glyph==='norm')&&!parent.includes('/se/')){const graph=anatomyFor(parent,observed),edge=graph.edges.find(e=>e.to===part.id&&!e.residual),previous=graph.parts.find(p=>p.id===edge?.from);if(previous&&previous.id!==part.id){const input=mmhvaeModule(previous,parent,observed,0),dims=sourceWindow(input.shape,input.glyph).dimensions;if(dims.length>=3&&dims.slice(-2).every(n=>n!==null)){if(part.glyph==='norm')module.shape=input.shape;else if(ch){const op=module.operator,spatial=dims.slice(-2).map(n=>Math.floor((n!+2*op.padding-op.kernel)/op.stride)+1);module.shape=`${ch[1]} × ${spatial.join(' × ')}`}}}}
 }

 else module.expand=()=>mmhvaeModule(part,parent,observed,1);
 return module;
}
