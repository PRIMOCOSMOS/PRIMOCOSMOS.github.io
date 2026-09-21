import type {AnatomyEdge,AnatomyPart,Glyph} from '../mmhvae/anatomy'
import {anatomyFor} from '../mmhvae/anatomy'
import {mathSpec,type MathSpec} from '../mmhvae/mathematics'
import {atomId,partName} from '../mmhvae/navigation'
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
 if(part.child&&depth>0){const graph=anatomyFor(part.child,observed);module.children=graph.parts.filter(p=>!p.role).map(p=>mmhvaeModule(p,part.child!,observed,depth-1));module.edges=graph.edges.filter(e=>graph.parts.some(p=>p.id===e.from&&!p.role)&&graph.parts.some(p=>p.id===e.to&&!p.role)).map(e=>({...e,from:module.children[graph.parts.filter(p=>!p.role).findIndex(p=>p.id===e.from)]?.id,to:module.children[graph.parts.filter(p=>!p.role).findIndex(p=>p.id===e.to)]?.id})).filter(e=>e.from&&e.to)}
 else if(!part.child)module.operator=mathSpec(part,'');
 else module.expand=()=>mmhvaeModule(part,parent,observed,1);
 return module;
}
