import * as T from 'three'
import type {Glyph} from './anatomy'

/** A contiguous coordinate window, never a downsampled proxy. Unknown symbolic
 * extents stay symbolic. The axis/range record accompanies each rendered cell. */
export function sourceWindow(shape:string,glyph:Glyph,page=0){
 const expanded=shape.replace(/(\d+|[DHW])³/g,'$1 × $1 × $1').replace(/(\d+|[HW])²/g,'$1 × $1');
 let tail=expanded.split(/→|⇒/).at(-1)!.trim(),axes:string[]=[];
 const match=tail.match(/(?:\([^)]{1,20}\)|[A-Za-z]\w*|\d+)(?:\s*[×x]\s*(?:\([^)]{1,20}\)|[A-Za-z]\w*|\d+)){1,4}/);
 if(match)axes=match[0].split(/\s*[×x]\s*/);
 if(shape.includes('·')){const [channels,spatial]=expanded.split('·'),last=spatial.split(/→|⇒/).at(-1)!,dims=last.match(/\d+/g),c=channels.split(/→|⇒/).at(-1)!.match(/\d+/g);if(dims?.length&&c?.length)axes=[String(c.length>1?c.reduce((a,b)=>a+Number(b),0):Number(c[0])),...dims]}
 if(!axes.length){const pair=tail.match(/^\((\d+),\s*(\d+)\)$/);if(pair)axes=['2',pair[1]];else if(/^\d+(?:\s*ch|-vector)?$/.test(tail))axes=[tail.match(/\d+/)![0]]}
 const batchAxis=axes[0],leadingBatch=/^[BN]$/.test(batchAxis)||axes.length===5;if(leadingBatch)axes=axes.slice(1);
 const volume=axes.length>=4||/³|\bD\s*×\s*H\s*×\s*W\b/.test(shape.split(/→|⇒/).at(-1)!);
 if(!axes.length&&/scalar|标量/i.test(shape))axes=['1'];
 const dimensions=axes.map(a=>/^\d+$/.test(a)?Math.max(1,Number(a)):null),counts=dimensions.map((d,i)=>d===null?1:Math.min(d,axes.length===1?32:volume?i<axes.length-3?2:4:i<axes.length-2?3:6));
 const blocks=dimensions.map((d,i)=>d===null?1:Math.ceil(d/counts[i]));let remaining=Math.max(0,Math.floor(page));const starts=counts.map(()=>0);for(let i=counts.length-1;i>=0;i--){starts[i]=(remaining%blocks[i])*counts[i];remaining=Math.floor(remaining/blocks[i])}const visible=counts.map((c,i)=>Math.min(c,(dimensions[i]??c)-starts[i]));
 const positions:[number,number,number][]=[],coordinates:number[][]=[];
 if(axes.length&&dimensions.some(d=>d!==null)){const total=visible.reduce((a,b)=>a*b,1);for(let index=0;index<total;index++){let n=index;const c=visible.map(()=>0);for(let i=c.length-1;i>=0;i--){c[i]=n%visible[i];n=Math.floor(n/visible[i])}coordinates.push(c.map((v,i)=>v+starts[i]));const w=c.at(-1)!,h=c.at(-2)??0,channel=c.length>=3?c[c.length-3]:0,outer=c.length>=4?c[0]:0;positions.push(axes.length===1?[(w%8-3.5)*.55,0,(Math.floor(w/8)-1.5)*.55]:[(w-(counts.at(-1)!-1)/2)*.55+(volume?outer*3:0),channel*.7,(h-((counts.at(-2)??1)-1)/2)*.55])}}
 const names=axes.length===1?['F']:axes.length===2?['行','列']:axes.length===3?['C','H','W']:['C','D','H','W'];const ranges=axes.map((a,i)=>`${names[i]??'轴'+i} ${dimensions[i]===null?a:`${starts[i]}…${starts[i]+visible[i]-1} / ${a}`}`).join(' · ');
 return {axes:leadingBatch?[batchAxis,...axes]:axes,dimensions,counts,starts,pages:blocks.reduce((a,b)=>a*b,1),positions,coordinates,volume,label:positions.length?`${leadingBatch?'单 batch 切片 · ':''}坐标窗口 ${ranges}`:'符号尺寸 · 待指定输入',partial:dimensions.some((d,i)=>d===null||d>counts[i]),glyph};
}
export function sourceTensor(glyph:Glyph,shape:string,color:string,page=0){
 const window=sourceWindow(shape,glyph,page),group=new T.Group();Object.assign(group.userData,{tensorAxes:window.axes,volume:window.volume,extentOnly:!window.positions.length,tensorWindow:window});
 // Static, value-neutral cells: position is a real coordinate, color is the
 // source branch. Numerical value coloring is supplied by the active operator.
 if(window.positions.length){
  const cube=new T.BoxGeometry(.38,.38,.38).toNonIndexed(),edges=new T.EdgesGeometry(new T.BoxGeometry(.38,.38,.38)),solid:number[]=[],wire:number[]=[];
  for(const pos of window.positions){for(const [geometry,target] of [[cube,solid],[edges,wire]] as const){const a=geometry.getAttribute('position');for(let i=0;i<a.count;i++)target.push(a.getX(i)+pos[0],a.getY(i)+pos[1],a.getZ(i)+pos[2])}}
  const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(solid,3));geometry.computeVertexNormals();const line=new T.BufferGeometry();line.setAttribute('position',new T.Float32BufferAttribute(wire,3));
  group.add(new T.Mesh(geometry,new T.MeshPhysicalMaterial({color,transparent:true,opacity:.2,roughness:.13,metalness:.08,clearcoat:1,depthWrite:false})),new T.LineSegments(line,new T.LineBasicMaterial({color,transparent:true,opacity:.62,depthWrite:false})));cube.dispose();edges.dispose();
 }else{const frame=new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(3,.06,2)),new T.LineBasicMaterial({color,transparent:true,opacity:.35}));group.add(frame)}
 return group;
}
