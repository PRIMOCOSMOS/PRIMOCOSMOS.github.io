import type {Run} from '../workstation/engine'
export interface SourceValues {format:'primocosmos.tensor-values.v1';model:string;tensors:{id:string;coordinates:number[][];values:(number|null)[]}[]}
/** Imported numbers refer to explicit source addresses, never a downsampled
 * stand-in. No missing cell is silently filled with zero or a random value. */
export function parseSourceValues(text:string,run:Run,model:string):SourceValues{
 const data=JSON.parse(text) as SourceValues;
 if(data?.format!=='primocosmos.tensor-values.v1'||data.model!==model||!Array.isArray(data.tensors))throw Error('文件格式或文献不匹配，请使用当前文献的数值模板。');
 const known=new Map(run.tensors.map(t=>[t.id,t])),seen=new Set<string>();let count=0;
 for(const item of data.tensors){const t=known.get(item.id);if(!t||seen.has(item.id))throw Error(`未知或重复的张量：${item.id}`);seen.add(item.id);
  if(!Array.isArray(item.coordinates)||!Array.isArray(item.values)||item.coordinates.length!==item.values.length)throw Error(`${item.id} 的坐标与数值数量不一致。`);
  const addresses=new Set<string>();for(let i=0;i<item.values.length;i++){const c=item.coordinates[i],v=item.values[i],dims=t.window!.dimensions;
   if(!Array.isArray(c)||c.length!==dims.length||c.some((n,j)=>!Number.isInteger(n)||n<0||(dims[j]!==null&&n>=dims[j]!)))throw Error(`${item.id} 存在越界坐标。`);
   if(addresses.has(c.join(',')))throw Error(`${item.id} 存在重复坐标。`);addresses.add(c.join(','));
   if(v!==null&&(typeof v!=='number'||!Number.isFinite(v)))throw Error(`${item.id} 只接受有限数值或 null。`);
   if(t.constant&&v!==null&&v!==t.values[0])throw Error(`${item.id} 是源码常量，不能改写。`);
   if(v!==null)count++;
  }
 }
 if(!count)throw Error('文件中还没有数值。请把模板中的 null 替换为对应坐标的实际数值。');
 return data;
}
export function applySourceValues(run:Run,data?:SourceValues){
 const tensors=new Map(run.tensors.map(t=>[t.id,t]));
 for(const item of data?.tensors??[]){const t=tensors.get(item.id);if(!t)continue;const values=new Map(item.coordinates.map((c,i)=>[c.join(','),item.values[i]]));t.values=t.window!.coordinates.map((c,i)=>values.get(c.join(','))??t.values[i]);}
 return run;
}
export function sourceValuesTemplate(run:Run,model:string):SourceValues{
 return {format:'primocosmos.tensor-values.v1',model,tensors:run.tensors.map(t=>({id:t.id,coordinates:t.window!.coordinates,values:t.values.map(v=>Number.isFinite(v)?v:null)}))};
}
