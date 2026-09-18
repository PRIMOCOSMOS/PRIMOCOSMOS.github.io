import {numericPalette as palette} from '../mmhvae/numericPaletteTokens'
import {EditableText} from '../EditableContent'

/** The legend and WebGL use the same palette tokens. Gradient encodes data, not decoration. */
export default function ColorLegend(){
 const ramp=`linear-gradient(90deg,${palette.negative},${palette.negativeMid},${palette.zero},${palette.positiveMid},${palette.positive})`
 return <details className="ws-color-legend">
  <summary><span className="ws-color-ramp" style={{background:ramp}} aria-hidden="true"/><span>数值与连线色标</span></summary>
  <div className="ws-color-key">
   <div className="ws-color-scale"><span>− 强负值</span><span>0</span><span>强正值 ＋</span></div>
   <div className="ws-color-ramp ws-color-ramp-full" style={{background:ramp}} aria-hidden="true"/>
   <EditableText as="p" textKey="ws-color-scale-v2">玫瑰铜 → 琥珀表示负值，冰蓝 → 青绿表示正值，中性灰蓝表示零。色阶按当前张量或系数张量的最大绝对值缩放，并用平滑的非线性色阶保留小值差异；不同张量的颜色不代表相同绝对数值。</EditableText>
   <div className="ws-color-roles"><span><i style={{background:palette.relation}}/>普通数学依赖</span><span><i style={{background:palette.focus}}/>当前元素轮廓</span></div>
   <EditableText as="p" textKey="ws-color-flow-v2">权重线沿用数值色阶；普通依赖线用淡紫色，不暗示正负。流动高亮保留原色，表示当前计算联系。点击水晶查看准确数值与坐标。</EditableText>
  </div>
 </details>
}
