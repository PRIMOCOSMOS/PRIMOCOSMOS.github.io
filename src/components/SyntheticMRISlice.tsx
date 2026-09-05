import { useEffect, useRef } from 'react'

interface SyntheticMRISliceProps {
  aliasing?: number
  contrast?: number
  label: string
}

export function SyntheticMRISlice({ aliasing = 0, contrast = 0.7, label }: SyntheticMRISliceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const size = 180
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.scale(dpr, dpr)
    ctx.fillStyle = '#030608'
    ctx.fillRect(0, 0, size, size)

    const cx = size / 2
    const cy = size / 2 + 3
    ctx.save()
    ctx.translate(cx, cy)
    ctx.fillStyle = `rgba(226,232,222,${0.18 + contrast * 0.22})`
    ctx.beginPath()
    ctx.ellipse(0, 0, 60, 72, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = `rgba(82,104,112,${0.42 + contrast * 0.25})`
    ctx.beginPath()
    ctx.ellipse(-20, -2, 28, 52, -0.18, 0, Math.PI * 2)
    ctx.ellipse(20, -2, 28, 52, 0.18, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = `rgba(216,255,69,${0.18 + contrast * 0.24})`
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(-8, -43)
    ctx.bezierCurveTo(-27, -15, -22, 18, -7, 43)
    ctx.moveTo(8, -43)
    ctx.bezierCurveTo(27, -15, 22, 18, 7, 43)
    ctx.stroke()
    ctx.fillStyle = '#030608'
    ctx.beginPath()
    ctx.ellipse(0, 2, 10, 22, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.restore()

    if (aliasing > 0.01) {
      ctx.save()
      ctx.globalCompositeOperation = 'screen'
      for (let i = -5; i <= 5; i += 1) {
        const y = cy + i * 12
        const shift = Math.sin(i * 2.1) * 24 * aliasing
        ctx.fillStyle = `rgba(120,184,255,${0.018 + aliasing * 0.055})`
        ctx.fillRect(18 + shift, y, 144, 4 + aliasing * 5)
      }
      ctx.restore()
    }
  }, [aliasing, contrast])

  return (
    <figure className="synthetic-slice">
      <canvas ref={canvasRef} role="img" aria-label={`${label}，代码生成的 MRI 示意图`} />
      <figcaption>{label}</figcaption>
    </figure>
  )
}
