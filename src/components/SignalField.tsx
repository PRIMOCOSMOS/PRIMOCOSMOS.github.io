import { useEffect, useRef } from 'react'

const SIGNALS = [
  { color: '#d8ff45', offset: 0.25, speed: 0.44 },
  { color: '#78b8ff', offset: 0.5, speed: 0.32 },
  { color: '#ff5c35', offset: 0.74, speed: 0.26 },
]

export function SignalField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const pointer = { x: 0.52, y: 0.48 }
    let frame = 0
    let visible = true
    let width = 1
    let height = 1

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      width = Math.max(1, rect.width)
      height = Math.max(1, rect.height)
      canvas.width = Math.floor(width * dpr)
      canvas.height = Math.floor(height * dpr)
      context.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    const draw = (stamp: number) => {
      context.clearRect(0, 0, width, height)
      const t = reduceMotion ? 7.5 : stamp / 1000

      context.save()
      context.strokeStyle = 'rgba(216,255,69,0.09)'
      context.lineWidth = 1
      for (let x = 32; x < width; x += 64) {
        context.beginPath()
        context.moveTo(x, height * 0.08)
        context.lineTo(x, height * 0.92)
        context.stroke()
      }
      for (let y = height * 0.16; y < height; y += 72) {
        context.beginPath()
        context.moveTo(width * 0.04, y)
        context.lineTo(width * 0.96, y)
        context.stroke()
      }
      context.restore()

      SIGNALS.forEach((signal, index) => {
        context.save()
        context.strokeStyle = signal.color
        context.lineWidth = index === 1 ? 1.8 : 1.15
        context.globalAlpha = index === 2 ? 0.7 : 0.9
        context.beginPath()
        for (let x = -10; x <= width + 10; x += 8) {
          const pull = Math.exp(-Math.pow((x / width - pointer.x) * 5.2, 2))
          const y =
            height * signal.offset +
            Math.sin(x * 0.014 + t * signal.speed * 4 + index) * (10 + index * 4) +
            Math.sin(x * 0.0035 - t * 0.35) * 13 +
            (pointer.y - 0.5) * height * 0.11 * pull
          if (x < 0) context.moveTo(x, y)
          else context.lineTo(x, y)
        }
        context.stroke()

        const pulseX = ((t * 58 * signal.speed + index * width * 0.21) % (width * 0.88)) + width * 0.06
        const pulseY =
          height * signal.offset +
          Math.sin(pulseX * 0.014 + t * signal.speed * 4 + index) * (10 + index * 4) +
          Math.sin(pulseX * 0.0035 - t * 0.35) * 13
        context.fillStyle = signal.color
        context.globalAlpha = 1
        context.beginPath()
        context.arc(pulseX, pulseY, index === 0 ? 4.5 : 3.5, 0, Math.PI * 2)
        context.fill()
        context.restore()
      })

      const coreX = width * (0.67 + (pointer.x - 0.5) * 0.025)
      const coreY = height * (0.49 + (pointer.y - 0.5) * 0.035)
      context.save()
      context.translate(coreX, coreY)
      for (let ring = 0; ring < 3; ring += 1) {
        const radius = 48 + ring * 22 + Math.sin(t * 0.5 + ring) * 3
        context.strokeStyle = ring === 1 ? 'rgba(120,184,255,.62)' : 'rgba(216,255,69,.28)'
        context.lineWidth = ring === 1 ? 1.5 : 1
        context.setLineDash(ring === 2 ? [3, 9] : [])
        context.beginPath()
        context.arc(0, 0, radius, t * 0.1 + ring, Math.PI * 1.64 + t * 0.1 + ring)
        context.stroke()
      }
      context.setLineDash([])
      context.fillStyle = '#07111f'
      context.strokeStyle = '#d8ff45'
      context.lineWidth = 1.4
      context.beginPath()
      for (let i = 0; i < 6; i += 1) {
        const angle = Math.PI / 3 * i - Math.PI / 6
        const x = Math.cos(angle) * 27
        const y = Math.sin(angle) * 27
        if (i === 0) context.moveTo(x, y)
        else context.lineTo(x, y)
      }
      context.closePath()
      context.fill()
      context.stroke()
      context.fillStyle = '#d8ff45'
      context.fillRect(-5, -5, 10, 10)
      context.restore()

      if (!reduceMotion && visible) frame = window.requestAnimationFrame(draw)
    }

    const onPointerMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect()
      pointer.x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
      pointer.y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))
    }

    const resizeObserver = new ResizeObserver(resize)
    const visibilityObserver = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting
      if (visible && !reduceMotion && frame === 0) frame = window.requestAnimationFrame(draw)
      if (!visible && frame) {
        window.cancelAnimationFrame(frame)
        frame = 0
      }
    })
    resizeObserver.observe(canvas)
    visibilityObserver.observe(canvas)
    canvas.addEventListener('pointermove', onPointerMove)
    resize()
    draw(0)

    return () => {
      resizeObserver.disconnect()
      visibilityObserver.disconnect()
      canvas.removeEventListener('pointermove', onPointerMove)
      if (frame) window.cancelAnimationFrame(frame)
    }
  }, [])

  return <canvas ref={canvasRef} className="signal-field" aria-hidden="true" />
}
