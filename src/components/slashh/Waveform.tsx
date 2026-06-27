import { useEffect, useRef } from "react";
import { useSlashh } from "@/lib/slashh-store";

export function Waveform({ height = 72 }: { height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { listening, stress } = useSlashh();
  const listeningRef = useRef(listening);
  const stressRef = useRef(stress);
  listeningRef.current = listening;
  stressRef.current = stress;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let t = 0;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      const w = canvas.clientWidth;
      const h = height;
      ctx.clearRect(0, 0, w, h);
      const active = listeningRef.current;
      const s = stressRef.current / 100;
      const amp = active ? (4 + s * (h * 0.32)) : 1.5;
      const mid = h / 2;
      t += 0.05 + s * 0.05;

      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "oklch(0.83 0.13 165)");
      grad.addColorStop(0.5, "oklch(0.76 0.12 188)");
      grad.addColorStop(1, "oklch(0.82 0.07 290)");

      const lines = 3;
      for (let l = 0; l < lines; l++) {
        ctx.beginPath();
        const phase = t - l * 0.6;
        const la = amp * (1 - l * 0.28);
        for (let x = 0; x <= w; x += 2) {
          const k = x / w;
          const env = Math.sin(Math.PI * k);
          const y =
            mid +
            Math.sin(k * 10 + phase) * la * env +
            Math.sin(k * 23 + phase * 1.7) * la * 0.4 * env;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = grad;
        ctx.globalAlpha = 0.85 - l * 0.25;
        ctx.lineWidth = 2.5 - l * 0.6;
        ctx.lineCap = "round";
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [height]);

  return <canvas ref={ref} className="w-full" style={{ height }} />;
}
