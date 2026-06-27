import { useEffect, useState } from "react";

const PHASES = [
  { label: "Breathe in", dur: 4000, scale: 1, grad: "var(--grad-brand)" },
  { label: "Hold", dur: 4000, scale: 1, grad: "var(--grad-lavender)" },
  { label: "Breathe out", dur: 6000, scale: 0.55, grad: "var(--grad-soft)" },
];

export function BreathingGuide() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setI((v) => (v + 1) % PHASES.length), PHASES[i].dur);
    return () => clearTimeout(t);
  }, [i]);
  const p = PHASES[i];
  return (
    <div className="flex h-full flex-col items-center justify-center gap-12">
      <div className="relative grid place-items-center">
        <div className="absolute h-72 w-72 rounded-full opacity-30 blur-3xl" style={{ background: p.grad }} />
        <div
          className="grid h-56 w-56 place-items-center rounded-full text-white"
          style={{ background: p.grad, transform: `scale(${p.scale})`, transition: `transform ${p.dur}ms ease-in-out`, boxShadow: "var(--shadow-glow)" }}
        >
          <span className="font-display text-2xl font-semibold">{p.label}</span>
        </div>
      </div>
      <p className="max-w-xs text-center text-[14px] text-muted-foreground">
        Follow the circle. In through the nose, soft hold, slow release through the mouth.
      </p>
    </div>
  );
}
