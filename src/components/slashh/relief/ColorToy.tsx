import { useRef } from "react";

const COLORS = [
  "oklch(0.83 0.13 165)", "oklch(0.79 0.12 205)", "oklch(0.75 0.11 185)",
  "oklch(0.9 0.05 235)", "oklch(0.82 0.07 290)", "oklch(0.87 0.07 175)",
];

export function ColorToy() {
  const ref = useRef<HTMLDivElement>(null);
  const ci = useRef(0);

  const spawn = (clientX: number, clientY: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const dot = document.createElement("span");
    const size = 26 + Math.random() * 34;
    ci.current = (ci.current + 1) % COLORS.length;
    dot.style.cssText = `position:absolute;left:${x - size / 2}px;top:${y - size / 2}px;width:${size}px;height:${size}px;border-radius:9999px;pointer-events:none;background:${COLORS[ci.current]};filter:blur(2px);opacity:0.75;transition:transform 1.1s ease,opacity 1.1s ease`;
    el.appendChild(dot);
    requestAnimationFrame(() => {
      dot.style.transform = `scale(2.4) translateY(-30px)`;
      dot.style.opacity = "0";
    });
    setTimeout(() => dot.remove(), 1150);
  };

  return (
    <div className="flex h-full flex-col gap-3">
      <div
        ref={ref}
        onPointerMove={(e) => Math.random() > 0.4 && spawn(e.clientX, e.clientY)}
        onPointerDown={(e) => spawn(e.clientX, e.clientY)}
        className="relative flex-1 touch-none overflow-hidden rounded-[2rem] shadow-[var(--shadow-card)]"
        style={{ background: "var(--grad-soft)" }}
      >
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          <p className="font-display text-[17px] font-medium text-muted-foreground/70">Drag your finger to paint calm</p>
        </div>
      </div>
    </div>
  );
}
