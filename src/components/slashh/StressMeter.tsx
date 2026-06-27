import { BAND_META, useSlashh } from "@/lib/slashh-store";

export function StressMeter() {
  const { stress, band, listening } = useSlashh();
  const meta = BAND_META[band];
  const size = 260;
  const stroke = 14;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.round(stress);
  const offset = c - (pct / 100) * c;

  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      {/* ambient glow */}
      <div
        className="absolute inset-2 rounded-full blur-2xl transition-colors duration-700"
        style={{ background: meta.color, opacity: 0.28 }}
      />
      {listening && (
        <div
          className="absolute inset-0 rounded-full animate-ring-pulse"
          style={{ boxShadow: `0 0 0 2px ${meta.ring}`, opacity: 0.5 }}
        />
      )}

      <svg width={size} height={size} className="relative -rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--muted)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={meta.color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.25s linear, stroke 0.7s ease" }}
        />
      </svg>

      {/* breathing orb + readout */}
      <div className="absolute inset-0 grid place-items-center">
        <div
          className={`grid h-[150px] w-[150px] place-items-center rounded-full ${listening ? "animate-breathe" : ""}`}
          style={{
            background: `radial-gradient(circle at 50% 35%, color-mix(in oklab, ${meta.color} 45%, white), color-mix(in oklab, ${meta.color} 12%, white))`,
            boxShadow: `inset 0 2px 12px color-mix(in oklab, ${meta.color} 35%, transparent)`,
            transition: "background 0.7s ease",
          }}
        >
          <div className="text-center">
            <div className="font-display text-[40px] font-bold leading-none text-foreground/85">
              {pct}
              <span className="text-base font-medium text-muted-foreground">%</span>
            </div>
            <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
              {listening ? "stress" : "resting"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
