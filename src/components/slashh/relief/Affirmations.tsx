import { useState } from "react";
import { RefreshCw } from "lucide-react";

const CARDS = [
  { t: "You're doing better than you think.", k: "affirmation" },
  { t: "My calendar is now 100% blocked for breathing. Very productive.", k: "humour" },
  { t: "This feeling is a wave. It rises, and it always passes.", k: "affirmation" },
  { t: "Plants don't stress about growing. Be a little more plant.", k: "humour" },
  { t: "You are allowed to take up space and to slow down.", k: "affirmation" },
  { t: "Inbox zero is a myth. Inner peace is the real flex.", k: "humour" },
  { t: "One steady breath is enough to begin again.", k: "affirmation" },
];

export function Affirmations() {
  const [idx, setIdx] = useState(0);
  const [flip, setFlip] = useState(false);
  const card = CARDS[idx];
  const next = () => { setFlip(true); setTimeout(() => { setIdx((v) => (v + 1) % CARDS.length); setFlip(false); }, 180); };
  return (
    <div className="flex h-full flex-col items-center justify-center gap-8 px-2">
      <button
        onClick={next}
        className="grid min-h-[260px] w-full max-w-sm place-items-center rounded-[2rem] p-8 text-center shadow-[var(--shadow-soft)] transition-all duration-200 active:scale-[0.98]"
        style={{ background: card.k === "humour" ? "var(--grad-coral)" : "var(--grad-brand)", opacity: flip ? 0 : 1, transform: flip ? "scale(0.96)" : "scale(1)" }}
      >
        <div>
          <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-white/70">{card.k}</span>
          <p className="mt-4 font-display text-[22px] font-semibold leading-snug text-white">{card.t}</p>
        </div>
      </button>
      <button onClick={next} className="flex items-center gap-2 rounded-full bg-card px-5 py-2.5 text-[14px] font-medium text-foreground shadow-[var(--shadow-card)]">
        <RefreshCw className="h-4 w-4" /> Next card
      </button>
    </div>
  );
}
