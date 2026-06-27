import { useState } from "react";
import { CloudRain, Waves, Wind, Radio, Volume2 } from "lucide-react";
import { Slider } from "@/components/ui/slider";

const SOUNDS = [
  { id: "white", label: "White noise", icon: Radio },
  { id: "brown", label: "Brown noise", icon: Waves },
  { id: "pink", label: "Pink noise", icon: Wind },
  { id: "rain", label: "Soft rain", icon: CloudRain },
];

export function Soundscapes() {
  const [active, setActive] = useState<Record<string, boolean>>({});
  const [vol, setVol] = useState<Record<string, number>>({ white: 50, brown: 50, pink: 50, rain: 60 });
  return (
    <div className="flex h-full flex-col justify-center gap-3.5 px-1">
      {SOUNDS.map((s) => {
        const Icon = s.icon;
        const on = active[s.id];
        return (
          <div key={s.id} className="rounded-3xl bg-card p-4 shadow-[var(--shadow-card)]">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setActive((a) => ({ ...a, [s.id]: !a[s.id] }))}
                className="grid h-11 w-11 place-items-center rounded-2xl transition"
                style={{ background: on ? "var(--grad-brand)" : "var(--secondary)" }}
              >
                <Icon className={`h-5 w-5 ${on ? "text-white" : "text-muted-foreground"}`} />
              </button>
              <div className="flex-1">
                <p className="text-[15px] font-semibold text-foreground">{s.label}</p>
                <p className="text-[12px] text-muted-foreground">{on ? "Playing" : "Tap to play"}</p>
              </div>
              <button
                onClick={() => setActive((a) => ({ ...a, [s.id]: !a[s.id] }))}
                className="relative h-6 w-11 rounded-full transition-colors"
                style={{ background: on ? "var(--brand-teal)" : "var(--muted)" }}
              >
                <span className="absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all" style={{ left: on ? 22 : 2 }} />
              </button>
            </div>
            {on && (
              <div className="animate-fade-up mt-3 flex items-center gap-3 px-1">
                <Volume2 className="h-4 w-4 text-muted-foreground" />
                <Slider value={[vol[s.id]]} max={100} step={1} onValueChange={(v) => setVol((p) => ({ ...p, [s.id]: v[0] }))} className="flex-1" />
                <span className="w-8 text-right text-[12px] text-muted-foreground">{vol[s.id]}</span>
              </div>
            )}
          </div>
        );
      })}
      <p className="mt-1 text-center text-[12px] text-muted-foreground">Mock playback · layer sounds to find your calm mix.</p>
    </div>
  );
}
