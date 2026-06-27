import { X, ChevronLeft, Wind, Gamepad2, Music, Palette, Smile, type LucideIcon } from "lucide-react";
import { useSlashh } from "@/lib/slashh-store";
import { BreathingGuide } from "./BreathingGuide";
import { ZenTicTacToe } from "./ZenTicTacToe";
import { Soundscapes } from "./Soundscapes";
import { ColorToy } from "./ColorToy";
import { Affirmations } from "./Affirmations";

type WidgetId = "breath" | "game" | "sound" | "color" | "affirm";

const WIDGETS: { id: WidgetId; label: string; desc: string; icon: LucideIcon; grad: string }[] = [
  { id: "breath", label: "Breathing guide", desc: "Slow, guided breaths", icon: Wind, grad: "var(--grad-brand)" },
  { id: "sound", label: "Soundscapes", desc: "Noise & soft rain", icon: Music, grad: "var(--grad-soft)" },
  { id: "color", label: "Color toy", desc: "Paint calming trails", icon: Palette, grad: "var(--grad-lavender)" },
  { id: "game", label: "Zen tic-tac-toe", desc: "A gentle little game", icon: Gamepad2, grad: "var(--grad-coral)" },
  { id: "affirm", label: "Affirmations", desc: "Kind words & light jokes", icon: Smile, grad: "var(--grad-brand)" },
];

const TITLES: Record<WidgetId, string> = {
  breath: "Breathing guide", game: "Zen tic-tac-toe", sound: "Soundscapes", color: "Color toy", affirm: "Affirmations & humour",
};

export function ReliefCenter() {
  const { setReliefOpen, stress, activeReliefType, startRelief } = useSlashh();

  const active = (() => {
    if (!activeReliefType || activeReliefType === "") return null;
    if (activeReliefType === "breathing") return "breath";
    if (activeReliefType === "tictactoe") return "game";
    if (activeReliefType === "sounds") return "sound";
    if (activeReliefType === "jokes") return "affirm";
    if (activeReliefType === "color") return "color";
    return activeReliefType as WidgetId;
  })();

  const setActive = (id: WidgetId | null) => {
    if (id === null) {
      startRelief(""); // Empty string means relief menu is open
    } else {
      const keyMap: Record<WidgetId, string> = {
        breath: "breathing",
        game: "tictactoe",
        sound: "sounds",
        affirm: "jokes",
        color: "color",
      };
      startRelief(keyMap[id]);
    }
  };

  return (
    <div className="animate-fade-up absolute inset-0 z-50 flex flex-col bg-background/96 backdrop-blur-xl">
      <header className="flex items-center justify-between px-6 pt-6">
        {active ? (
          <button onClick={() => setActive(null)} className="flex items-center gap-1 text-[14px] font-medium text-muted-foreground">
            <ChevronLeft className="h-5 w-5" /> Relief
          </button>
        ) : (
          <div>
            <h2 className="font-display text-xl font-bold tracking-tight">Relief center</h2>
            <p className="text-[12.5px] text-muted-foreground">Calming activities, whenever you need them</p>
          </div>
        )}
        <button onClick={() => setReliefOpen(false)} className="grid h-9 w-9 place-items-center rounded-full bg-card text-muted-foreground shadow-[var(--shadow-card)]">
          <X className="h-4.5 w-4.5" />
        </button>
      </header>

      {active ? (
        <div className="flex flex-1 flex-col px-6 pb-8 pt-3">
          <h3 className="mb-2 font-display text-lg font-semibold">{TITLES[active]}</h3>
          <div className="flex-1">
            {active === "breath" && <BreathingGuide />}
            {active === "game" && <ZenTicTacToe />}
            {active === "sound" && <Soundscapes />}
            {active === "color" && <ColorToy />}
            {active === "affirm" && <Affirmations />}
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-6 pb-8 pt-5">
          {stress > 70 && (
            <div className="mb-4 rounded-2xl px-4 py-3 text-[13px] font-medium text-white" style={{ background: "var(--grad-lavender)" }}>
              Things felt a little tense — let's slow down together.
            </div>
          )}
          <div className="grid gap-3.5">
            {WIDGETS.map((w) => {
              const Icon = w.icon;
              return (
                <button
                  key={w.id}
                  onClick={() => setActive(w.id)}
                  className="flex items-center gap-4 rounded-3xl bg-card p-4 text-left shadow-[var(--shadow-card)] transition active:scale-[0.98]"
                >
                  <div className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl" style={{ background: w.grad }}>
                    <Icon className="h-6 w-6 text-white" strokeWidth={1.7} />
                  </div>
                  <div>
                    <p className="text-[16px] font-semibold text-foreground">{w.label}</p>
                    <p className="text-[13px] text-muted-foreground">{w.desc}</p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
