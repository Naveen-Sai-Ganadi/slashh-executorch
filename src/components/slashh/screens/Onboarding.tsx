import { useState } from "react";
import { ArrowRight, Check, Mic, ShieldCheck, SlidersHorizontal, HeartHandshake } from "lucide-react";
import { Logo } from "@/components/slashh/Logo";
import { useSlashh } from "@/lib/slashh-store";
import { Button } from "@/components/ui/button";

const SLIDES = [
  { icon: Mic, title: "Listen gently", body: "Slashh quietly senses tension in your voice — pace, pitch and energy — to notice when stress is rising.", grad: "var(--grad-brand)" },
  { icon: ShieldCheck, title: "Private by design", body: "No cloud. Everything is processed on your device and your voice never leaves it.", grad: "var(--grad-soft)" },
  { icon: SlidersHorizontal, title: "Calibrate to your voice", body: "A short, optional calibration learns your personal baseline so detection feels tuned just for you.", grad: "var(--grad-lavender)" },
  { icon: HeartHandshake, title: "Relief when you need it", body: "Breathing, grounding, soundscapes and gentle play — calming activities ready the moment you want them.", grad: "var(--grad-coral)" },
];

/**
 * Welcome carousel shown once after login/signup (stage "onboarding"). On finish
 * it persists the onboarded flag (via the native bridge) so returning users go
 * straight to the dashboard. Relief preferences default to all enabled natively.
 */
export function Onboarding() {
  const { completeOnboarding } = useSlashh();
  const [slide, setSlide] = useState(0);

  const s = SLIDES[slide];
  const Icon = s.icon;
  const last = slide === SLIDES.length - 1;

  return (
    <div className="flex min-h-full flex-col px-6 pb-10 pt-14">
      <div className="flex justify-center">
        <Logo size={26} />
      </div>
      <div key={slide} className="animate-fade-up flex flex-1 flex-col items-center justify-center text-center">
        <div
          className="grid h-40 w-40 place-items-center rounded-[2.5rem] shadow-[var(--shadow-soft)] animate-float-soft"
          style={{ background: s.grad }}
        >
          <Icon className="h-16 w-16 text-white/90" strokeWidth={1.6} />
        </div>
        <h2 className="mt-9 font-display text-[26px] font-bold tracking-tight">{s.title}</h2>
        <p className="mt-3 max-w-xs text-[15px] leading-relaxed text-muted-foreground">{s.body}</p>
      </div>

      <div className="mb-6 flex justify-center gap-2">
        {SLIDES.map((_, i) => (
          <button
            key={i}
            onClick={() => setSlide(i)}
            className="h-2 rounded-full transition-all"
            style={{
              width: i === slide ? 26 : 8,
              background: i === slide ? "var(--brand-teal)" : "var(--muted)",
            }}
          />
        ))}
      </div>

      <div className="flex items-center gap-3">
        {!last && (
          <Button variant="ghost" className="h-14 rounded-2xl px-5 text-muted-foreground" onClick={() => completeOnboarding()}>
            Skip
          </Button>
        )}
        <Button
          className="h-14 flex-1 rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)]"
          onClick={() => (last ? completeOnboarding() : setSlide((v) => v + 1))}
        >
          {last ? (
            <>Enter Slashh <Check className="ml-1 h-5 w-5" /></>
          ) : (
            <>Continue <ArrowRight className="ml-1 h-5 w-5" /></>
          )}
        </Button>
      </div>
    </div>
  );
}
