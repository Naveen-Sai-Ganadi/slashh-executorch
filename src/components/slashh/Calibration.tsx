import { useEffect, useState } from "react";
import { X, Mic, Check, ArrowRight } from "lucide-react";
import { useSlashh } from "@/lib/slashh-store";
import { Button } from "@/components/ui/button";

const STEPS = [
  { title: "Baseline voice", prompt: "Read this slowly:", quote: "\u201cI am breathing calmly and slowly.\u201d", hint: "We listen for your relaxed tone." },
  { title: "Active voice", prompt: "Now, a little faster:", quote: "\u201cCount from one to ten.\u201d", hint: "This sets your upper energy range." },
  { title: "Save calibration", prompt: "Your personalised settings", quote: "", hint: "Tuned to your unique voice." },
];

function CalibrationGraph({ signalType, calmAnchor, stressAnchor }: { signalType: string; calmAnchor: number; stressAnchor: number }) {
  const w = 280, h = 120, pad = 14;
  const x = (i: number) => pad + (i / 100) * (w - pad * 2);
  const y = (v: number) => h - pad - (v / 100) * (h - pad * 2);
  const calmPct = Math.round(calmAnchor * 100);
  const stressPct = Math.round(stressAnchor * 100);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <defs>
        <linearGradient id="cg" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="oklch(0.83 0.13 165)" />
          <stop offset="1" stopColor="oklch(0.76 0.12 195)" />
        </linearGradient>
      </defs>
      <line x1={pad} y1={y(stressPct)} x2={w - pad} y2={y(stressPct)} stroke="oklch(0.82 0.07 290)" strokeWidth="1.5" strokeDasharray="4 4" />
      <line x1={pad} y1={y(calmPct)} x2={w - pad} y2={y(calmPct)} stroke="oklch(0.83 0.13 165)" strokeWidth="1.5" strokeDasharray="4 4" />
      <path
        d={`M ${x(0)} ${y(20)} C ${x(25)} ${y(28)}, ${x(40)} ${y(85)}, ${x(60)} ${y(90)} S ${x(85)} ${y(35)}, ${x(100)} ${y(25)}`}
        fill="none" stroke="url(#cg)" strokeWidth="3" strokeLinecap="round"
      />
      <circle cx={x(50)} cy={y(stressPct)} r="4" fill="oklch(0.82 0.07 290)" />
      <circle cx={x(72)} cy={y(calmPct)} r="4" fill="oklch(0.83 0.13 165)" />
      <text x={pad} y={y(stressPct) - 5} fontSize="9" fill="oklch(0.6 0.03 230)">stress {stressAnchor.toFixed(2)}</text>
      <text x={pad} y={y(calmPct) + 12} fontSize="9" fill="oklch(0.6 0.03 230)">calm {calmAnchor.toFixed(2)}</text>
    </svg>
  );
}

export function Calibration() {
  const { setCalibrationOpen, saveCalibration, calibration, startRecording } = useSlashh();
  const [step, setStep] = useState(0);

  const signalType = calibration.signalType;
  const calmAnchor = calibration.calmAnchor;
  const stressAnchor = calibration.stressAnchor;
  
  const calmCount = calibration.calmCount;
  const stressCount = calibration.stressCount;
  const activeStep = calibration.step;

  const recording = activeStep === step;
  const count = step === 0 ? calmCount : stressCount;
  const progress = Math.min(100, Math.round((count / 6) * 100));
  const done = count >= 6;

  const s = STEPS[step];
  
  // Guard: ensure we don't go out of bounds
  if (!s) {
    setCalibrationOpen(false);
    return null;
  }
  
  const next = () => {
    if (step < 2) {
      setStep(step + 1);
    }
  };

  const handleClose = () => {
    try {
      setCalibrationOpen(false);
    } catch (e) {
      console.error("Error closing calibration:", e);
    }
  };

  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-background/95 backdrop-blur-xl">
      <header className="flex items-center justify-between px-6 pt-6">
        <span className="font-display text-[15px] font-semibold text-muted-foreground">Calibrate to my voice</span>
        <button onClick={handleClose} className="grid h-9 w-9 place-items-center rounded-full bg-card text-muted-foreground shadow-[var(--shadow-card)]">
          <X className="h-4.5 w-4.5" />
        </button>
      </header>

      <div className="mt-6 flex gap-2 px-6">
        {STEPS.map((_, i) => (
          <div key={i} className="h-1.5 flex-1 rounded-full transition-colors" style={{ background: i <= step ? "var(--brand-teal)" : "var(--muted)" }} />
        ))}
      </div>

      <div key={step} className="animate-fade-up flex flex-1 flex-col items-center justify-center px-7 text-center">
        {step < 2 ? (
          <>
            <div className={`grid h-32 w-32 place-items-center rounded-full ${recording ? "animate-breathe" : ""}`} style={{ background: "var(--grad-soft)", boxShadow: "var(--shadow-glow)" }}>
              <Mic className="h-12 w-12 text-primary" strokeWidth={1.6} />
            </div>
            <h2 className="mt-8 font-display text-2xl font-bold tracking-tight">{s.title}</h2>
            <p className="mt-3 text-[14px] text-muted-foreground">{s.prompt}</p>
            <p className="mt-1 font-display text-[19px] font-medium text-foreground">{s.quote}</p>
            {(recording || count > 0) && (
              <div className="mt-6 flex flex-col items-center gap-2">
                <div className="h-2 w-48 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full transition-all duration-300" style={{ width: `${progress}%`, background: "var(--grad-brand)" }} />
                </div>
                <span className="text-[12px] font-medium text-muted-foreground">{count}/6 samples recorded</span>
              </div>
            )}
            {done && !recording && (
              <p className="animate-scale-in mt-6 flex items-center gap-2 text-[14px] font-medium text-primary">
                <Check className="h-4 w-4" /> Captured successfully
              </p>
            )}
            <p className="mt-4 text-[12.5px] text-muted-foreground">{s.hint}</p>
          </>
        ) : (
          <>
            <h2 className="font-display text-2xl font-bold tracking-tight">{s.title}</h2>
            <p className="mt-2 text-[14px] text-muted-foreground">{s.hint}</p>
            <div className="mt-6 w-full rounded-3xl bg-card p-5 shadow-[var(--shadow-card)]">
              <CalibrationGraph signalType={signalType} calmAnchor={calmAnchor} stressAnchor={stressAnchor} />
              <div className="mt-3 grid grid-cols-3 gap-3">
                <div className="rounded-2xl bg-accent/50 p-3">
                  <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Signal</p>
                  <p className="font-display text-base font-bold text-foreground">{signalType === "model" ? "🧠 Model" : "🔊 Energy"}</p>
                </div>
                <div className="rounded-2xl bg-secondary p-3">
                  <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Calm</p>
                  <p className="font-display text-lg font-bold text-foreground">{calmAnchor.toFixed(2)}</p>
                </div>
                <div className="rounded-2xl bg-accent/50 p-3">
                  <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Stress</p>
                  <p className="font-display text-lg font-bold text-foreground">{stressAnchor.toFixed(2)}</p>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="px-6 pb-10">
        {step < 2 ? (
          done ? (
            <Button className="h-14 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)]" onClick={next}>
              Continue <ArrowRight className="ml-1 h-5 w-5" />
            </Button>
          ) : (
            <Button className="h-14 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)]" disabled={recording} onClick={() => startRecording(step)}>
              {recording ? "Listening for speech…" : "Start recording"}
            </Button>
          )
        ) : (
          <Button
            className="h-14 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)]"
            onClick={() => { 
              saveCalibration({ signalType, calmAnchor, stressAnchor }); 
              handleClose(); 
            }}
          >
            Use these settings <Check className="ml-1 h-5 w-5" />
          </Button>
        )}
      </div>
    </div>
  );
}
