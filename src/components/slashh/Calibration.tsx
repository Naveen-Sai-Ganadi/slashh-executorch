import { useEffect, useState, useRef } from "react";
import { X, Mic, Check, ArrowRight, ChevronLeft, Shield } from "lucide-react";
import { useSlashh } from "@/lib/slashh-store";
import { Button } from "@/components/ui/button";

const NORMAL_PROMPTS = [
  "Good morning. I finished the update and everything looks stable.",
  "I reviewed the notes and I think the plan is clear for today.",
  "The dashboard is working, and the results are easy to understand.",
  "I will send a short summary after I finish testing the flow.",
  "There are no major blockers right now. Everything is moving smoothly.",
  "I feel calm, focused, and ready to continue with the next task."
];

const ACTIVE_PROMPTS = [
  "I am running late and I need to finish this before the meeting starts!",
  "The deadline is coming fast and I still have too many things to fix!",
  "I just got another urgent message and I do not know what to handle first!",
  "The build failed again and I need this working right now!",
  "I have five minutes left and everything is moving too fast!",
  "I need to calm down, but I feel rushed and overwhelmed right now!"
];

export function Calibration() {
  const { setCalibrationOpen, saveCalibration, calibration, startRecording } = useSlashh();
  
  const [uiPhase, setUiPhase] = useState<"intro" | "flow" | "complete">("intro");
  const [currentSampleIndex, setCurrentSampleIndex] = useState(0);
  const [recordingState, setRecordingState] = useState<"idle" | "recording" | "saved">("idle");
  const [initialCountForSample, setInitialCountForSample] = useState(0);
  const [simulateTimer, setSimulateTimer] = useState<NodeJS.Timeout | null>(null);

  // Local state fallbacks for browser simulator
  const [localCalmCount, setLocalCalmCount] = useState(0);
  const [localStressCount, setLocalStressCount] = useState(0);

  const calmCount = window.AndroidBridge ? calibration.calmCount : localCalmCount;
  const stressCount = window.AndroidBridge ? calibration.stressCount : localStressCount;

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (simulateTimer) clearTimeout(simulateTimer);
    };
  }, [simulateTimer]);

  // Monitor counts to auto-advance to "saved" state
  useEffect(() => {
    const isNormal = currentSampleIndex < 6;
    if (isNormal) {
      if (recordingState === "recording" && calmCount > initialCountForSample) {
        setRecordingState("saved");
        if (window.AndroidBridge) {
          startRecording(-1); // Pause recording
        }
      }
    } else {
      if (recordingState === "recording" && stressCount > initialCountForSample) {
        setRecordingState("saved");
        if (window.AndroidBridge) {
          startRecording(-1); // Pause recording
        }
      }
    }
  }, [calmCount, stressCount, currentSampleIndex, recordingState, initialCountForSample, startRecording]);

  const handleRecord = () => {
    const isNormal = currentSampleIndex < 6;
    if (window.AndroidBridge) {
      if (isNormal) {
        setInitialCountForSample(calmCount);
        startRecording(0);
      } else {
        setInitialCountForSample(stressCount);
        startRecording(1);
      }
      setRecordingState("recording");
    } else {
      // Local preview simulation
      setRecordingState("recording");
      const timer = setTimeout(() => {
        if (isNormal) {
          setLocalCalmCount((prev) => prev + 1);
        } else {
          setLocalStressCount((prev) => prev + 1);
        }
        setRecordingState("saved");
      }, 2500);
      setSimulateTimer(timer);
    }
  };

  const handleNext = () => {
    if (currentSampleIndex < 11) {
      setCurrentSampleIndex(currentSampleIndex + 1);
      setRecordingState("idle");
    } else {
      setUiPhase("complete");
    }
  };

  const handleCancel = () => {
    if (simulateTimer) clearTimeout(simulateTimer);
    if (window.AndroidBridge) {
      startRecording(-1);
    }
    setCalibrationOpen(false);
  };

  const handleStartCalibration = () => {
    setLocalCalmCount(0);
    setLocalStressCount(0);
    setUiPhase("flow");
    setCurrentSampleIndex(0);
    setRecordingState("idle");
  };

  const handleFinish = () => {
    if (!window.AndroidBridge) {
      // For browser simulation preview, mock save settings
      saveCalibration({
        signalType: "model",
        calmAnchor: 0.05,
        stressAnchor: 0.18,
      });
    }
    setCalibrationOpen(false);
  };

  const isNormalPhase = currentSampleIndex < 6;

  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-background/95 backdrop-blur-xl">
      {/* Header */}
      <header className="flex items-center justify-between px-6 pt-6">
        {uiPhase === "flow" ? (
          <button
            onClick={handleCancel}
            className="flex items-center gap-1.5 text-[14px] font-semibold text-muted-foreground hover:text-foreground transition"
          >
            <ChevronLeft className="h-5 w-5" /> Cancel
          </button>
        ) : (
          <span className="font-display text-[15px] font-semibold text-muted-foreground">
            Calibrate to my voice
          </span>
        )}
        <button
          onClick={handleCancel}
          className="grid h-9 w-9 place-items-center rounded-full bg-card text-muted-foreground shadow-[var(--shadow-card)] hover:text-foreground transition"
        >
          <X className="h-4.5 w-4.5" />
        </button>
      </header>

      {/* Intro Phase */}
      {uiPhase === "intro" && (
        <div className="animate-fade-up flex flex-1 flex-col items-center justify-between px-7 py-10 text-center">
          <div className="my-auto max-w-sm">
            <div className="grid h-16 w-16 place-items-center rounded-2xl bg-cyan-500/10 text-cyan-600 dark:bg-cyan-500/20 dark:text-cyan-400 mx-auto">
              <Shield className="h-8 w-8" />
            </div>
            <h2 className="mt-6 font-display text-2xl font-bold tracking-tight text-foreground">
              Tuned to your voice.
            </h2>
            <p className="mt-3 text-[14px] leading-relaxed text-muted-foreground">
              Calibration captures 6 calm samples and 6 high-pressure samples. Slashh learns your personal baseline and sets a stress threshold that fits you — not an average.
            </p>

            <div className="mt-8 space-y-4">
              <div className="flex items-center gap-4 rounded-3xl border border-primary/5 bg-card/50 p-5 text-left shadow-[var(--shadow-soft)]">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-cyan-500/10 text-cyan-600 dark:bg-cyan-500/20 dark:text-cyan-400">
                  <Mic className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-display text-sm font-bold text-foreground">Normal Phase</h3>
                  <p className="text-[12px] text-muted-foreground">6 samples · calm speech</p>
                </div>
              </div>

              <div className="flex items-center gap-4 rounded-3xl border border-primary/5 bg-card/50 p-5 text-left shadow-[var(--shadow-soft)]">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-rose-500/10 text-rose-600 dark:bg-rose-500/20 dark:text-rose-400">
                  <Mic className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-display text-sm font-bold text-foreground">Active Phase</h3>
                  <p className="text-[12px] text-muted-foreground">6 samples · hurried / high-pressure speech</p>
                </div>
              </div>
            </div>
          </div>

          <Button
            className="h-14 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)] mt-6"
            onClick={handleStartCalibration}
          >
            Start Calibration
          </Button>
        </div>
      )}

      {/* Flow Phase */}
      {uiPhase === "flow" && (
        <div className="flex flex-1 flex-col justify-between pb-10">
          {/* Progress Indicators */}
          <div>
            <div className="mt-6 flex gap-1 px-6">
              {Array.from({ length: 12 }).map((_, i) => {
                const isActive = i === currentSampleIndex;
                const isCompleted = i < currentSampleIndex;
                let bgColor = "var(--muted)";
                if (isActive || isCompleted) {
                  bgColor = i < 6 ? "var(--brand-teal)" : "var(--brand-coral)";
                }
                return (
                  <div
                    key={i}
                    className="h-1.5 flex-1 rounded-full transition-all duration-300"
                    style={{
                      background: bgColor,
                      opacity: isActive ? 1 : isCompleted ? 0.7 : 0.2,
                    }}
                  />
                );
              })}
            </div>

            <div className="mt-4 flex items-center justify-between px-6 text-[12px] font-semibold uppercase tracking-wider text-muted-foreground">
              <span>Sample {currentSampleIndex + 1} / 12</span>
              <span
                className={`rounded-full px-2 py-0.5 text-[11px] ${
                  isNormalPhase
                    ? "bg-cyan-500/10 text-cyan-600 dark:bg-cyan-500/20 dark:text-cyan-400"
                    : "bg-rose-500/10 text-rose-600 dark:bg-rose-500/20 dark:text-rose-400"
                }`}
              >
                {isNormalPhase
                  ? `Normal Phase (${currentSampleIndex + 1}/6)`
                  : `Active Phase (${currentSampleIndex - 5}/6)`}
              </span>
            </div>
          </div>

          {/* Interactive Circle & Prompts */}
          <div className="flex flex-1 flex-col items-center justify-center px-7 text-center">
            <div
              className={`grid h-28 w-28 place-items-center rounded-full transition-all duration-500 ${
                recordingState === "recording" ? "animate-breathe" : ""
              } ${
                isNormalPhase
                  ? "bg-cyan-500/5 border border-cyan-500/10 text-cyan-600"
                  : "bg-rose-500/5 border border-rose-500/10 text-rose-600"
              }`}
              style={{
                boxShadow: recordingState === "recording" ? "var(--shadow-glow)" : "none",
              }}
            >
              {recordingState === "saved" ? (
                <Check className="h-10 w-10" />
              ) : (
                <Mic className="h-10 w-10" strokeWidth={1.6} />
              )}
            </div>

            <div className="mt-8 w-full rounded-3xl bg-card/60 border border-primary/5 p-6 shadow-[var(--shadow-card)] backdrop-blur">
              <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground block mb-2">
                Read Aloud
              </span>
              <p className="font-display text-[18px] font-semibold leading-relaxed text-foreground">
                “{isNormalPhase ? NORMAL_PROMPTS[currentSampleIndex] : ACTIVE_PROMPTS[currentSampleIndex - 6]}”
              </p>
            </div>

            <p className="mt-4 text-[13px] font-medium text-muted-foreground max-w-xs">
              {isNormalPhase
                ? "Read slowly in your natural calm voice."
                : "Read like you are in a hurry. Slightly faster, higher energy, but still clear."}
            </p>

            <div className="h-10 mt-6 flex items-center justify-center">
              {recordingState === "recording" && (
                <span className="text-[13px] font-medium text-muted-foreground animate-pulse">
                  Listening for speech…
                </span>
              )}
              {recordingState === "saved" && (
                <span className="flex items-center gap-1.5 text-[13px] font-semibold text-emerald-500">
                  <Check className="h-4 w-4" /> Sample saved.
                </span>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="px-6">
            {recordingState === "saved" ? (
              <Button
                className="h-14 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)] animate-fade-in"
                onClick={handleNext}
              >
                {currentSampleIndex === 11 ? "Complete Calibration" : "Next Sample"} <ArrowRight className="ml-1.5 h-4 w-4" />
              </Button>
            ) : (
              <Button
                className={`h-14 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)] transition-all ${
                  recordingState === "recording" ? "bg-muted text-muted-foreground hover:bg-muted" : ""
                }`}
                disabled={recordingState === "recording"}
                onClick={handleRecord}
              >
                {recordingState === "recording" ? "Recording…" : "Tap to Record"}
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Complete Phase */}
      {uiPhase === "complete" && (
        <div className="animate-fade-up flex flex-1 flex-col items-center justify-between px-7 py-10 text-center">
          <div className="my-auto max-w-sm">
            <div className="grid h-16 w-16 place-items-center rounded-2xl bg-emerald-500/10 text-emerald-600 dark:bg-emerald-500/20 dark:text-emerald-400 mx-auto">
              <Check className="h-8 w-8" />
            </div>
            <h2 className="mt-6 font-display text-2xl font-bold tracking-tight text-foreground">
              Calibration complete.
            </h2>
            <p className="mt-3 text-[14px] leading-relaxed text-muted-foreground">
              Slashh has learned your calm baseline and high-pressure voice pattern. Your personal stress threshold is now active.
            </p>

            <div className="mt-8 grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-primary/5 bg-card/60 p-4 shadow-[var(--shadow-soft)]">
                <span className="text-[18px] block mb-1">🧘</span>
                <p className="text-[12px] font-bold text-foreground">Calm Baseline</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">Captured successfully</p>
              </div>

              <div className="rounded-2xl border border-primary/5 bg-card/60 p-4 shadow-[var(--shadow-soft)]">
                <span className="text-[18px] block mb-1">⚡</span>
                <p className="text-[12px] font-bold text-foreground">Active Voice</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">Captured successfully</p>
              </div>

              <div className="rounded-2xl border border-primary/5 bg-card/60 p-4 shadow-[var(--shadow-soft)]">
                <span className="text-[18px] block mb-1">🎯</span>
                <p className="text-[12px] font-bold text-foreground">Stress Threshold</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">Personalized</p>
              </div>

              <div className="rounded-2xl border border-primary/5 bg-card/60 p-4 shadow-[var(--shadow-soft)]">
                <span className="text-[18px] block mb-1">🔒</span>
                <p className="text-[12px] font-bold text-foreground">Stored On-Device</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">Private and secure</p>
              </div>
            </div>
          </div>

          <Button
            className="h-14 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)] mt-6"
            onClick={handleFinish}
          >
            Return to Dashboard
          </Button>
        </div>
      )}
    </div>
  );
}
