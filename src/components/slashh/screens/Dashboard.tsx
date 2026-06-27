import { Settings as SettingsIcon, Play, Pause, Sparkles, RotateCcw, Wifi, Lock, HeartHandshake, SlidersHorizontal, Mic } from "lucide-react";
import { Logo } from "@/components/slashh/Logo";
import { StressMeter } from "@/components/slashh/StressMeter";
import { Waveform } from "@/components/slashh/Waveform";
import { BAND_META, useSlashh } from "@/lib/slashh-store";
import { Button } from "@/components/ui/button";

export function Dashboard() {
  const {
    listening, band, settings, toggleListening, simulateStress, resetSession,
    setView, setReliefOpen, setCalibrationOpen, micPermissionState, requestMicPermission,
    backendType,
  } = useSlashh();
  
  // Map backend type to user-friendly label
  const getBackendLabel = () => {
    if (backendType?.includes("NPU") || backendType?.includes("QNN")) {
      return "Snapdragon NPU";
    } else if (backendType?.includes("CPU") || backendType?.includes("XNNPACK")) {
      return "CPU fallback";
    } else if (backendType?.includes("Demo")) {
      return "Demo mode";
    } else if (backendType?.includes("Unavailable")) {
      return "Unavailable";
    }
    return backendType || "Local mode";
  };
  
  // Show permission denied state if mic permission is denied
  if (micPermissionState === "denied") {
    return (
      <div className="flex min-h-full flex-col px-5 pb-8 pt-5">
        <header className="flex items-center justify-between">
          <Logo size={24} />
          <button
            onClick={() => setView("settings")}
            aria-label="Settings"
            className="grid h-10 w-10 place-items-center rounded-full bg-card text-muted-foreground shadow-[var(--shadow-card)] transition hover:text-foreground active:scale-95"
          >
            <SettingsIcon className="h-5 w-5" />
          </button>
        </header>

        <div className="mt-12 flex flex-col items-center">
          <div className="grid h-24 w-24 place-items-center rounded-full bg-accent/50">
            <Mic className="h-12 w-12 text-accent-foreground" strokeWidth={1.5} />
          </div>
          <h2 className="mt-6 font-display text-2xl font-bold tracking-tight text-foreground">Microphone required</h2>
          <p className="mt-3 text-center text-[14px] text-muted-foreground">
            Slashh needs microphone access to listen to your voice and detect stress levels.
          </p>
          
          <Button onClick={requestMicPermission} className="mt-8 h-12 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)]">
            <Mic className="mr-2 h-5 w-5" /> Allow microphone access
          </Button>

          <p className="mt-8 text-center text-[12px] text-muted-foreground">
            Your voice is processed locally on this device and never stored or sent to a server.
          </p>
        </div>
      </div>
    );
  }
  
  const meta = BAND_META[band];

  return (
    <div className="flex min-h-full flex-col px-5 pb-8 pt-5">
      {/* top bar */}
      <header className="flex items-center justify-between">
        <Logo size={24} />
        <button
          onClick={() => setView("settings")}
          aria-label="Settings"
          className="grid h-10 w-10 place-items-center rounded-full bg-card text-muted-foreground shadow-[var(--shadow-card)] transition hover:text-foreground active:scale-95"
        >
          <SettingsIcon className="h-5 w-5" />
        </button>
      </header>

      {/* status chips */}
      <div className="mt-4 flex items-center justify-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-accent/60 px-3 py-1.5 text-[12px] font-medium text-accent-foreground">
          <Lock className="h-3 w-3" /> {getBackendLabel()}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-3 py-1.5 text-[12px] font-medium text-secondary-foreground">
          <Wifi className="h-3 w-3" /> Network: Offline
        </span>
        {settings.demoMode && (
          <span className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[12px] font-semibold text-white" style={{ background: "var(--grad-brand)" }}>
            Demo
          </span>
        )}
      </div>

      {/* meter */}
      <div className="mt-7 flex flex-col items-center">
        <StressMeter />
        <div key={band} className="animate-fade-up mt-6 text-center">
          <h2 className="font-display text-[22px] font-bold tracking-tight text-foreground/90">{meta.label}</h2>
          <p className="mt-1 text-[14px] text-muted-foreground">{meta.sub}</p>
        </div>
      </div>

      {/* waveform */}
      <div className="mt-6 rounded-3xl bg-card/70 p-4 shadow-[var(--shadow-card)] backdrop-blur">
        <div className="mb-1 flex items-center justify-between px-1">
          <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground">Voice activity</span>
          <span className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: listening ? "var(--brand-teal)" : "var(--muted-foreground)" }} />
            {listening ? "Live" : "Idle"}
          </span>
        </div>
        <Waveform />
      </div>

      {/* primary control */}
      <div className="mt-6 space-y-3">
        <Button
          onClick={toggleListening}
          disabled={micPermissionState !== "granted"}
          className="h-15 w-full rounded-2xl py-4 text-[16px] font-semibold shadow-[var(--shadow-soft)]"
        >
          {listening ? <><Pause className="mr-2 h-5 w-5" /> Pause</> : <><Play className="mr-2 h-5 w-5" /> Begin listening</>}
        </Button>

        <div className="grid grid-cols-2 gap-3">
          <Button variant="outline" onClick={simulateStress} className="h-12 rounded-2xl border-primary/20 bg-card text-[14px] font-medium text-foreground hover:bg-accent/50">
            <Sparkles className="mr-1.5 h-4 w-4" /> Simulate
          </Button>
          <Button variant="outline" onClick={resetSession} className="h-12 rounded-2xl border-primary/20 bg-card text-[14px] font-medium text-foreground hover:bg-accent/50">
            <RotateCcw className="mr-1.5 h-4 w-4" /> Reset
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Button onClick={() => setReliefOpen(true)} className="h-12 rounded-2xl text-[14px] font-semibold text-white shadow-[var(--shadow-card)]" style={{ background: "var(--grad-lavender)" }}>
            <HeartHandshake className="mr-1.5 h-4 w-4" /> Relief
          </Button>
          <Button onClick={() => setCalibrationOpen(true)} variant="ghost" className="h-12 rounded-2xl bg-secondary/70 text-[14px] font-medium text-secondary-foreground hover:bg-secondary">
            <SlidersHorizontal className="mr-1.5 h-4 w-4" /> Calibrate
          </Button>
        </div>
      </div>

      {/* privacy badge */}
      <p className="mt-5 text-center text-[12px] text-muted-foreground">
        🔒 Processed locally · voice stays on device
      </p>
    </div>
  );
}
