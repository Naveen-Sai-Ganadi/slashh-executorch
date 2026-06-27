import { useState } from "react";
import { ChevronLeft, ChevronDown, Cpu, Activity, Mic, Gauge, Bell, FlaskConical, RotateCcw, ShieldCheck, Terminal } from "lucide-react";
import { useSlashh } from "@/lib/slashh-store";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`rounded-3xl bg-card p-5 shadow-[var(--shadow-card)] ${className}`}>{children}</div>;
}

export function SettingsScreen() {
  const {
    settings, updateSettings, resetCalibration, setView, calibration,
    listening, vadActive, latency, lastOutput,
    backendType, fastRpcStatus, modelStatus, micPermissionState,
  } = useSlashh();
  const [techOpen, setTechOpen] = useState(false);

  return (
    <div className="flex min-h-full flex-col px-5 pb-10 pt-5">
      <header className="flex items-center gap-3">
        <button onClick={() => setView("dashboard")} className="grid h-10 w-10 place-items-center rounded-full bg-card text-muted-foreground shadow-[var(--shadow-card)] transition active:scale-95">
          <ChevronLeft className="h-5 w-5" />
        </button>
        <h1 className="font-display text-xl font-bold tracking-tight">Settings</h1>
      </header>

      <div className="mt-5 space-y-4">
        {/* sensitivity */}
        <Card>
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-accent/70"><Gauge className="h-5 w-5 text-primary" /></div>
            <div className="flex-1">
              <p className="text-[15px] font-semibold text-foreground">Stress sensitivity</p>
              <p className="text-[12.5px] text-muted-foreground">How readily Slashh notices tension</p>
            </div>
            <span className="text-[14px] font-semibold text-primary">{settings.sensitivity}%</span>
          </div>
          <Slider
            value={[settings.sensitivity]}
            min={10} max={100} step={1}
            onValueChange={(v) => updateSettings({ sensitivity: v[0] })}
            className="mt-5"
          />
          <div className="mt-2 flex justify-between text-[11px] text-muted-foreground">
            <span>Gentle</span><span>Responsive</span>
          </div>
        </Card>

        {/* toggles */}
        <Card className="space-y-1">
          <div className="flex items-center gap-3 py-1.5">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-accent/70"><FlaskConical className="h-5 w-5 text-primary" /></div>
            <div className="flex-1">
              <p className="text-[15px] font-semibold text-foreground">Demo mode</p>
              <p className="text-[12.5px] text-muted-foreground">Simulate stress rising and falling</p>
            </div>
            <Switch checked={settings.demoMode} onCheckedChange={(v) => updateSettings({ demoMode: v })} />
          </div>
          <div className="h-px bg-border" />
          <div className="flex items-center gap-3 py-1.5">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-accent/70"><Bell className="h-5 w-5 text-primary" /></div>
            <div className="flex-1">
              <p className="text-[15px] font-semibold text-foreground">Local notifications</p>
              <p className="text-[12.5px] text-muted-foreground">Gentle nudges, on this device only</p>
            </div>
            <Switch checked={settings.notifications} onCheckedChange={(v) => updateSettings({ notifications: v })} />
          </div>
        </Card>

        {/* calibration */}
        <Card>
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-accent/70"><RotateCcw className="h-5 w-5 text-primary" /></div>
            <div className="flex-1">
              <p className="text-[15px] font-semibold text-foreground">Calibration</p>
              <p className="text-[12.5px] text-muted-foreground">
                {calibration.done ? `${calibration.signalType === "model" ? "Model" : "Intensity"} · calm ${calibration.calmAnchor.toFixed(2)} · stressed ${calibration.stressAnchor.toFixed(2)}` : "Using default thresholds"}
              </p>
            </div>
            <Button variant="ghost" className="h-9 rounded-xl bg-secondary/70 px-3 text-[13px] font-medium" onClick={resetCalibration}>Reset</Button>
          </div>
        </Card>

        {/* privacy */}
        <Card>
          <div className="flex items-start gap-3">
            <div className="grid h-10 w-10 shrink-0 place-items-center rounded-2xl bg-accent/70"><ShieldCheck className="h-5 w-5 text-primary" /></div>
            <div>
              <p className="text-[15px] font-semibold text-foreground">Privacy</p>
              <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">
                Slashh runs entirely on-device. Your voice is analysed locally in real time and is never recorded, stored, or sent to any server.
              </p>
            </div>
          </div>
        </Card>

        {/* technical panel */}
        <div className="overflow-hidden rounded-3xl bg-card shadow-[var(--shadow-card)]">
          <button onClick={() => setTechOpen((v) => !v)} className="flex w-full items-center gap-3 p-5">
            <div className="grid h-10 w-10 place-items-center rounded-2xl bg-secondary"><Terminal className="h-5 w-5 text-muted-foreground" /></div>
            <div className="flex-1 text-left">
              <p className="text-[15px] font-semibold text-foreground">Technical status</p>
              <p className="text-[12.5px] text-muted-foreground">For judges &amp; developers</p>
            </div>
            <ChevronDown className={`h-5 w-5 text-muted-foreground transition-transform ${techOpen ? "rotate-180" : ""}`} />
          </button>
          {techOpen && (
            <div className="animate-fade-up space-y-2.5 border-t border-border px-5 pb-5 pt-4 font-mono text-[12.5px]">
              <Row icon={Cpu} k="Inference backend" v={backendType} ok={backendType.includes("NPU") || backendType.includes("QNN")} />
              <Row icon={Activity} k="FastRPC / CDSP" v={fastRpcStatus} ok={fastRpcStatus === "ok"} />
              <Row icon={Gauge} k="Execution latency" v={`${latency} ms average`} ok />
              <Row icon={Mic} k="VAD status" v={vadActive ? "Voice detected" : "No voice"} ok={vadActive} />
              <Row icon={Mic} k="Microphone" v={micPermissionState === "granted" ? (listening ? "Active" : "Paused") : "Permission needed"} ok={listening && micPermissionState === "granted"} />
              <Row icon={Terminal} k="Last raw output" v={lastOutput} />
              <Row icon={FlaskConical} k="Model status" v={modelStatus} ok={modelStatus === "loaded"} />
              <p className="pt-1 text-[11px] leading-relaxed text-muted-foreground/80">
                Falls back to CPU (XNNPACK) when the NPU is unavailable. Values shown are a simulated diagnostic preview.
              </p>
            </div>
          )}
        </div>

        <p className="px-2 pt-1 text-center text-[12px] leading-relaxed text-muted-foreground">
          Slashh supports relaxation and reflection. It is not a medical device or diagnosis tool.
        </p>
      </div>
    </div>
  );
}

function Row({ icon: Icon, k, v, ok }: { icon: React.ElementType; k: string; v: string; ok?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl bg-secondary/50 px-3 py-2.5">
      <span className="flex items-center gap-2 text-muted-foreground"><Icon className="h-3.5 w-3.5" /> {k}</span>
      <span className="flex items-center gap-1.5 text-right text-foreground/80">
        {v}
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: ok ? "var(--brand-mint)" : "var(--muted-foreground)" }} />
      </span>
    </div>
  );
}
