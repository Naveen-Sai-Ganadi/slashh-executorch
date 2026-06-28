import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

export type Stage = "auth" | "onboarding" | "app";
export type AppView = "dashboard" | "settings";
export type StressBand = "idle" | "calm" | "listening" | "elevated" | "high";

export interface Settings {
  sensitivity: number; // 0-100
  demoMode: boolean;
  notifications: boolean;
}

export interface Calibration {
  done: boolean;
  signalType: string;   // "model" | "energy"
  calmAnchor: number;
  stressAnchor: number;
  calmCount: number;
  stressCount: number;
  step: number;
}

export interface BandMeta {
  label: string;
  sub: string;
  color: string; // css var token name
  ring: string;
}

export const BAND_META: Record<StressBand, BandMeta> = {
  idle: { label: "Ready", sub: "Tap begin when you're ready", color: "var(--brand-paleblue)", ring: "var(--brand-seafoam)" },
  calm: { label: "Calm and steady", sub: "Your voice sounds relaxed", color: "var(--brand-mint)", ring: "var(--brand-teal)" },
  listening: { label: "Listening gently", sub: "Staying with you", color: "var(--brand-teal)", ring: "var(--brand-cyan)" },
  elevated: { label: "Noticing tension", sub: "A slower breath may help", color: "var(--brand-seafoam)", ring: "var(--brand-paleblue)" },
  high: { label: "Let's slow down", sub: "We can ease this together", color: "var(--brand-lavender)", ring: "var(--brand-coral)" },
};

export function bandFor(listening: boolean, stress: number): StressBand {
  if (!listening) return "idle";
  if (stress < 28) return "calm";
  if (stress < 58) return "listening";
  if (stress < 80) return "elevated";
  return "high";
}

interface AndroidBridgeType {
  toggleListening: () => void;
  startListening: () => void;
  stopListening: () => void;
  requestMicPermission: () => void;
  requestNotificationPermission: () => void;
  setSensitivity: (sensitivity: number) => void;
  setDemoMode: (enabled: boolean) => void;
  startCalibration: () => void;
  cancelCalibration: () => void;
  saveCalibration: (useModel: boolean, calmAnchor: number, stressAnchor: number) => void;
  startRelief: (type: string) => void;
  stopRelief: () => void;
  dismissRelief: () => void;
  getBackendStatus: () => void;
  startRecording: (step: number) => void;
  devBypassAuth: () => boolean;
  runNpuProbe: () => void;
  hasAccount: () => boolean;
  getName: () => string;
  createAccount: (name: string, email: string, password: string) => void;
  checkLogin: (email: string, password: string) => boolean;
  isOnboarded: () => boolean;
  setEnabled: (csvKeys: string) => void;
  setOnboarded: (done: boolean) => void;
}

declare global {
  interface Window {
    AndroidBridge?: AndroidBridgeType;
    updateAndroidState?: (dataStr: string) => void;
  }
}

interface SlashhState {
  stage: Stage;
  view: AppView;
  hasPasscode: boolean;
  listening: boolean;
  stress: number;
  band: StressBand;
  settings: Settings;
  calibration: Calibration;
  reliefOpen: boolean;
  activeReliefType: string | null;
  calibrationOpen: boolean;
  vadActive: boolean;
  latency: number;
  lastOutput: string;
  backendType: string;
  fastRpcStatus: string;
  modelStatus: string;
  micPermissionState: string;
  notificationPermissionState: string;
  backendState: string;
  isOnline: boolean;
  // Live transcription (active session) + the per-signal breakdown feeding the meter.
  transcript: string;
  textStress: number | null;   // 0-100, text-model stress of the latest transcript
  audioScore: number | null;   // 0-100, WavLM(NPU)-or-energy audio leg
  fused: number | null;        // 0-100, audio+text fusion output (null = audio-only)
  whisperBackend: string;      // e.g. "Snapdragon NPU (Whisper)" or "—"
  setStage: (s: Stage) => void;
  setView: (v: AppView) => void;
  setHasPasscode: (b: boolean) => void;
  toggleListening: () => void;
  setListening: (b: boolean) => void;
  simulateStress: () => void;
  resetSession: () => void;
  updateSettings: (p: Partial<Settings>) => void;
  saveCalibration: (c: Partial<Calibration>) => void;
  startRecording: (step: number) => void;
  resetCalibration: () => void;
  setReliefOpen: (b: boolean) => void;
  startRelief: (type: string) => void;
  stopRelief: () => void;
  setCalibrationOpen: (b: boolean) => void;
  requestMicPermission: () => void;
  requestNotificationPermission: () => void;
  authHasAccount: () => boolean;
  signup: (name: string, email: string, password: string) => Promise<void>;
  login: (email: string, password: string) => Promise<boolean>;
  completeOnboarding: (enabledKeys?: string[]) => void;
  devBypass: () => void;
  runNpuProbe: () => void;
}

const LS_ACCT = "slashh_acct";
const LS_ONBOARDED = "slashh_onboarded";

/** SHA-256 hex of the raw string — matches the native Prefs hashing scheme
 *  (lowercase hex, no salt). Used only by the no-bridge web-preview fallback;
 *  in the WebView the native bridge owns hashing. */
async function sha256hex(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

const SlashhContext = createContext<SlashhState | null>(null);

const DEFAULT_CALIB: Calibration = {
  done: false,
  signalType: "energy",
  calmAnchor: 0.03,
  stressAnchor: 0.14,
  calmCount: 0,
  stressCount: 0,
  step: -1,
};

export function SlashhProvider({ children }: { children: ReactNode }) {
  const [stage, setStage] = useState<Stage>("auth");
  const [view, setView] = useState<AppView>("dashboard");
  const [hasPasscode, setHasPasscode] = useState(false);
  const [listening, setListeningState] = useState(false);
  const [stress, setStress] = useState(5);
  const [settings, setSettings] = useState<Settings>({
    sensitivity: 55,
    demoMode: false,
    notifications: true,
  });
  const [calibration, setCalibration] = useState<Calibration>(DEFAULT_CALIB);
  const [reliefOpen, setReliefOpen] = useState(false);
  const [activeReliefType, setActiveReliefType] = useState<string | null>(null);
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [latency, setLatency] = useState(12);
  const [lastOutput, setLastOutput] = useState("0.31");

  // Android specific integration states
  const [backendType, setBackendType] = useState("Demo Fallback (No Bridge)");
  const [fastRpcStatus, setFastRpcStatus] = useState("checking");
  const [modelStatus, setModelStatus] = useState("loaded");
  const [micPermissionState, setMicPermissionState] = useState("checking");
  const [notificationPermissionState, setNotificationPermissionState] = useState("checking");
  const [backendState, setBackendState] = useState("IDLE");
  const [isOnline, setIsOnline] = useState(() => typeof navigator !== "undefined" ? navigator.onLine : true);
  const [transcript, setTranscript] = useState("");
  const [textStress, setTextStress] = useState<number | null>(null);
  const [audioScore, setAudioScore] = useState<number | null>(null);
  const [fused, setFused] = useState<number | null>(null);
  const [whisperBackend, setWhisperBackend] = useState("—");

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // refs for the simulation loop (only used if running without bridge)
  const listeningRef = useRef(listening);
  const demoRef = useRef(settings.demoMode);
  const sensRef = useRef(settings.sensitivity);
  const stressRef = useRef(stress);
  const simBoost = useRef(0);
  const demoPhase = useRef(0);
  const elevatedTicks = useRef(0);
  const dismissedAt = useRef(0);

  useEffect(() => { listeningRef.current = listening; }, [listening]);
  useEffect(() => { demoRef.current = settings.demoMode; }, [settings.demoMode]);
  useEffect(() => { sensRef.current = settings.sensitivity; }, [settings.sensitivity]);
  useEffect(() => { stressRef.current = stress; }, [stress]);

  const band = bandFor(listening, stress);
  const vadActive = listening && stress > 12;

  // Bidirectional JS Bridge: register callback for Android updates
  useEffect(() => {
    window.updateAndroidState = (dataStr: string) => {
      try {
        const data = JSON.parse(dataStr);
        if (data.listening !== undefined) setListeningState(data.listening);
        if (data.stress !== undefined) setStress(data.stress);
        if (data.latency !== undefined) setLatency(data.latency);
        if (data.lastOutput !== undefined) setLastOutput(data.lastOutput);
        if (data.backendType !== undefined) setBackendType(data.backendType);
        if (data.fastRpcStatus !== undefined) setFastRpcStatus(data.fastRpcStatus);
        if (data.modelStatus !== undefined) setModelStatus(data.modelStatus);
        if (data.micPermissionState !== undefined) setMicPermissionState(data.micPermissionState);
        if (data.notificationPermissionState !== undefined) setNotificationPermissionState(data.notificationPermissionState);
        if (data.state !== undefined) setBackendState(data.state);
        if (data.isOnline !== undefined) setIsOnline(data.isOnline);
        if (data.transcript !== undefined) setTranscript(data.transcript);
        if (data.textStress !== undefined) setTextStress(data.textStress);
        if (data.audioScore !== undefined) setAudioScore(data.audioScore);
        if (data.fused !== undefined) setFused(data.fused);
        if (data.whisperBackend !== undefined) setWhisperBackend(data.whisperBackend);
        
        if (data.calibration !== undefined) {
          setCalibration((prev) => ({
            ...prev,
            done: data.calibration.done ?? prev.done,
            signalType: data.calibration.signalType ?? prev.signalType,
            calmAnchor: data.calibration.calmAnchor ?? prev.calmAnchor,
            stressAnchor: data.calibration.stressAnchor ?? prev.stressAnchor,
            calmCount: data.calibration.calmCount ?? prev.calmCount,
            stressCount: data.calibration.stressCount ?? prev.stressCount,
            step: data.calibration.step ?? prev.step,
          }));
        }

        if (data.reliefOpen !== undefined) {
          setReliefOpen(data.reliefOpen);
        }

        if (data.activeReliefType !== undefined) {
          setActiveReliefType(data.activeReliefType);
        }

        if (data.calibrationOpen !== undefined) {
          setCalibrationOpen(data.calibrationOpen);
        }
      } catch (e) {
        console.error("Failed to parse Android update JSON:", e);
      }
    };

    if (window.AndroidBridge) {
      try {
        window.AndroidBridge.getBackendStatus();
      } catch (e) {
        console.error("Error asking Android for initial status:", e);
      }
    }

    return () => {
      delete window.updateAndroidState;
    };
  }, []);

  // master simulation tick (only runs if there is no Android bridge)
  useEffect(() => {
    if (window.AndroidBridge) return;

    const id = setInterval(() => {
      const isListening = listeningRef.current;
      const demo = demoRef.current;
      const sens = sensRef.current;
      let target: number;

      if (!isListening) {
        target = 4;
      } else {
        let base = 22;
        if (demo) {
          demoPhase.current += 0.018;
          base = 46 + 40 * Math.sin(demoPhase.current);
        }
        const sensScale = 0.7 + (sens / 100) * 0.65;
        target = base * sensScale + simBoost.current;
      }
      simBoost.current *= 0.955;

      const prev = stressRef.current;
      const next = Math.max(0, Math.min(100, prev + (target - prev) * 0.08));
      stressRef.current = next;
      setStress(next);

      // technical mocks
      if (isListening && Math.random() > 0.6) {
        setLatency(9 + Math.round(Math.random() * 8));
        setLastOutput((0.2 + (next / 100) * 0.7).toFixed(2));
      }

      // auto relief simulation
      if (next > 78) elevatedTicks.current += 1;
      else if (next < 60) elevatedTicks.current = 0;
      if (
        elevatedTicks.current > 28 &&
        Date.now() - dismissedAt.current > 25000
      ) {
        setReliefOpen((open) => {
          if (!open) {
            elevatedTicks.current = 0;
            setActiveReliefType("breath");
          }
          return true;
        });
      }
    }, 120);
    return () => clearInterval(id);
  }, []);

  const setListening = (b: boolean) => {
    if (window.AndroidBridge) {
      if (b) window.AndroidBridge.startListening();
      else window.AndroidBridge.stopListening();
    } else {
      setListeningState(b);
    }
  };

  const toggleListening = () => {
    if (window.AndroidBridge) {
      window.AndroidBridge.toggleListening();
    } else {
      setListeningState((v) => !v);
    }
  };

  const requestMicPermission = () => {
    if (window.AndroidBridge) {
      window.AndroidBridge.requestMicPermission();
    } else {
      setMicPermissionState("granted");
    }
  };

  const requestNotificationPermission = () => {
    if (window.AndroidBridge) {
      window.AndroidBridge.requestNotificationPermission();
    } else {
      setNotificationPermissionState("granted");
    }
  };

  const simulateStress = () => {
    if (window.AndroidBridge) {
      // Native demo trigger: spikes stress and fires a (rotating) relief so the
      // Simulate button actually demos relief methods on-device.
      window.AndroidBridge.setDemoMode(true);
    } else {
      simBoost.current += 45;
      if (!listeningRef.current) setListening(true);
    }
  };

  const resetSession = () => {
    simBoost.current = 0;
    demoPhase.current = 0;
    elevatedTicks.current = 0;
    setListening(false);
    setStress(5);
    updateSettings({ demoMode: false });
  };

  const updateSettings = (p: Partial<Settings>) => {
    setSettings((s) => {
      const next = { ...s, ...p };
      if (window.AndroidBridge) {
        if (p.sensitivity !== undefined) window.AndroidBridge.setSensitivity(p.sensitivity);
        if (p.demoMode !== undefined) window.AndroidBridge.setDemoMode(p.demoMode);
      }
      return next;
    });
  };

  const saveCalibration = (c: Partial<Calibration>) => {
    setCalibration((cur) => {
      const next = { ...cur, ...c, done: true };
      if (window.AndroidBridge) {
        window.AndroidBridge.saveCalibration(
          next.signalType === "model",
          next.calmAnchor,
          next.stressAnchor
        );
      }
      return next;
    });
  };

  const startRecording = (step: number) => {
    if (window.AndroidBridge) {
      window.AndroidBridge.startRecording(step);
    } else {
      setCalibration((prev) => ({ ...prev, step }));
      let count = 0;
      const id = setInterval(() => {
        count += 1;
        setCalibration((prev) => {
          if (step === 0) {
            return { ...prev, calmCount: count };
          } else {
            return { ...prev, stressCount: count };
          }
        });
        if (count >= 6) {
          clearInterval(id);
          setCalibration((prev) => ({ ...prev, step: -1 }));
        }
      }, 300);
    }
  };

  const resetCalibration = () => {
    if (window.AndroidBridge) {
      window.AndroidBridge.saveCalibration(
        DEFAULT_CALIB.signalType === "model",
        DEFAULT_CALIB.calmAnchor,
        DEFAULT_CALIB.stressAnchor
      );
    }
    setCalibration(DEFAULT_CALIB);
  };

  const closeRelief = (b: boolean) => {
    if (!b) dismissedAt.current = Date.now();
    setReliefOpen(b);
    if (!b) {
      setActiveReliefType(null);
      if (window.AndroidBridge) {
        window.AndroidBridge.dismissRelief();
      }
    }
  };

  const startRelief = (type: string) => {
    setReliefOpen(true);
    setActiveReliefType(type);
    if (window.AndroidBridge) {
      window.AndroidBridge.startRelief(type);
    }
  };

  const stopRelief = () => {
    closeRelief(false);
  };

  const setCalibrationOpenWithBridge = (b: boolean) => {
    setCalibrationOpen(b);
    if (window.AndroidBridge) {
      if (b) {
        window.AndroidBridge.startCalibration();
      } else {
        window.AndroidBridge.cancelCalibration();
      }
    }
  };

  // ---- Local login/signup (on-device). In the WebView the native bridge owns
  // credential storage + hashing (Prefs); the localStorage path is only for the
  // no-bridge browser preview. No network in either path.
  const authHasAccount = (): boolean => {
    if (window.AndroidBridge?.hasAccount) {
      try { return !!window.AndroidBridge.hasAccount(); } catch { /* fall through */ }
    }
    return !!localStorage.getItem(LS_ACCT);
  };

  const isOnboardedNow = (): boolean => {
    if (window.AndroidBridge?.isOnboarded) {
      try { return !!window.AndroidBridge.isOnboarded(); } catch { /* fall through */ }
    }
    return localStorage.getItem(LS_ONBOARDED) === "1";
  };

  const signup = async (name: string, email: string, password: string): Promise<void> => {
    if (window.AndroidBridge?.createAccount) {
      window.AndroidBridge.createAccount(name, email, password);
    } else {
      const hash = await sha256hex(password);
      localStorage.setItem(
        LS_ACCT,
        JSON.stringify({ name: name.trim(), email: email.trim().toLowerCase(), hash })
      );
    }
    setStage(isOnboardedNow() ? "app" : "onboarding");
  };

  const login = async (email: string, password: string): Promise<boolean> => {
    let ok = false;
    if (window.AndroidBridge?.checkLogin) {
      ok = !!window.AndroidBridge.checkLogin(email, password);
    } else {
      const raw = localStorage.getItem(LS_ACCT);
      if (raw) {
        try {
          const a = JSON.parse(raw);
          ok = a.email === email.trim().toLowerCase() && a.hash === (await sha256hex(password));
        } catch { ok = false; }
      }
    }
    if (ok) setStage(isOnboardedNow() ? "app" : "onboarding");
    return ok;
  };

  const completeOnboarding = (enabledKeys?: string[]) => {
    if (window.AndroidBridge) {
      if (enabledKeys && enabledKeys.length) window.AndroidBridge.setEnabled?.(enabledKeys.join(","));
      window.AndroidBridge.setOnboarded?.(true);
    } else {
      localStorage.setItem(LS_ONBOARDED, "1");
    }
    setStage("app");
  };

  /** Dev bypass: skip login and onboarding in one tap. Uses native bridge when
   *  available (creates a dummy account + marks onboarded on-device); falls back
   *  to localStorage for the browser preview. */
  const devBypass = () => {
    if (window.AndroidBridge?.devBypassAuth) {
      try { window.AndroidBridge.devBypassAuth(); } catch { /* ignore */ }
    } else {
      localStorage.setItem(LS_ACCT, JSON.stringify({ name: "Dev User", email: "dev@slashh.ai", hash: "" }));
      localStorage.setItem(LS_ONBOARDED, "1");
    }
    setStage("app");
  };

  /** Fire the WavLM NPU smoke test. Results appear in logcat tag "WavLMProbe". */
  const runNpuProbe = () => {
    if (window.AndroidBridge?.runNpuProbe) {
      try { window.AndroidBridge.runNpuProbe(); } catch { /* ignore */ }
    } else {
      console.log("[NPU probe] AndroidBridge not available — no-op in browser");
    }
  };

  const value = useMemo<SlashhState>(
    () => ({
      stage, view, hasPasscode, listening, stress, band, settings, calibration,
      reliefOpen, activeReliefType, calibrationOpen, vadActive, latency, lastOutput,
      backendType, fastRpcStatus, modelStatus, micPermissionState, notificationPermissionState,
      backendState, isOnline,
      transcript, textStress, audioScore, fused, whisperBackend,
      setStage, setView, setHasPasscode, toggleListening, setListening,
      simulateStress, resetSession, updateSettings, saveCalibration,
      resetCalibration, setReliefOpen: closeRelief, startRelief, stopRelief,
      setCalibrationOpen: setCalibrationOpenWithBridge,
      requestMicPermission, requestNotificationPermission, startRecording,
      authHasAccount, signup, login, completeOnboarding, devBypass, runNpuProbe,
    }),
    [
      stage, view, hasPasscode, listening, stress, band, settings, calibration,
      reliefOpen, activeReliefType, calibrationOpen, vadActive, latency, lastOutput,
      backendType, fastRpcStatus, modelStatus, micPermissionState, notificationPermissionState,
      backendState, isOnline,
      transcript, textStress, audioScore, fused, whisperBackend,
    ]
  );

  return <SlashhContext.Provider value={value}>{children}</SlashhContext.Provider>;
}

export function useSlashh() {
  const ctx = useContext(SlashhContext);
  if (!ctx) throw new Error("useSlashh must be used within SlashhProvider");
  return ctx;
}
