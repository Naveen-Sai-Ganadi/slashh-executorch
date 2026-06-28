import { useState } from "react";
import { ArrowRight, Lock } from "lucide-react";
import { Logo } from "@/components/slashh/Logo";
import { useSlashh } from "@/lib/slashh-store";
import { Button } from "@/components/ui/button";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * Local, on-device login / signup. Replicates the sk/npu-binaries AuthView
 * semantics 1:1 — first run (no stored account) defaults to SIGNUP, returning
 * runs default to LOGIN; credentials are stored only on this device (the native
 * bridge owns SHA-256 hashing). No network anywhere in this path.
 */
export function AuthScreen() {
  const { authHasAccount, signup, login, devBypass } = useSlashh();
  const [mode, setMode] = useState<"signup" | "login">(authHasAccount() ? "login" : "signup");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const isSignup = mode === "signup";

  const submit = async () => {
    setError(null);
    if (isSignup && !name.trim()) return setError("Please enter your name");
    if (!EMAIL_RE.test(email.trim())) return setError("Enter a valid email");
    if (password.length < 4) return setError("Password must be at least 4 characters");
    setBusy(true);
    try {
      if (isSignup) {
        await signup(name, email, password);
      } else {
        const ok = await login(email, password);
        if (!ok) {
          setError("Incorrect email or password");
          setBusy(false);
        }
      }
    } catch {
      setError("Something went wrong. Please try again.");
      setBusy(false);
    }
  };

  const inputCls =
    "h-14 w-full rounded-2xl border border-border bg-card px-4 text-[15px] text-foreground " +
    "placeholder:text-muted-foreground/70 outline-none transition focus:border-primary/50 " +
    "focus:ring-2 focus:ring-ring/30";

  return (
    <div className="flex min-h-full flex-col px-6 pb-10 pt-16">
      <div className="animate-fade-up flex flex-col items-center text-center">
        <div className="animate-float-soft">
          <Logo size={52} showWord={false} />
        </div>
        <h1 className="mt-6 font-display text-[28px] font-bold leading-tight tracking-tight">
          <span className="text-gradient">slashh</span>
        </h1>
        <p className="mt-2 text-[15px] font-medium text-muted-foreground">
          Private, on-device stress relief.
        </p>
      </div>

      <div className="animate-fade-up mt-8 flex flex-1 flex-col">
        <h2 className="font-display text-[22px] font-bold tracking-tight">
          {isSignup ? "Create your account" : "Welcome back"}
        </h2>
        <p className="mt-1 text-[13.5px] text-muted-foreground">
          {isSignup
            ? "Set up a local profile to personalize your relief."
            : "Log in to your on-device profile."}
        </p>

        <div className="mt-5 space-y-3">
          {isSignup && (
            <input
              className={inputCls}
              placeholder="Your name"
              value={name}
              autoCapitalize="words"
              onChange={(e) => setName(e.target.value)}
            />
          )}
          <input
            className={inputCls}
            placeholder="Email"
            type="email"
            inputMode="email"
            autoCapitalize="none"
            autoCorrect="off"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            className={inputCls}
            placeholder="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
          />
        </div>

        {error && (
          <p className="mt-3 text-[13px] font-medium" style={{ color: "#E57373" }}>
            {error}
          </p>
        )}

        <Button
          className="mt-5 h-14 w-full rounded-2xl text-[15px] font-semibold shadow-[var(--shadow-soft)]"
          disabled={busy}
          onClick={submit}
        >
          {isSignup ? "Create account" : "Log in"}
          <ArrowRight className="ml-1 h-5 w-5" />
        </Button>

        <button
          type="button"
          className="mt-4 text-center text-[13.5px] text-muted-foreground"
          onClick={() => {
            setMode(isSignup ? "login" : "signup");
            setError(null);
          }}
        >
          {isSignup ? (
            <>
              Already have an account?{" "}
              <span className="font-semibold text-primary">Log in</span>
            </>
          ) : (
            <>
              New here? <span className="font-semibold text-primary">Create an account</span>
            </>
          )}
        </button>
      </div>

      <div className="mt-6 flex items-center justify-center gap-2 text-[12px] text-muted-foreground">
        <Lock className="h-3.5 w-3.5" />
        Stored only on this device
      </div>

      <button
        type="button"
        className="mt-3 text-center text-[11px] text-muted-foreground/50 underline underline-offset-2"
        onClick={devBypass}
      >
        Dev skip →
      </button>
    </div>
  );
}
