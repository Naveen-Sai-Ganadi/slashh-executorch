import { SlashhProvider, useSlashh } from "@/lib/slashh-store";
import { AuthScreen } from "@/components/slashh/screens/AuthScreen";
import { Onboarding } from "@/components/slashh/screens/Onboarding";
import { Dashboard } from "@/components/slashh/screens/Dashboard";
import { SettingsScreen } from "@/components/slashh/screens/Settings";
import { Calibration } from "@/components/slashh/Calibration";
import { ReliefCenter } from "@/components/slashh/relief/ReliefCenter";

function Screens() {
  const { stage, view, reliefOpen, calibrationOpen } = useSlashh();
  return (
    <>
      {stage === "auth" ? (
        <AuthScreen />
      ) : stage === "onboarding" ? (
        <Onboarding />
      ) : view === "settings" ? (
        <SettingsScreen />
      ) : (
        <Dashboard />
      )}
      {calibrationOpen && <Calibration />}
      {reliefOpen && <ReliefCenter />}
    </>
  );
}

export function SlashhApp() {
  return (
    <SlashhProvider>
      <div
        className="min-h-screen w-full"
        style={{
          background:
            "radial-gradient(1100px 560px at 80% -6%, oklch(0.93 0.05 195 / 0.7), transparent 60%), radial-gradient(820px 460px at 0% 4%, oklch(0.92 0.05 290 / 0.45), transparent 55%), var(--background)",
        }}
      >
        <div className="mx-auto flex min-h-screen max-w-[440px] items-stretch md:items-center md:py-6">
          <div className="relative flex h-screen w-full flex-col overflow-hidden bg-background shadow-[var(--shadow-soft)] md:h-[880px] md:rounded-[2.75rem] md:border md:border-border">
            <div className="relative flex-1 overflow-y-auto">
              <Screens />
            </div>
          </div>
        </div>
      </div>
    </SlashhProvider>
  );
}
