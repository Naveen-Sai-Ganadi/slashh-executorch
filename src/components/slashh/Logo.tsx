import logo from "@/assets/slashh-logo.png";
import { cn } from "@/lib/utils";

export function Logo({
  className,
  showWord = true,
  size = 30,
}: {
  className?: string;
  showWord?: boolean;
  size?: number;
}) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <img
        src={logo}
        alt="Slashh AI logo"
        width={size * 1.5}
        height={size}
        style={{ height: size, width: "auto" }}
        className="select-none"
        draggable={false}
      />
      {showWord && (
        <span className="font-display text-[15px] font-semibold tracking-tight text-foreground/80">
          slashh <span className="text-gradient font-bold">ai</span>
        </span>
      )}
    </div>
  );
}
