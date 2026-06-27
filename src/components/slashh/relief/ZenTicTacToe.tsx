import { useEffect, useState } from "react";
import { RotateCcw } from "lucide-react";

type Cell = "X" | "O" | null;
const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
const MSGS = ["Nicely played.", "No rush, just play.", "Lovely move.", "Breathe and continue.", "You've got this."];

function winner(b: Cell[]): Cell {
  for (const [a, c, d] of LINES) if (b[a] && b[a] === b[c] && b[a] === b[d]) return b[a];
  return null;
}

export function ZenTicTacToe() {
  const [b, setB] = useState<Cell[]>(Array(9).fill(null));
  const [turn, setTurn] = useState<"X" | "O">("X");
  const [msg, setMsg] = useState("Your move — take your time.");
  const win = winner(b);
  const full = b.every(Boolean);

  const reset = () => { setB(Array(9).fill(null)); setTurn("X"); setMsg("Your move — take your time."); };

  const play = (i: number) => {
    if (b[i] || win || turn !== "X") return;
    const nb = [...b]; nb[i] = "X"; setB(nb); setTurn("O");
    setMsg(MSGS[Math.floor(Math.random() * MSGS.length)]);
  };

  useEffect(() => {
    if (turn !== "O" || win || b.every(Boolean)) return;
    const t = setTimeout(() => {
      const empty = b.map((c, i) => (c ? null : i)).filter((x) => x !== null) as number[];
      // friendly AI: mostly random, never tries hard
      const pick = empty[Math.floor(Math.random() * empty.length)];
      const nb = [...b]; nb[pick] = "O"; setB(nb); setTurn("X");
    }, 650);
    return () => clearTimeout(t);
  }, [turn, b, win]);

  useEffect(() => {
    if (win === "X") setMsg("You won — gentle victory! 🌿");
    else if (win === "O") setMsg("I won this one, but you're still winning at calm.");
    else if (full) setMsg("A peaceful draw.");
  }, [win, full]);

  return (
    <div className="flex h-full flex-col items-center justify-center gap-7">
      <p className="min-h-6 text-center font-display text-[17px] font-medium text-foreground">{msg}</p>
      <div className="grid grid-cols-3 gap-3">
        {b.map((c, i) => (
          <button
            key={i}
            onClick={() => play(i)}
            className="grid h-[88px] w-[88px] place-items-center rounded-3xl bg-card font-display text-4xl font-bold shadow-[var(--shadow-card)] transition active:scale-95"
            style={{ color: c === "X" ? "oklch(0.72 0.11 188)" : "oklch(0.82 0.07 290)" }}
          >
            <span className={c ? "animate-scale-in" : ""}>{c}</span>
          </button>
        ))}
      </div>
      <button onClick={reset} className="flex items-center gap-2 rounded-full bg-secondary px-5 py-2.5 text-[14px] font-medium text-secondary-foreground">
        <RotateCcw className="h-4 w-4" /> New game
      </button>
    </div>
  );
}
