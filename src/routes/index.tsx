import { createFileRoute } from "@tanstack/react-router";
import { SlashhApp } from "@/components/slashh/SlashhApp";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Slashh AI — Private, on-device stress relief" },
      { name: "description", content: "A calm, privacy-first voice stress monitor and mental wellbeing companion. Your voice stays on your device." },
      { property: "og:title", content: "Slashh AI — Private, on-device stress relief" },
      { property: "og:description", content: "A calm, privacy-first voice stress monitor and mental wellbeing companion." },
    ],
  }),
  component: SlashhApp,
});
