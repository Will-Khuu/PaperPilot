import type { Metadata } from "next";
import type { ReactNode } from "react";
import { StateProvider } from "./state";
import { Shell } from "@/components/shell";
import "./globals.css";

export const metadata: Metadata = {
  title: "PaperPilot",
  description: "Ask questions, compare research papers, and save cited answers.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="en"><body><StateProvider><Shell>{children}</Shell></StateProvider></body></html>;
}
