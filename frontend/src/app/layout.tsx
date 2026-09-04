import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "SatQuery AI", description: "Agentic remote-sensing assistant" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
