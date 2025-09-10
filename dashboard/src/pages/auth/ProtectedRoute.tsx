import { useAuth } from "@/context/AuthContext";
import { ReactNode } from "react";
import { Navigate } from "react-router-dom";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { session } = useAuth();

  // Allow bypassing auth when explicitly disabled or when Supabase is not configured
  const disableAuthFlag = String(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (import.meta as any).env?.VITE_DISABLE_AUTH ?? ""
  )
    .toLowerCase()
    .trim();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const supabaseUrl = (import.meta as any).env?.VITE_SUPABASE_URL as
    | string
    | undefined;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const supabaseKey = (import.meta as any).env?.VITE_SUPABASE_KEY as
    | string
    | undefined;
  const noSupabaseConfigured = !supabaseUrl || !supabaseKey;
  const authDisabled =
    disableAuthFlag === "true" || disableAuthFlag === "1" || disableAuthFlag === "yes";

  if (authDisabled || noSupabaseConfigured) {
    return <>{children}</>;
  }

  if (!session) {
    return <Navigate to="/sign-up" />;
  }

  return <>{children}</>;
}
