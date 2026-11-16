import { lazy, Suspense } from "react";
import type { ComponentType, LazyExoticComponent } from "react";
import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import { AppShell } from "./AppShell";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { LoginPage } from "./auth/LoginPage";
import { RunProvider } from "./RunContext";
import { ToastProvider } from "./ToastContext";
import { ThemeProvider } from "../design-system/ThemeProvider";
import { LoadingState } from "../design-system/ui";
import { NotFoundPage } from "../pages/NotFoundPage";

const OverviewPage = lazy(() => import("../pages/OverviewPage").then((page) => ({ default: page.OverviewPage })));
const ForecastsPage = lazy(() => import("../pages/ForecastsPage").then((page) => ({ default: page.ForecastsPage })));
const SchedulePage = lazy(() => import("../pages/SchedulePage").then((page) => ({ default: page.SchedulePage })));
const InventoryPage = lazy(() => import("../pages/InventoryPage").then((page) => ({ default: page.InventoryPage })));
const KpisPage = lazy(() => import("../pages/KpisPage").then((page) => ({ default: page.KpisPage })));
const AssistantPage = lazy(() => import("../pages/AssistantPage").then((page) => ({ default: page.AssistantPage })));
const AuditPage = lazy(() => import("../pages/AuditPage").then((page) => ({ default: page.AuditPage })));
const RunsPage = lazy(() => import("../pages/RunsPage").then((page) => ({ default: page.RunsPage })));
const SettingsPage = lazy(() => import("../pages/SettingsPage").then((page) => ({ default: page.SettingsPage })));

function LazyPage({ page: Page }: { page: LazyExoticComponent<ComponentType> }) {
  return (
    <Suspense fallback={<main className="page-loading"><LoadingState label="Loading workspace page" /></main>}>
      <Page />
    </Suspense>
  );
}

function AuthenticatedRoute() {
  const { user, ready } = useAuth();
  if (!ready) {
    return (
      <main className="auth-loading">
        <LoadingState label="Restoring your session" />
      </main>
    );
  }
  if (!user) return <Navigate to="/login" replace />;
  return <Outlet />;
}

function PlanningLayout() {
  return (
    <RunProvider>
      <AppShell />
    </RunProvider>
  );
}

function RouteTree() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<AuthenticatedRoute />}>
        <Route element={<PlanningLayout />}>
          <Route index element={<LazyPage page={OverviewPage} />} />
          <Route path="forecasts" element={<LazyPage page={ForecastsPage} />} />
          <Route path="schedule" element={<LazyPage page={SchedulePage} />} />
          <Route path="inventory" element={<LazyPage page={InventoryPage} />} />
          <Route path="kpis" element={<LazyPage page={KpisPage} />} />
          <Route path="assistant" element={<LazyPage page={AssistantPage} />} />
          <Route path="audit" element={<LazyPage page={AuditPage} />} />
          <Route path="runs" element={<LazyPage page={RunsPage} />} />
          <Route path="settings" element={<LazyPage page={SettingsPage} />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Route>
    </Routes>
  );
}

export function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <ToastProvider>
          <BrowserRouter>
            <RouteTree />
          </BrowserRouter>
        </ToastProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
