import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

export type UiEvent = components["schemas"]["UIEventBody"];

/**
 * Fire-and-forget Admin UI telemetry. The proxy validates the shape and drops the event unless telemetry
 * is on at the `full` level, so callers only pass route segments and values from code, never user data.
 */
export const recordUiEvent = (event: UiEvent): void => {
  fetchClient.POST("/telemetry/ui_events", { body: event }).catch(() => undefined);
};
