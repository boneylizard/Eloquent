import { resolvePrimaryEndpointIdForRequest } from './resolveEndpointDisplay';

const EXCEPTION_PURPOSES = new Set([
  'character_intro',
  'call_mode_character_about',
  'create_character',
]);

function resolveAutoAnchor(settings = {}) {
  const endpoints = Array.isArray(settings?.customApiEndpoints) ? settings.customApiEndpoints : [];
  const rotating = endpoints.find((e) => e?.enabled !== false && e?.rotate_enabled !== false && e?.id);
  if (rotating?.id) return String(rotating.id);
  const anyEnabled = endpoints.find((e) => e?.enabled !== false && e?.id);
  return anyEnabled?.id ? String(anyEnabled.id) : null;
}

export function isRoutingExceptionPurpose(requestPurpose) {
  return EXCEPTION_PURPOSES.has(String(requestPurpose || '').trim());
}

export function resolveUnifiedRequestRoute({
  primaryModel,
  primaryIsAPI,
  settings,
  requestPurpose = null,
  overrideModel = null,
} = {}) {
  const selectedModel = primaryIsAPI
    ? resolvePrimaryEndpointIdForRequest(overrideModel || primaryModel || '', true, settings)
    : (overrideModel || primaryModel || '');
  const autoEnabled = Boolean(primaryIsAPI && settings?.apiEndpointRoundRobinEnabled === true);
  const exceptionPinned = isRoutingExceptionPurpose(requestPurpose) || Boolean(overrideModel);
  const effectiveModel = primaryIsAPI
    ? (
        autoEnabled && !exceptionPinned
          ? (resolveAutoAnchor(settings) || selectedModel)
          : selectedModel
      )
    : selectedModel;
  return {
    autoEnabled,
    selectedModel,
    effectiveModel,
    exceptionPinned,
  };
}

export function createRouteTraceId(prefix = 'router') {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function buildRouteTraceFields({ action, route, requestPurpose, traceId }) {
  return {
    trace_id: traceId || null,
    action: action || 'unknown',
    request_purpose: requestPurpose || 'none',
    auto_enabled: Boolean(route?.autoEnabled),
    selected_model: route?.selectedModel || 'none',
    effective_model: route?.effectiveModel || 'none',
    exception_pinned: Boolean(route?.exceptionPinned),
  };
}

export function logRouteTrace({ action, route, requestPurpose, traceId }) {
  const fields = buildRouteTraceFields({ action, route, requestPurpose, traceId });
  console.info(
    `${fields.action}_router_state trace_id=${fields.trace_id || 'none'} action=${fields.action} request_purpose=${fields.request_purpose} auto_enabled=${fields.auto_enabled} selected_model=${fields.selected_model} effective_model=${fields.effective_model} exception_pinned=${fields.exception_pinned}`,
  );
  return fields;
}

export function assertRouteContractOrThrow({ route, requestPurpose, traceId, action = 'unknown' }) {
  if (!route || isRoutingExceptionPurpose(requestPurpose)) return;
  if (
    route.autoEnabled === false
    && route.selectedModel
    && route.effectiveModel
    && route.selectedModel !== route.effectiveModel
  ) {
    const detail = `router_contract_mismatch_reconciled trace_id=${traceId || 'none'} action=${action} request_purpose=${requestPurpose || 'user_chat'} auto_enabled=false selected_model=${route.selectedModel} effective_model=${route.effectiveModel}`;
    console.warn(detail);
  }
}

export function extractRouteMetaFromGenerateResult(result = {}, headers = null) {
  const fromHeaders = (name) => headers?.get?.(name) || null;
  const action = result?.route_action || fromHeaders('X-Route-Action') || null;
  const requestPurpose = result?.route_purpose || fromHeaders('X-Route-Purpose') || null;
  const traceId = result?.route_trace_id || fromHeaders('X-Router-Trace-Id') || null;
  const selectedModel = result?.selected_model || fromHeaders('X-Route-Selected-Model') || null;
  const effectiveModel = result?.routed_model || fromHeaders('X-Route-Effective-Model') || null;
  const providerModel = result?.routed_provider_model || fromHeaders('X-Route-Provider-Model') || null;
  const autoRaw = result?.round_robin_enabled ?? fromHeaders('X-Route-Auto-Enabled');
  const pinnedRaw = result?.exception_pinned ?? fromHeaders('X-Route-Exception-Pinned');
  return {
    action: action || 'unknown',
    requestPurpose: requestPurpose || 'user_chat',
    traceId,
    selectedModel: selectedModel || null,
    effectiveModel: effectiveModel || null,
    providerModel: providerModel || null,
    autoEnabled: autoRaw === true || autoRaw === 'true',
    exceptionPinned: pinnedRaw === true || pinnedRaw === 'true',
    receivedAt: Date.now(),
  };
}
