/**

 * Dedicated API routing for auxiliary /generate flows (intro, call-mode about).

 * All requests for these flows must go through buildFlowGenerateRequestBody so

 * flow_api_url / flow_api_model / flow_api_key stay pinned across retries.

 */



import { formatApiError, normalizeEndpointModelId } from './chatlogCondenserUtils';



export const FLOW_REQUEST_PURPOSES = {

  characterIntro: 'character_intro',

  systemIntro: 'system_intro',

  callModeAbout: 'call_mode_character_about',

};



/** system_intro shares character-intro API override settings */

const FLOW_SETTINGS_PREFIX = {

  characterIntro: 'characterIntro',

  systemIntro: 'characterIntro',

  callModeAbout: 'callModeAboutCharacter',

};



function pickEnabledEndpoint(settings, endpointId) {

  if (!endpointId) return null;

  const id = String(endpointId).trim();

  if (!id) return null;

  const norm = normalizeEndpointModelId(id);

  return (settings.customApiEndpoints || []).find((ep) => {

    if (!ep?.enabled || !ep?.id) return false;

    const epNorm = normalizeEndpointModelId(ep.id);

    return ep.id === id || epNorm === norm || epNorm === id || ep.id === norm.replace(/^endpoint-/, '');

  }) || null;

}



function defaultGenerateUrl(apiUrl) {

  return `${String(apiUrl || '').replace(/\/$/, '')}/generate`;

}



/**

 * @param {'characterIntro'|'systemIntro'|'callModeAbout'} flowKind

 * @param {object} options

 * @returns {{

 *   url: string,

 *   modelName: string,

 *   extraBody: Record<string, string>,

 *   overrideActive: boolean,

 * }}

 */

export function resolveFlowGenerateConfig({

  flowKind,

  settings = {},

  apiUrl,

  fallbackModelName = 'default',

}) {

  const prefix = FLOW_SETTINGS_PREFIX[flowKind] || 'characterIntro';

  const overrideOn = settings[`${prefix}ApiOverrideEnabled`] === true;

  const endpointId = String(settings[`${prefix}ApiEndpointId`] || '').trim();

  const urlOverride = String(settings[`${prefix}Endpoint`] || '').trim();



  // Legacy: custom URL without override toggle

  if (!overrideOn) {

    const legacyUrl =

      urlOverride

      || (flowKind === 'systemIntro'

        ? (settings.systemIntroEndpoint || settings.characterIntroEndpoint || settings.callModeAboutCharacterEndpoint || '').trim()

        : '');

    return {

      url: legacyUrl || defaultGenerateUrl(apiUrl),

      modelName: fallbackModelName || 'default',

      extraBody: {},

      overrideActive: false,

    };

  }



  const matched = pickEnabledEndpoint(settings, endpointId);

  if (!matched?.id) {

    throw new Error(

      'Dedicated API is enabled but no custom endpoint is selected. '

      + 'Choose one under Settings → Character intro / Call mode About → Dedicated API.'

    );

  }



  const modelName = normalizeEndpointModelId(matched.id);

  const base = String(matched.url || '').replace(/\/$/, '');

  if (!base) {

    throw new Error(

      `Custom API endpoint "${matched.name || matched.id}" has no URL configured.`

    );

  }



  const epModel = String(matched.model || '').trim();

  const epKey = String(matched.apiKey || '').trim();



  const extraBody = {

    flow_api_url: base,

    flow_api_model: epModel || 'gpt-3.5-turbo',

  };

  if (epKey) extraBody.flow_api_key = epKey;



  const url = urlOverride || defaultGenerateUrl(apiUrl);



  return {

    url,

    modelName,

    extraBody,

    overrideActive: true,

  };

}



/**

 * Merge flow override fields into a /generate JSON body (use for every attempt, including JSON repair).

 */

export function buildFlowGenerateRequestBody({

  flowKind,

  settings = {},

  apiUrl,

  fallbackModelName = 'default',

  basePayload = {},

}) {

  const flow = resolveFlowGenerateConfig({

    flowKind,

    settings,

    apiUrl,

    fallbackModelName,

  });



  const model_name = flow.overrideActive

    ? flow.modelName

    : (basePayload.model_name || flow.modelName || fallbackModelName || 'default');



  return {

    ...basePayload,

    model_name,

    ...flow.extraBody,

  };

}



/** Read FastAPI error detail from a failed /generate response. */
export async function readFlowGenerateError(response) {
  try {
    const data = await response.json();
    return formatApiError(data, response.statusText);
  } catch {
    return `Request failed (${response.status})`;
  }
}

/** @deprecated Use buildFlowGenerateRequestBody */

export function mergeFlowGeneratePayload(basePayload, flowConfig) {

  if (!flowConfig?.extraBody || !Object.keys(flowConfig.extraBody).length) {

    return basePayload;

  }

  return { ...basePayload, ...flowConfig.extraBody };

}


