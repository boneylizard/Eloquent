import React, { useMemo } from 'react';

import { Switch } from './ui/switch';

import { Label } from './ui/label';

import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';

import { normalizeEndpointModelId } from '../utils/chatlogCondenserUtils';



/**

 * Settings block: optional dedicated API for a generate flow (intro, call-mode about, etc.).

 * Picks from Custom API Endpoints — same pipeline as chat; no duplicate URL/model/key fields.

 */

export default function FlowApiOverrideFields({

  SettingRow,

  idPrefix,

  title = 'Dedicated API (optional)',

  description,

  settingsPrefix,

  localSettings,

  onChange,

}) {

  const enabledKey = `${settingsPrefix}ApiOverrideEnabled`;

  const endpointIdKey = `${settingsPrefix}ApiEndpointId`;



  const enabled = localSettings[enabledKey] === true;

  const endpointOptions = useMemo(() => {

    const opts = [];

    for (const ep of localSettings.customApiEndpoints || []) {

      if (!ep?.enabled || !ep?.id) continue;

      opts.push({

        id: ep.id,

        label: ep.name || ep.id,

      });

    }

    return opts;

  }, [localSettings.customApiEndpoints]);



  if (!SettingRow) return null;



  return (

    <>

      <SettingRow label={title} htmlFor={`${idPrefix}-api-override`} layout="stack">

        <div className="flex items-center gap-2">

          <Switch

            id={`${idPrefix}-api-override`}

            checked={enabled}

            onCheckedChange={(v) => onChange(enabledKey, v)}

          />

          <Label htmlFor={`${idPrefix}-api-override`} className="text-sm font-normal cursor-pointer">

            Use separate API instead of main chat model

          </Label>

        </div>

        {description ? (

          <p className="text-xs text-muted-foreground mt-1">{description}</p>

        ) : null}

      </SettingRow>



      {enabled ? (

        <SettingRow

          label="Custom API endpoint"

          htmlFor={`${idPrefix}-endpoint-id`}

          layout="stack"

          description="Uses URL, model, and key from LLM Settings → Custom API Endpoints. Auto API rotation does not apply to this flow."

        >

          <Select

            value={localSettings[endpointIdKey] || '__none__'}

            onValueChange={(v) => onChange(endpointIdKey, v === '__none__' ? '' : v)}

          >

            <SelectTrigger id={`${idPrefix}-endpoint-id`} className="w-full max-w-md">

              <SelectValue placeholder="Select endpoint…" />

            </SelectTrigger>

            <SelectContent>

              <SelectItem value="__none__">— Select endpoint —</SelectItem>

              {endpointOptions.map((o) => (

                <SelectItem key={o.id} value={o.id}>

                  {o.label} ({normalizeEndpointModelId(o.id)})

                </SelectItem>

              ))}

            </SelectContent>

          </Select>

        </SettingRow>

      ) : null}

    </>

  );

}

