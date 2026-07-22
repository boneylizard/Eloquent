import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MIRID_PROVIDER_CAMPAIGN,
  buildProviderCampaignUrl,
  getPublicProviderPromotion,
} from './providerPromotions.js';

const manifest = {
  campaign: {
    id: 'mirid-provider-partners',
    code: 'MIRID',
    source: 'mirid',
    medium: 'desktop-app',
  },
  providers: [{
    providerId: 'example',
    status: 'active',
    promoCode: 'MIRID',
    customerBenefit: 'Ten per cent off the first purchase.',
    referralUrl: 'https://provider.example/join?ref=mirid',
    purchaseUrls: { credits: 'https://provider.example/credits' },
    startsAt: null,
    endsAt: null,
  }],
};

test('reserves MIRID as the canonical provider campaign code', () => {
  assert.equal(MIRID_PROVIDER_CAMPAIGN.code, 'MIRID');
});

test('adds stable campaign attribution to provider links', () => {
  const url = new URL(buildProviderCampaignUrl('example', 'credits', { manifest }));
  assert.equal(url.pathname, '/join');
  assert.equal(url.searchParams.get('ref'), 'mirid');
  assert.equal(url.searchParams.get('utm_source'), 'mirid');
  assert.equal(url.searchParams.get('utm_medium'), 'desktop-app');
  assert.equal(url.searchParams.get('utm_campaign'), 'mirid-provider-partners');
  assert.equal(url.searchParams.get('utm_content'), 'example');
});

test('does not publish proposed discounts', () => {
  const proposed = structuredClone(manifest);
  proposed.providers[0].status = 'proposed';
  assert.equal(getPublicProviderPromotion('example', { manifest: proposed }), null);
  assert.match(buildProviderCampaignUrl('example', 'credits', { manifest: proposed }), /\/credits/);
});

test('requires a concrete customer benefit before publishing an active offer', () => {
  const incomplete = structuredClone(manifest);
  incomplete.providers[0].customerBenefit = '';
  assert.equal(getPublicProviderPromotion('example', { manifest: incomplete }), null);
});
