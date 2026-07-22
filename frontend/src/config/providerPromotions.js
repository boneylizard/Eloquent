import providerPromotions from './providerPromotions.json' with { type: 'json' };

const PUBLIC_STATUSES = new Set(['active', 'referral']);

export const MIRID_PROVIDER_CAMPAIGN = Object.freeze({ ...providerPromotions.campaign });

export function getProviderPromotion(providerId, manifest = providerPromotions) {
  const normalisedId = String(providerId || '').trim().toLowerCase();
  return (manifest?.providers || []).find(
    (provider) => String(provider?.providerId || '').trim().toLowerCase() === normalisedId,
  ) || null;
}

function isWithinOfferWindow(promotion, now) {
  const current = now instanceof Date ? now.getTime() : new Date(now).getTime();
  if (!Number.isFinite(current)) return false;
  const startsAt = promotion?.startsAt ? new Date(promotion.startsAt).getTime() : null;
  const endsAt = promotion?.endsAt ? new Date(promotion.endsAt).getTime() : null;
  if (startsAt && current < startsAt) return false;
  if (endsAt && current > endsAt) return false;
  return true;
}

export function getPublicProviderPromotion(providerId, {
  manifest = providerPromotions,
  now = new Date(),
} = {}) {
  const promotion = getProviderPromotion(providerId, manifest);
  if (!promotion || !PUBLIC_STATUSES.has(promotion.status) || !isWithinOfferWindow(promotion, now)) {
    return null;
  }
  if (promotion.status === 'active') {
    const hasRedemption = Boolean(promotion.promoCode || promotion.referralUrl);
    if (!hasRedemption || !String(promotion.customerBenefit || '').trim()) return null;
  }
  if (promotion.status === 'referral' && !promotion.referralUrl) return null;
  return promotion;
}

export function buildProviderCampaignUrl(providerId, destination, {
  manifest = providerPromotions,
} = {}) {
  const promotion = getProviderPromotion(providerId, manifest);
  if (!promotion) return '';
  const publicPromotion = getPublicProviderPromotion(providerId, { manifest });
  const rawUrl = publicPromotion?.referralUrl || promotion.purchaseUrls?.[destination] || '';
  if (!rawUrl) return '';

  try {
    const url = new URL(rawUrl);
    const campaign = manifest.campaign || {};
    if (!url.searchParams.has('utm_source')) url.searchParams.set('utm_source', campaign.source || 'mirid');
    if (!url.searchParams.has('utm_medium')) url.searchParams.set('utm_medium', campaign.medium || 'desktop-app');
    if (!url.searchParams.has('utm_campaign')) url.searchParams.set('utm_campaign', campaign.id || 'mirid-provider-partners');
    if (!url.searchParams.has('utm_content')) url.searchParams.set('utm_content', promotion.providerId);
    return url.toString();
  } catch {
    return rawUrl;
  }
}

export default providerPromotions;
