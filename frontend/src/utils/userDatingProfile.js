const STORAGE_KEY = 'LiangLocal-mirror-user-dating-profile';

export const DEFAULT_USER_PROFILE = {
  displayName: '',
  age: null,
  location: '',
  occupation: '',
  bio: '',
  seeking: '',
  relationshipStyle: 'exploring',
  interests: [],
  sectionPreferences: ['Intimate', 'Erotic'],
  preferredModality: 'both',
  turnOns: [],
  turnOffs: [],
  avatarUrl: '',
  createdAt: '',
  updatedAt: '',
};

export const RELATIONSHIP_STYLES = [
  { value: 'monogamous', label: 'Monogamous' },
  { value: 'poly', label: 'Polyamorous' },
  { value: 'open', label: 'Open relationship' },
  { value: 'exploring', label: 'Exploring / Not sure' },
  { value: 'undisclosed', label: 'Prefer not to say' },
];

export function addUserRating(profile, ratingData) {
  if (!profile) profile = { ...DEFAULT_USER_PROFILE };
  const ratings = [...(profile.ratings || [])];
  ratings.push({
    characterName: ratingData.characterName || 'Unknown',
    characterModel: ratingData.characterModel || '',
    rating: Math.max(1, Math.min(5, Number(ratingData.rating) || 3)),
    review: ratingData.review || '',
    conversationType: ratingData.conversationType || 'breakout',
    ratedAt: new Date().toISOString(),
  });
  const avg = ratings.reduce((s, r) => s + r.rating, 0) / ratings.length;
  return {
    ...profile,
    ratings,
    ratingCount: ratings.length,
    averagedRating: ratings.length >= 3 ? Math.round(avg * 10) / 10 : null,
  };
}

export function getAveragedRating(profile) {
  if (!profile?.ratings || profile.ratings.length < 3) return { rating: null, count: 0 };
  const r = profile.ratings;
  const avg = r.reduce((s, r2) => s + r2.rating, 0) / r.length;
  return { rating: Math.round(avg * 10) / 10, count: r.length };
}

export function loadUserDatingProfile() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      return { ...DEFAULT_USER_PROFILE, ...parsed };
    }
  } catch { }
  return { ...DEFAULT_USER_PROFILE };
}

export function saveUserDatingProfile(profile) {
  const now = new Date().toISOString();
  const data = {
    ...DEFAULT_USER_PROFILE,
    ...profile,
    updatedAt: now,
    createdAt: profile?.createdAt || now,
  };
  if (!data.displayName) data.displayName = 'User';
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  return data;
}

export function mergeFromAppProfile(appProfile) {
  if (!appProfile) return {};
  const updates = {};
  if (appProfile.name && !appProfile.name.startsWith('profile_')) {
    updates.displayName = appProfile.name;
  } else if (appProfile.username) {
    updates.displayName = appProfile.username;
  }
  if (appProfile.avatar) {
    updates.avatarUrl = appProfile.avatar;
  }
  return updates;
}

export function formatUserProfileForPrompt(profile) {
  if (!profile) return 'No profile information available.';
  const lines = [];
  if (profile.displayName) lines.push(`Name: ${profile.displayName}`);
  if (profile.age) lines.push(`Age: ${profile.age}`);
  if (profile.location) lines.push(`Location: ${profile.location}`);
  if (profile.occupation) lines.push(`Occupation: ${profile.occupation}`);
  if (profile.bio) lines.push(`About: ${profile.bio}`);
  if (profile.seeking) lines.push(`Seeking: ${profile.seeking}`);
  if (profile.relationshipStyle && profile.relationshipStyle !== 'undisclosed') {
    const style = RELATIONSHIP_STYLES.find(s => s.value === profile.relationshipStyle);
    if (style) lines.push(`Relationship style: ${style.label}`);
  }
  if (profile.interests?.length > 0) lines.push(`Interests: ${profile.interests.join(', ')}`);
  if (profile.sectionPreferences?.length > 0) lines.push(`Section preferences: ${profile.sectionPreferences.join(', ')}`);
  if (profile.preferredModality) {
    const modLabel = { text: 'Text only', neural_sex: 'Neural Sex', both: 'Both text and Neural Sex' };
    lines.push(`Preferred modality: ${modLabel[profile.preferredModality] || profile.preferredModality}`);
  }
  if (profile.turnOns?.length > 0) lines.push(`Turn-ons: ${profile.turnOns.join(', ')}`);
  if (profile.turnOffs?.length > 0) lines.push(`Turn-offs: ${profile.turnOffs.join(', ')}`);
  return lines.join('\n');
}
