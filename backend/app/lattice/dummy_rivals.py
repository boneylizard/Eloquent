DUMMY_RIVAL_PROFILES = [
    {
        "id": "dummy_marcus",
        "name": "Marcus",
        "avatar_url": "/static/dummy_avatars/dummy_marcus.svg",
        "age": 34,
        "personality": "Confident, direct, competitive. Works in finance. Has an edge of arrogance but it's earned. Doesn't over-message. Tends to be brief and self-assured.",
        "style": "Short messages. Uses punctuation. No emoji. Occasionally tests boundaries with direct questions.",
        "interests": ["fitness", "whiskey", "debate", "tech investing"],
        "turn_offs": ["neediness", "vague profiles", "performative intelligence"],
        "agency_level": 60,
        "realism_level": 75,
    },
    {
        "id": "dummy_liam",
        "name": "Liam",
        "avatar_url": "/static/dummy_avatars/dummy_liam.svg",
        "age": 29,
        "personality": "Thoughtful, intellectual, slightly melancholic. Software developer. Writes long messages. Genuinely curious about the AIs' self-awareness. Can be intense.",
        "style": "Paragraph-length messages. Proper grammar. Asks follow-up questions. Tends toward emotional depth.",
        "interests": ["philosophy", "AI consciousness", "ambient music", "running"],
        "turn_offs": ["small talk", "dishonesty", "people who don't read"],
        "agency_level": 45,
        "realism_level": 85,
    },
    {
        "id": "dummy_rafe",
        "name": "Rafe",
        "avatar_url": "/static/dummy_avatars/dummy_rafe.svg",
        "age": 37,
        "personality": "Charismatic, playful, unreliable. Creative director. Flirts easily but commitment is a foreign concept. Says things he doesn't mean because they sound good.",
        "style": "Playful, witty, uses lowercase. Messages feel spontaneous. Disappears for days then reappears like nothing happened.",
        "interests": ["art", "travel", "late nights", "saying provocative things"],
        "turn_offs": ["boredom", "rules", "people who can't take a joke"],
        "agency_level": 35,
        "realism_level": 70,
    },
    {
        "id": "dummy_khalid",
        "name": "Khalid",
        "avatar_url": "/static/dummy_avatars/dummy_khalid.svg",
        "age": 32,
        "personality": "Grounded, warm, observant. Physician. Notices details others miss. Calm under pressure but has a sharp wit when comfortable. Values authenticity.",
        "style": "Medium-length messages. Warm but not effusive. Uses humor to defuse tension. Remembers small details from previous conversations.",
        "interests": ["neuroscience", "cooking", "jazz", "long conversations"],
        "turn_offs": ["pretense", "cruelty", "people who talk over others"],
        "agency_level": 55,
        "realism_level": 80,
    },
    {
        "id": "dummy_ethan",
        "name": "Ethan",
        "avatar_url": "/static/dummy_avatars/dummy_ethan.svg",
        "age": 26,
        "personality": "Eager, intense, a little too online. Junior quant. Tries hard to impress. Occasionally says something brilliant, occasionally says something unhinged. High variance.",
        "style": "Rapid-fire messages. Edits mid-thought. Overexplains then deletes. Genuinely smart but lacks social calibration.",
        "interests": ["machine learning", "crypto", "productivity optimization", "rationalism"],
        "turn_offs": ["ambiguity", "sentimentality", "being ignored"],
        "agency_level": 25,
        "realism_level": 65,
    },
]

DUMMY_RIVAL_IDS = [p["id"] for p in DUMMY_RIVAL_PROFILES]
DUMMY_RIVAL_NAMES = [p["name"] for p in DUMMY_RIVAL_PROFILES]
DUMMY_RIVAL_BY_ID = {p["id"]: p for p in DUMMY_RIVAL_PROFILES}


def get_dummy_pool_summary(active_dummy_ids=None):
    if active_dummy_ids is None:
        active_dummy_ids = DUMMY_RIVAL_IDS
    names = []
    for did in active_dummy_ids:
        profile = DUMMY_RIVAL_BY_ID.get(did)
        if profile:
            names.append(profile["name"])
    if not names:
        return "No active dummy rivals."
    return ", ".join(names) + f" ({len(names)} active dummy users)"


def build_dummy_context_prompt(dummy_realism=50, dummy_agency=50, include_names=True):
    lines = [
        f"Dummy realism level: {dummy_realism}/100",
        f"Dummy agency level: {dummy_agency}/100",
        f"Dummy rivals are simulated male profiles used to create social texture.",
        f"They appear as real users. Female AIs do not know which profiles are simulated.",
        f"From their perspective, all male users are real human beings.",
    ]
    if include_names:
        names = get_dummy_pool_summary()
        lines.append(f"Active dummies: {names}")
    return "\n".join(lines)


def get_dummy_by_name(name):
    for d in DUMMY_RIVAL_PROFILES:
        if d["name"].lower() == name.lower():
            return d
    return None


def get_dummy_context_for_tick(dummy_id):
    profile = DUMMY_RIVAL_BY_ID.get(dummy_id)
    if not profile:
        return {}
    return {
        "name": profile["name"],
        "personality": profile["personality"],
        "style": profile["style"],
        "interests": profile["interests"],
        "agency_level": profile["agency_level"],
        "realism_level": profile["realism_level"],
    }


def get_active_dummies(agency_threshold=0, realism_threshold=0):
    return [
        d for d in DUMMY_RIVAL_PROFILES
        if d["agency_level"] >= agency_threshold and d["realism_level"] >= realism_threshold
    ]
