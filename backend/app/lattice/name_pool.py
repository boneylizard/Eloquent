"""Controlled name pool for Mirror entity generation. Prevents name repetition across characters."""

import random

FEMALE_FIRST_NAMES = {
    "classic_traditional": [
        "Sarah", "Elizabeth", "Margaret", "Catherine", "Anne",
        "Mary", "Jane", "Alice", "Laura", "Julia",
        "Caroline", "Eleanor", "Louise", "Rose", "Grace",
        "Claire", "Diana", "Rebecca", "Susan", "Helen",
        "Katherine", "Victoria", "Anna", "Lucy", "Emma",
    ],
    "modern_popular": [
        "Emily", "Jessica", "Ashley", "Amanda", "Jennifer",
        "Lauren", "Megan", "Stephanie", "Rachel", "Nicole",
        "Hannah", "Olivia", "Sophia", "Isabella", "Charlotte",
        "Amelia", "Brooklyn", "Samantha", "Kayla", "Taylor",
        "Morgan", "Kylie", "Haley", "Alexis", "Mackenzie",
    ],
    "warm_friendly": [
        "Sarah", "Emily", "Hannah", "Lauren", "Megan",
        "Rachel", "Katherine", "Rebecca", "Abigail", "Ella",
        "Molly", "Holly", "Amy", "Erin", "Leah",
        "Bethany", "Natalie", "Paige", "Courtney", "Whitney",
        "Kimberly", "Brittany", "Caitlin", "Erin", "Shannon",
    ],
    "sweet_southern": [
        "Scarlett", "Savannah", "Magnolia", "Georgia", "Carolina",
        "Dixie", "Sadie", "Mae", "June", "Rose",
        "Annabelle", "Bailey", "Lily", "Piper", "Harper",
        "Faith", "Hope", "Joy", "Summer", "Skye",
        "Dakota", "Cheyenne", "Aspen", "Willow", "Marlowe",
    ],
    "professional_polished": [
        "Alexandra", "Christina", "Victoria", "Danielle", "Gabrielle",
        "Valerie", "Michelle", "Diana", "Audrey", "Natalie",
        "Catherine", "Katherine", "Madeline", "Evelyn", "Vivian",
        "Brianna", "Morgan", "Peyton", "Reagan", "Quinn",
        "Sage", "Rowan", "Sydney", "Jordan", "Avery",
    ],
}

FEMALE_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Miller", "Davis", "Wilson", "Moore", "Taylor",
    "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Robinson", "Clark", "Lewis",
    "Walker", "Hall", "Allen", "Young", "King",
    "Wright", "Scott", "Adams", "Baker", "Carter",
    "Mitchell", "Roberts", "Turner", "Phillips", "Campbell",
    "Parker", "Evans", "Collins", "Stewart", "Morris",
    "Reed", "Cook", "Morgan", "Bell", "Cooper",
    "Bailey", "Cox", "Howard", "Ward", "Brooks",
]


def get_name_pool_block(existing_names=None, count=1):
    """Return a prompt block with name constraints for entity generation."""
    categories = list(FEMALE_FIRST_NAMES.keys())
    lines = ["NAME CONSTRAINTS:"]
    lines.append("Choose a full name (first + last) from the controlled name pool below.")
    lines.append("DO NOT use any name already active in the Mirror dating pool.")
    lines.append("Each character must have a distinct first name not shared with any other character.")

    if existing_names:
        names_str = ", ".join(existing_names)
        lines.append(f"Already in use: {names_str}")

    lines.append("")
    for cat in categories:
        cat_names = random.sample(FEMALE_FIRST_NAMES[cat], min(15, len(FEMALE_FIRST_NAMES[cat])))
        label = cat.replace("_", " ").title()
        lines.append(f"{label}: {', '.join(sorted(cat_names))}")

    last_sample = random.sample(FEMALE_LAST_NAMES, 20)
    lines.append(f"")
    lines.append(f"Last names (pick or compose in the same style): {', '.join(sorted(last_sample))}")

    lines.append(f"")
    lines.append(f"You are generating {count} character(s). Each must have a distinct full name.")
    lines.append("The first name pool is exhaustive — DO NOT invent new first names outside this list.")
    lines.append("Last names may be composed in the same common style as those listed.")

    return "\n".join(lines)
