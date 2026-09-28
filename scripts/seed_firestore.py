import logging
from google.cloud import firestore

# Hardcoded Project ID string as instructed (do not use google.auth.default() or GOOGLE_CLOUD_PROJECT)
PROJECT_ID = "qwiklabs-gcp-01-3f52eb6cad5a"
COLLECTION_NAME = "destinations"

logging.basicConfig(level=logging.INFO)

SEED_DESTINATIONS = [
    {
        "destination_id": "kyoto-japan",
        "name": "Kyoto",
        "country": "Japan",
        "budget_level": "mid-range",
        "travel_style": "culture",
        "description": "Ancient imperial capital known for historic temples, serene bamboo groves, traditional tea houses, and cherry blossoms.",
        "popular_activities": [
            "Fushimi Inari Shrine Hike",
            "Arashiyama Bamboo Grove Walk",
            "Gion Geisha District Evening Tour",
            "Traditional Matcha Tea Ceremony"
        ],
        "recommended_season": "Spring / Autumn",
        "rating": 4.9,
        "estimated_daily_cost_usd": 160
    },
    {
        "destination_id": "amalfi-italy",
        "name": "Amalfi Coast",
        "country": "Italy",
        "budget_level": "luxury",
        "travel_style": "relaxation",
        "description": "Dramatic cliffside coastline with pastel-colored villages, lemon groves, turquoise waters, and world-class seafood.",
        "popular_activities": [
            "Path of the Gods Coastal Hike",
            "Boat Cruise to Capri",
            "Positano Village Exploration",
            "Limoncello Tasting in Ravello"
        ],
        "recommended_season": "May - September",
        "rating": 4.8,
        "estimated_daily_cost_usd": 350
    },
    {
        "destination_id": "banff-canada",
        "name": "Banff National Park",
        "country": "Canada",
        "budget_level": "mid-range",
        "travel_style": "nature",
        "description": "Breathtaking Canadian Rockies destination famous for turquoise glacial lakes, majestic peaks, wildlife, and alpine hiking.",
        "popular_activities": [
            "Canoeing on Lake Louise",
            "Moraine Lake Sunrise View",
            "Banff Upper Hot Springs Swim",
            "Johnston Canyon Ice/Waterfall Walk"
        ],
        "recommended_season": "June - September / Dec - March (Skiing)",
        "rating": 4.9,
        "estimated_daily_cost_usd": 180
    },
    {
        "destination_id": "bali-indonesia",
        "name": "Bali",
        "country": "Indonesia",
        "budget_level": "budget",
        "travel_style": "relaxation",
        "description": "Tropical paradise featuring lush rice terraces, vibrant spiritual culture, volcanic beaches, and wellness retreats.",
        "popular_activities": [
            "Tegallalang Rice Terrace Trek",
            "Uluwatu Temple Sunset & Kecak Dance",
            "Mount Batur Sunrise Volcano Hike",
            "Seminyak Beach Club Relaxation"
        ],
        "recommended_season": "April - October",
        "rating": 4.7,
        "estimated_daily_cost_usd": 65
    },
    {
        "destination_id": "reykjavik-iceland",
        "name": "Reykjavik & Golden Circle",
        "country": "Iceland",
        "budget_level": "luxury",
        "travel_style": "adventure",
        "description": "Land of fire and ice featuring geysers, roaring waterfalls, volcanic craters, geothermal spas, and Northern Lights.",
        "popular_activities": [
            "Blue Lagoon Geothermal Spa Bathing",
            "Golden Circle Waterfall Tour (Gullfoss & Geysir)",
            "Northern Lights Aurora Hunting",
            "Glacier Hiking in Vatnajökull"
        ],
        "recommended_season": "Sept - March (Aurora) / June - Aug (Midnight Sun)",
        "rating": 4.8,
        "estimated_daily_cost_usd": 280
    }
]

def seed_firestore():
    print(f"Connecting to Firestore with project ID: '{PROJECT_ID}'...")
    db = firestore.Client(project=PROJECT_ID)
    collection_ref = db.collection(COLLECTION_NAME)

    for item in SEED_DESTINATIONS:
        doc_id = item["destination_id"]
        doc_ref = collection_ref.document(doc_id)
        doc_ref.set(item)
        print(f"Seeded destination: {item['name']} ({doc_id})")

    print(f"Successfully seeded {len(SEED_DESTINATIONS)} destinations into '{COLLECTION_NAME}' collection!")

if __name__ == "__main__":
    seed_firestore()
