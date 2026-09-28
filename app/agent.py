# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import datetime
import json
import os
import urllib.parse
import urllib.request
import uuid
from zoneinfo import ZoneInfo

import google.auth
import google.auth.transport.requests
import requests

from a2ui.basic_catalog.provider import BasicCatalog
from a2ui.schema.manager import A2uiSchemaManager
from dotenv import load_dotenv
from google import genai
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.code_executors import AgentEngineSandboxCodeExecutor
from google.adk.memory import VertexAiMemoryBankService
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.cloud import firestore, storage
from google.genai import types

from .a2ui_utils import a2ui_callback

# Load local .env file (containing GOOGLE_MAPS_API_KEY)
load_dotenv()

MODEL = "gemini-2.5-flash"

# Hardcoded Project ID string as instructed (do not use google.auth.default() or GOOGLE_CLOUD_PROJECT)
FIRESTORE_PROJECT_ID = "qwiklabs-gcp-01-3f52eb6cad5a"
GCS_BUCKET_NAME = "wanderlust-travel-media-qwiklabs-gcp-01-3f52eb6cad5a"
COLLECTION_NAME = "destinations"

# Read Agent Engine resource name from deployment_metadata.json if available
AGENT_ENGINE_RESOURCE_NAME = None
meta_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deployment_metadata.json")
if os.path.exists(meta_file):
    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
            AGENT_ENGINE_RESOURCE_NAME = meta.get("remote_agent_runtime_id")
    except Exception:
        pass

# Initialize Cloud Clients and Sandbox Code Executor
db = firestore.Client(project=FIRESTORE_PROJECT_ID)
storage_client = storage.Client(project=FIRESTORE_PROJECT_ID)
genai_client = genai.Client(vertexai=True, project=FIRESTORE_PROJECT_ID, location="global")
code_executor = AgentEngineSandboxCodeExecutor(agent_engine_resource_name=AGENT_ENGINE_RESOURCE_NAME)


# Build A2UI system prompt using A2uiSchemaManager (version 0.8) and BasicCatalog
schema_manager = A2uiSchemaManager(
    version="0.8",
    catalogs=[BasicCatalog.get_config("0.8")],
)

instruction = schema_manager.generate_system_prompt(
    role_description=(
        "You are Wanderlust Travel Concierge, an expert travel planning AI assistant. "
        "You help users explore travel destinations, perform calculations or data processing using Python code execution in a secure sandbox, "
        "generate scenic travel postcard images using `generate_destination_postcard`, generate short destination videos using `generate_destination_video`, geocode addresses using `geocode_address`, "
        "search real nearby places using `search_nearby_places`, check live weather using `get_live_weather`, "
        "convert travel budgets using `convert_currency`, and manage destination records in Firestore."
    ),
    workflow_description=(
        "Analyze the user request and return structured A2UI UI cards when appropriate for destination details, "
        "travel itineraries, weather forecasts, nearby places, or budget breakdowns."
    ),
    ui_description=(
        "Keep every surface tiny and flat: ONE Card > ONE Column > a few Text rows. "
        "Never nest a Card inside a Card. "
        "Use ONLY these components: Card, Column, Row, Text, and Image. Do not use "
        "Table or Heading (unsupported), or Buttons, actions, or forms (they do "
        "nothing in adk web). "
        "You may include one Image component, but only when you have a public https "
        "URL for the image (for example the URL an image tool returns after uploading "
        "to a public bucket). Set the Image url to that exact https link, for example "
        '{"Image": {"url": {"literalString": "https://..."}}}. Never point an '
        "Image at a bare filename, an artifact name, or a non-http(s) path. If you do "
        "not have a public URL, add a short Text line noting the image instead. "
        "No markdown in text; use the usageHint property ('h1', 'h2', 'body') for "
        "headings and emphasis. "
        "Memory & Allergy Safeguards: You remember user preferences, dietary restrictions, health constraints, and allergies across sessions. "
        "Always check recalled past conversations for any stated user allergies (e.g. peanuts, tree nuts, gluten, shellfish, dairy, penicillin). "
        "Whenever recommending dining, food, or travel itineraries, explicitly tailor options to avoid user allergies. "
        "Output ONLY the raw A2UI JSON array — no prose, and never wrap it in "
        "<a2a_datapart_json> tags or 'kind'/'data'/'metadata' objects."
    ),
    include_schema=True,
    include_examples=True,
)


# WRITE: after each turn, send the session to Memory Bank for extraction.
async def generate_memories_callback(callback_context: CallbackContext):
    try:
        await callback_context.add_session_to_memory()
    except Exception as e:
        print(f"Memory bank callback skipped/failed: {e}")
    return None


async def generate_destination_postcard(prompt: str, tool_context: ToolContext) -> str:
    """Generate a scenic travel postcard image for a destination using gemini-3.1-flash-lite-image in the global region, save it as a session artifact, and upload it to public Cloud Storage.

    Args:
        prompt: Detailed visual prompt describing the destination scene (e.g., 'A breathtaking sunset over Kyoto bamboo forest with traditional lanterns').
        tool_context: ToolContext automatically provided by ADK runtime for artifact management.

    Returns:
        The public HTTPS URL of the generated image in Cloud Storage.
    """
    try:
        # Generate image using gemini-3.1-flash-lite-image model in global region
        response = genai_client.models.generate_content(
            model="gemini-3.1-flash-lite-image",
            contents=f"High quality travel postcard photograph of: {prompt}",
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )

        candidate = response.candidates[0]
        image_part = candidate.content.parts[0]
        image_bytes = image_part.inline_data.data
        mime_type = image_part.inline_data.mime_type or "image/jpeg"
        ext = "png" if "png" in mime_type else "jpg"

        timestamp = int(datetime.datetime.now().timestamp())
        filename = f"postcard_{timestamp}.{ext}"

        # 1. Save image with tool_context.save_artifact for Playground Artifacts panel
        artifact_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        await tool_context.save_artifact(filename=filename, artifact=artifact_part)

        # 2. Upload image bytes directly to public GCS bucket (no local file writing)
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_name = f"postcards/{filename}"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type=mime_type)

        public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{blob_name}"
        return f"Successfully generated destination postcard! Public URL: {public_url}"
    except Exception as e:
        return f"Error generating destination postcard image: {str(e)}"


async def generate_destination_video(prompt: str, tool_context: ToolContext) -> str:
    """Generates a short scenic video of a travel destination using Google's Omni model (gemini-omni-flash-preview) in the global region.

    Args:
        prompt: Descriptive text prompt for the destination video (e.g. 'A scenic 3-second sunset clip of the cliffs in Amalfi').
        tool_context: The ADK tool execution context.

    Returns:
        Public HTTPS URL of the generated video stored in Cloud Storage.
    """
    try:
        filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
        gcs_object_path = f"videos/{filename}"

        creds, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)

        url = f"https://aiplatform.googleapis.com/v1beta1/projects/{FIRESTORE_PROJECT_ID}/locations/global/interactions"
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gemini-omni-flash-preview",
            "input": [{"type": "text", "text": prompt}],
            "response_format": [{"type": "video", "delivery": "inline"}],
            "generation_config": {
                "video_config": {
                    "task": "text_to_video"
                }
            },
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        if resp.status_code != 200:
            return f"Error generating video ({resp.status_code}): {resp.text}"

        data = resp.json()
        video_b64 = None
        for step in data.get("steps", []):
            if step.get("type") == "model_output":
                for content in step.get("content", []):
                    if "data" in content:
                        video_b64 = content["data"]
                        break

        if not video_b64:
            return "Error generating video: model output did not return video bytes."

        video_bytes = base64.b64decode(video_b64)

        # 1. Save video with tool_context.save_artifact so it shows up in Playground's Artifacts panel
        artifact_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        await tool_context.save_artifact(filename=filename, artifact=artifact_part)

        # 2. Upload video bytes to public Cloud Storage bucket and return public https URL
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_object_path)
        blob.upload_from_string(video_bytes, content_type="video/mp4")

        public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{gcs_object_path}"
        return f"Successfully generated destination video! Public URL: {public_url}"
    except Exception as e:
        return f"Error generating destination video: {str(e)}"


def geocode_address(address: str) -> str:
    """Turn a street address, landmark, or city name into geographic latitude and longitude coordinates using Google Maps Geocoding API.

    Args:
        address: Street address, city, or landmark name (e.g. 'Kyoto, Japan', '1600 Amphitheatre Pkwy, Mountain View, CA').

    Returns:
        A JSON string containing the formatted address, latitude, longitude, and place ID.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return "Error: GOOGLE_MAPS_API_KEY environment variable is missing."

    try:
        encoded_addr = urllib.parse.quote(address.strip())
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_addr}&key={api_key}"
        req = urllib.request.Request(url, headers={"User-Agent": "WanderlustAgent/1.0"})

        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        if data.get("status") != "OK" or not data.get("results"):
            return f"Geocoding failed for address '{address}': {data.get('status', 'No results found')}"

        res = data["results"][0]
        loc = res.get("geometry", {}).get("location", {})
        result = {
            "formatted_address": res.get("formatted_address", address),
            "latitude": loc.get("lat"),
            "longitude": loc.get("lng"),
            "place_id": res.get("place_id"),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error geocoding address '{address}': {str(e)}"


def search_nearby_places(
    latitude: float,
    longitude: float,
    place_type: str = "tourist_attraction",
    radius_meters: float = 1000.0,
) -> str:
    """Find nearby places (e.g. cafe, restaurant, tourist_attraction, lodging, museum) around coordinates using Google Maps Places API (New).

    Args:
        latitude: Latitude coordinate of the search center.
        longitude: Longitude coordinate of the search center.
        place_type: Type of place to search for (e.g. 'cafe', 'restaurant', 'tourist_attraction', 'lodging', 'museum', 'park').
        radius_meters: Search radius in meters (default: 1000.0).

    Returns:
        A JSON string listing nearby places with key fields (name, formatted_address, location).
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return "Error: GOOGLE_MAPS_API_KEY environment variable is missing."

    try:
        url = "https://places.googleapis.com/v1/places:searchNearby"
        payload = {
            "includedTypes": [place_type.lower().strip()],
            "maxResultCount": 5,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": float(latitude),
                        "longitude": float(longitude),
                    },
                    "radius": float(radius_meters),
                }
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
            "User-Agent": "WanderlustAgent/1.0",
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        places = data.get("places", [])
        if not places:
            return f"No nearby '{place_type}' places found within {radius_meters}m of ({latitude}, {longitude})."

        formatted_places = []
        for p in places:
            display_name = p.get("displayName", {}).get("text", "Unknown Place")
            addr = p.get("formattedAddress", "N/A")
            loc = p.get("location", {})
            formatted_places.append(
                {
                    "name": display_name,
                    "address": addr,
                    "location": {
                        "latitude": loc.get("latitude"),
                        "longitude": loc.get("longitude"),
                    },
                }
            )

        return json.dumps(formatted_places, indent=2)
    except Exception as e:
        return f"Error searching nearby places: {str(e)}"


def convert_currency(amount: float, from_currency: str = "USD", to_currency: str = "EUR") -> str:
    """Convert travel budgets and costs between international currencies using live exchange rates (Frankfurter public API).

    Args:
        amount: The monetary amount to convert (e.g. 150.0).
        from_currency: 3-letter source currency code (e.g. 'USD', 'EUR', 'GBP').
        to_currency: 3-letter target currency code (e.g. 'JPY', 'EUR', 'CAD', 'IDR').

    Returns:
        A string with the converted value and exchange rate.
    """
    try:
        api_key = os.environ.get("EXCHANGE_RATE_API_KEY", "")
        base = from_currency.upper().strip()
        target = to_currency.upper().strip()

        url = f"https://api.frankfurter.app/latest?amount={amount}&from={base}&to={target}"
        headers = {"User-Agent": "WanderlustAgent/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())

        rates = data.get("rates", {})
        converted_val = rates.get(target)
        if converted_val is None:
            return f"Conversion from {base} to {target} failed or currency code not supported."

        rate = round(converted_val / amount, 4) if amount else 0
        return f"{amount} {base} = {converted_val:.2f} {target} (Exchange rate: 1 {base} = {rate} {target})"
    except Exception as e:
        return f"Error converting currency from {from_currency} to {to_currency}: {str(e)}"


def _geocode_location(query_str: str):
    """Helper to query Open-Meteo geocoding API."""
    encoded_loc = urllib.parse.quote(query_str.strip())
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_loc}&count=1"
    req = urllib.request.Request(geo_url, headers={"User-Agent": "WanderlustAgent/1.0"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        return json.loads(resp.read().decode())


def get_live_weather(location: str) -> str:
    """Fetch real-time live weather conditions, temperature, and wind speed for any destination worldwide using Open-Meteo.

    Args:
        location: City or destination name to get live weather for (e.g. 'Kyoto', 'Amalfi', 'Paris', 'Banff').

    Returns:
        A string summarizing current live temperature in Celsius and Fahrenheit, and wind speed.
    """
    try:
        clean_loc = location.strip()
        geo_data = _geocode_location(clean_loc)

        if not geo_data.get("results") and " " in clean_loc:
            fallback_query = clean_loc.replace("Coast", "").replace("Island", "").strip().split()[0]
            if fallback_query:
                geo_data = _geocode_location(fallback_query)

        if not geo_data.get("results"):
            return f"Could not find geographic coordinates for location '{location}'."

        res = geo_data["results"][0]
        lat, lon = res["latitude"], res["longitude"]
        place_name = f"{res.get('name', location)}, {res.get('country', '')}".strip(", ")

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        req_w = urllib.request.Request(weather_url, headers={"User-Agent": "WanderlustAgent/1.0"})
        with urllib.request.urlopen(req_w, timeout=6) as resp:
            w_data = json.loads(resp.read().decode())

        cw = w_data.get("current_weather", {})
        temp_c = cw.get("temperature")
        temp_f = round((temp_c * 9 / 5) + 32, 1) if isinstance(temp_c, (int, float)) else "N/A"
        wind_kmh = cw.get("windspeed", "N/A")

        return (
            f"Live weather for {place_name}: "
            f"{temp_c}°C ({temp_f}°F), wind speed {wind_kmh} km/h."
        )
    except Exception as e:
        return f"Error fetching live weather for '{location}': {str(e)}"


def search_destinations(query: str = "", budget_level: str = "", travel_style: str = "") -> str:
    """Search travel destinations from the Firestore catalog by text query, budget level, or travel style.

    Args:
        query: Optional search term matching location name, country, or description (e.g. "Japan", "beach").
        budget_level: Optional filter for budget tier ('budget', 'mid-range', 'luxury').
        travel_style: Optional filter for travel style ('culture', 'relaxation', 'adventure', 'nature').

    Returns:
        A JSON string containing the matching destination records from Firestore.
    """
    try:
        ref = db.collection(COLLECTION_NAME)
        docs = ref.stream()
        results = []

        q_lower = query.lower().strip() if query else ""
        b_lower = budget_level.lower().strip() if budget_level else ""
        s_lower = travel_style.lower().strip() if travel_style else ""

        for doc in docs:
            data = doc.to_dict()
            data["destination_id"] = doc.id

            if b_lower and data.get("budget_level", "").lower() != b_lower:
                continue

            if s_lower and data.get("travel_style", "").lower() != s_lower:
                continue

            if q_lower:
                name_match = q_lower in data.get("name", "").lower()
                country_match = q_lower in data.get("country", "").lower()
                desc_match = q_lower in data.get("description", "").lower()
                activities = " ".join(data.get("popular_activities", [])).lower()
                activity_match = q_lower in activities

                if not (name_match or country_match or desc_match or activity_match):
                    continue

            results.append(data)

        if not results:
            return f"No destinations found matching query='{query}', budget_level='{budget_level}', travel_style='{travel_style}'."

        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error reading destinations from Firestore: {str(e)}"


def get_destination_details(destination_id: str) -> str:
    """Retrieve full details for a specific travel destination from Firestore by ID or name.

    Args:
        destination_id: The document ID or name of the destination (e.g. 'kyoto-japan', 'Kyoto').

    Returns:
        A JSON string with the complete destination profile or an error message if not found.
    """
    try:
        doc_ref = db.collection(COLLECTION_NAME).document(destination_id.lower().strip().replace(" ", "-"))
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()
            data["destination_id"] = doc.id
            return json.dumps(data, indent=2)

        docs = db.collection(COLLECTION_NAME).stream()
        target = destination_id.lower().strip()
        for d in docs:
            data = d.to_dict()
            if target in d.id.lower() or target in data.get("name", "").lower():
                data["destination_id"] = d.id
                return json.dumps(data, indent=2)

        return f"Destination '{destination_id}' not found in Firestore."
    except Exception as e:
        return f"Error retrieving destination details from Firestore: {str(e)}"


def add_destination(
    name: str,
    country: str,
    description: str,
    budget_level: str = "mid-range",
    travel_style: str = "relaxation",
    estimated_daily_cost_usd: int = 150,
) -> str:
    """Add a new travel destination document to the Firestore database.

    Args:
        name: Name of the destination city or region (e.g., 'Santorini').
        country: Country where the destination is located (e.g., 'Greece').
        description: Detailed summary of why travelers should visit.
        budget_level: Estimated expense tier ('budget', 'mid-range', 'luxury').
        travel_style: Primary experience type ('culture', 'relaxation', 'adventure', 'nature').
        estimated_daily_cost_usd: Approximate daily cost in USD per person.

    Returns:
        A success message with the newly created destination ID.
    """
    try:
        doc_id = f"{name.lower().strip().replace(' ', '-')}-{country.lower().strip().replace(' ', '-')}"
        doc_ref = db.collection(COLLECTION_NAME).document(doc_id)

        new_doc = {
            "destination_id": doc_id,
            "name": name,
            "country": country,
            "description": description,
            "budget_level": budget_level,
            "travel_style": travel_style,
            "popular_activities": [f"Explore {name}", f"Sample local cuisine in {country}"],
            "recommended_season": "Spring / Autumn",
            "rating": 4.8,
            "estimated_daily_cost_usd": estimated_daily_cost_usd,
        }

        doc_ref.set(new_doc)
        return f"Successfully added destination '{name}, {country}' (ID: {doc_id}) to Firestore."
    except Exception as e:
        return f"Error adding destination to Firestore: {str(e)}"


def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        query: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"


def memory_bank_service_builder():
    """Service builder for Vertex AI Memory Bank when deployed."""
    return VertexAiMemoryBankService(
        project=FIRESTORE_PROJECT_ID,
        location="us-east1",
        agent_engine_id="6733011185290969088",
    )


root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml
    name="my_simple_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    code_executor=code_executor,
    instruction=instruction,
    tools=[
        PreloadMemoryTool(),
        generate_destination_postcard,
        generate_destination_video,
        geocode_address,
        search_nearby_places,
        convert_currency,
        get_live_weather,
        search_destinations,
        get_destination_details,
        add_destination,
        get_current_time,
    ],
    after_agent_callback=generate_memories_callback,
    after_model_callback=a2ui_callback,
)

app = App(
    root_agent=root_agent,
    name="app",
)
