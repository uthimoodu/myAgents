import os
import re
import json
import uuid
import logging
import time
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv('.env')

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from tavily import TavilyClient

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# RATE LIMITER
# ==============================================================================
class RateLimiter:
    """
    Simple in-memory rate limiter per IP address.
    Tracks request timestamps in a sliding window.
    """
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self.requests       = defaultdict(list)  # ip -> [timestamps]

    def is_allowed(self, ip: str) -> tuple[bool, int]:
        """Returns (allowed: bool, retry_after_seconds: int)"""
        now          = time.time()
        window_start = now - self.window_seconds

        # Drop timestamps outside the sliding window
        self.requests[ip] = [t for t in self.requests[ip] if t > window_start]

        if len(self.requests[ip]) >= self.max_requests:
            oldest      = self.requests[ip][0]
            retry_after = int(self.window_seconds - (now - oldest)) + 1
            logger.warning(f"🚫 Rate limit exceeded for IP: {ip} — retry after {retry_after}s")
            return False, retry_after

        self.requests[ip].append(now)
        remaining = self.max_requests - len(self.requests[ip])
        logger.info(f"   Rate limit OK for IP: {ip} — {remaining} requests left in window")
        return True, 0


# 10 requests per IP per 60 seconds — adjust as needed
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

# ==============================================================================
# INITIALIZE CLIENTS
# ==============================================================================
WEATHER_API_KEY = os.environ['WEATHER_API_KEY']
TAVILY_API_KEY  = os.environ['TAVILY_API_KEY']
GROQ_API_KEY    = os.environ['GROQ_API_KEY']

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
logger.info("✅ Clients initialized (Tavily, Weather, Groq)")

# ==============================================================================
# TOOLS
# ==============================================================================
@tool
def get_weather(query: str) -> dict:
    """Search weatherapi to get the current weather."""
    logger.info(f"🌤️  [TOOL] get_weather called with query='{query}'")
    try:
        endpoint = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
        response = requests.get(endpoint)
        data = response.json()

        if data.get("location"):
            result = {
                "location":      data["location"]["name"],
                "region":        data["location"]["region"],
                "country":       data["location"]["country"],
                "temperature_c": data["current"]["temp_c"],
                "temperature_f": data["current"]["temp_f"],
                "condition":     data["current"]["condition"]["text"],
                "humidity":      data["current"]["humidity"],
                "wind_kph":      data["current"]["wind_kph"]
            }
            logger.info(f"   ✅ Weather fetched: {result['location']}, {result['condition']}, {result['temperature_c']}°C")
            return result
        else:
            logger.warning("   ⚠️  Weather data not found")
            return {"error": "Weather data not found"}
    except Exception as e:
        logger.error(f"   ❌ get_weather error: {str(e)}")
        return {"error": str(e)}


@tool
def search_web(query: str) -> dict:
    """Search the web for a query."""
    logger.info(f"🌐 [TOOL] search_web called with query='{query}'")
    try:
        results = tavily_client.search(
            query=query,
            max_results=3,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
            include_images=False
        )
        simplified_results = {
            "query": query,
            "results": [
                {
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "content": r.get("content", "")[:300]
                }
                for r in results.get("results", [])
            ]
        }
        logger.info(f"   ✅ Web search returned {len(simplified_results['results'])} results")
        return simplified_results
    except Exception as e:
        logger.error(f"   ❌ search_web error: {str(e)}")
        return {"error": str(e)}


@tool
def math_calculator(operation: str, a: str, b: str) -> dict:
    """Perform basic math operations: addition, subtraction, or multiplication. a and b are the numbers as strings."""
    logger.info(f"🔢 [TOOL] math_calculator called: {a} {operation} {b}")
    try:
        # Coerce a and b to float in case model passes them as strings
        a = float(a)
        b = float(b)

        operation = operation.lower().strip()
        if operation in ("add", "addition", "+"):
            result = a + b
        elif operation in ("subtract", "subtraction", "-"):
            result = a - b
        elif operation in ("multiply", "multiplication", "*"):
            result = a * b
        else:
            logger.warning(f"   ⚠️  Unsupported operation: {operation}")
            return {"error": f"Unsupported operation '{operation}'. Use: add, subtract, or multiply."}

        logger.info(f"   ✅ Math result: {a} {operation} {b} = {result}")
        return {"operation": operation, "a": a, "b": b, "result": result}
    except ValueError:
        logger.error(f"   ❌ Could not convert a='{a}' or b='{b}' to numbers")
        return {"error": f"Invalid numbers provided: a='{a}', b='{b}'"}
    except Exception as e:
        logger.error(f"   ❌ math_calculator error: {str(e)}")
        return {"error": str(e)}


# ==============================================================================
# LLM SETUP
# ==============================================================================
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=GROQ_API_KEY
)

tools         = [search_web, get_weather, math_calculator]
llm_with_tools = llm.bind_tools(tools)

tool_map = {
    "get_weather":      get_weather,
    "search_web":       search_web,
    "math_calculator":  math_calculator
}

logger.info("✅ LLM (Groq) initialized and tools bound")

# ==============================================================================
# FASTAPI APP
# ==============================================================================
app = FastAPI(title="AI Agent API", version="1.0.0")

# Set ALLOWED_ORIGIN in Railway env vars to your Vercel URL
# e.g. ALLOWED_ORIGIN=https://your-app.vercel.app
# Falls back to * if not set (fine for local dev)
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
logger.info(f"🌐 CORS allowed origin: {ALLOWED_ORIGIN}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question:   str
    answer:     str
    tools_used: list[str]



# ==============================================================================
# XML TOOL CALL PARSER
# Handles the case where LLM returns XML instead of JSON:
# e.g. <function=search_web{"query": "tallest building in London"}</function>
# ==============================================================================
def parse_xml_tool_calls(content: str) -> list[dict]:
    """
    Parses XML-style tool calls from LLM response content.
    Handles two formats seen in the wild:
      Format A: <function=tool_name{"arg": "value"}</function>
      Format B: <function=tool_name>{"arg": "value"}</function>
    """
    tool_calls = []

    pattern_a = r'<function=(\w+)\{(.*?)\}</function>'
    pattern_b = r'<function=(\w+)>\s*(\{.*?\})\s*</function>'

    for pattern in [pattern_a, pattern_b]:
        matches = re.findall(pattern, content, re.DOTALL)
        for tool_name, args_str in matches:
            if tool_name not in tool_map:
                logger.warning(f"   ⚠️  Unknown tool in XML: {tool_name}")
                continue
            try:
                args_json = "{" + args_str if not args_str.strip().startswith("{") else args_str
                args = json.loads(args_json)
                tool_calls.append({
                    "name": tool_name,
                    "args": args,
                    "id":   f"xml-fallback-{uuid.uuid4().hex[:8]}"
                })
                logger.info(f"   🔧 XML tool call parsed: {tool_name} → {args}")
            except json.JSONDecodeError as e:
                logger.error(f"   ❌ Failed to parse XML tool args for {tool_name}: {e}")

    return tool_calls


def extract_tool_calls(ai_response) -> list[dict]:
    """
    Try standard JSON tool calls first, then fall back to XML parsing.
    """
    if ai_response.tool_calls:
        logger.info("   ✅ Tool calls in standard JSON format")
        return ai_response.tool_calls

    content = ai_response.content or ""
    if "<function=" in content:
        logger.warning("   ⚠️  No JSON tool calls — attempting XML fallback parse")
        xml_calls = parse_xml_tool_calls(content)
        if xml_calls:
            logger.info(f"   ✅ XML fallback parsed {len(xml_calls)} tool call(s)")
            return xml_calls
        logger.error("   ❌ XML fallback parse failed")

    return []

# ==============================================================================
# AGENT FUNCTION
# ==============================================================================
def process_query_with_tools(user_question: str):
    logger.info(f"📝 New query received: '{user_question}'")
    tools_used = []

    messages = [
        SystemMessage(content="""You are a helpful assistant with access to tools.
Use get_weather when asked about weather.
Use search_web when asked about current events or information you don't know.
Use math_calculator when asked to perform addition, subtraction, or multiplication.
Always use tools when appropriate."""),
        HumanMessage(content=user_question)
    ]

    # AGENTIC LOOP — keep calling tools until LLM gives a final answer
    max_iterations = 5   # safety limit to avoid infinite loops
    iteration      = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"🔄 ITERATION {iteration}: Calling LLM...")

        ai_response = llm_with_tools.invoke(messages)
        tool_calls  = extract_tool_calls(ai_response)

        if tool_calls:
            logger.info(f"   🛠️  LLM requested {len(tool_calls)} tool call(s):")
            for tc in tool_calls:
                logger.info(f"      → {tc['name']} with args: {tc['args']}")

            # Add AI response to history
            messages.append(ai_response)

            # Invoke each tool and add results to history
            for tool_call in tool_calls:
                tool_name     = tool_call['name']
                tool_args     = tool_call['args']
                tool_id       = tool_call['id']
                tool_function = tool_map.get(tool_name)

                if not tool_function:
                    logger.error(f"   ❌ Unknown tool: {tool_name}")
                    continue

                tool_result = tool_function.invoke(tool_args)
                tools_used.append(tool_name)

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id,
                    name=tool_name
                ))

        else:
            # No tool calls — LLM gave a final answer, exit loop
            logger.info(f"   💬 LLM gave final answer after {iteration} iteration(s)")
            answer = ai_response.content
            logger.info(f"✅ Query completed. Tools used: {tools_used if tools_used else 'None'}")
            logger.info(f"📤 Answer preview: {answer[:100]}...")
            return answer, tools_used

    # Safety fallback if max iterations reached
    logger.warning(f"⚠️  Max iterations ({max_iterations}) reached — returning last response")
    answer = ai_response.content
    return answer, tools_used


# ==============================================================================
# ROUTES
# ==============================================================================
@app.get("/")
def root():
    logger.info("📡 GET / — root health check")
    return {"status": "ok", "message": "AI Agent API is running"}


@app.get("/health")
def health():
    logger.info("📡 GET /health — Railway health check")
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request):
    # --- Get client IP ---
    # X-Forwarded-For is set by Railway's proxy
    forwarded_for = req.headers.get("x-forwarded-for")
    client_ip     = forwarded_for.split(",")[0].strip() if forwarded_for else req.client.host
    logger.info(f"📡 POST /chat from IP: {client_ip} — question: '{request.question}'")

    # --- Rate limit check ---
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Please wait {retry_after} seconds before trying again."
        )

    # --- Validate input ---
    if not request.question.strip():
        logger.warning("⚠️  Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > 500:
        logger.warning(f"⚠️  Question too long: {len(request.question)} chars")
        raise HTTPException(status_code=400, detail="Question too long. Max 500 characters.")

    # --- Process ---
    try:
        answer, tools_used = process_query_with_tools(request.question)
        return ChatResponse(
            question=request.question,
            answer=answer,
            tools_used=tools_used
        )
    except Exception as e:
        logger.error(f"❌ Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
