"""
🎯 COPY YOUR ORIGINAL CODE HERE AND REPLACE ONLY THE SPELLING VALIDATION PART
This version works with your EXISTING tools_final.py
WITH TYPO FIX FOR PLACE NAMES (landon → london, etc.)
NO NEW INSTALLATIONS NEEDED - USES ONLY EXISTING IMPORTS
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END
import json
import re

# ✅ IMPORT ONLY FROM YOUR EXISTING CODE
from tools_final import tool1_weather, tool2_stock, tool3_qa

load_dotenv()

# ============================================================================
# STREAMLIT STATE INITIALIZATION
# ============================================================================

if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

memory = st.session_state.memory

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class AgentState(TypedDict, total=False):
    """State passed through the graph"""
    messages: Annotated[list, "shared"]
    chat_history: list
    weather_data: str
    stock_data: str
    qa_data: str
    final_answer: str
    tool_plan: dict
    execution_status: str

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    timeout=35
)

# ✅ NEW: Lightweight LLM for spell checking (optional)
try:
    spell_check_llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.1,
        timeout=10
    )
    USE_SPELL_CHECK = True
except Exception as e:
    print(f"⚠️ Spell check disabled: {e}")
    USE_SPELL_CHECK = False

# ============================================================================
# ✅ TYPO FIX - COMMON PLACE NAME CORRECTIONS
# ============================================================================

PLACE_NAME_CORRECTIONS = {
    'landon': 'london',
    'newyork': 'new york',
    'newyorkcity': 'new york city',
    'sanfransisco': 'san francisco',
    'sanfrancisco': 'san francisco',
    'losaneles': 'los angeles',
    'losangeles': 'los angeles',
    'losvegas': 'las vegas',
    'dubllin': 'dublin',
    'parris': 'paris',
    'berllin': 'berlin',
    'rom': 'rome',
    'tokyoo': 'tokyo',
    'singaproe': 'singapore',
    'bangkock': 'bangkok',
    'hongknog': 'hong kong',
    'newdelhi': 'new delhi',
    'mumbay': 'mumbai',
    'australai': 'australia',
    'mexcio': 'mexico',
    'toronoto': 'toronto',
    'vancover': 'vancouver',
    'sydnay': 'sydney',
    'melbouern': 'melbourne',
    'singapoure': 'singapore',
    'bangkk': 'bangkok',
    'istambul': 'istanbul',
    'instanbul': 'istanbul',
}

def correct_place_names(user_input: str) -> str:
    """
    ✅ FIX: Correct common place name typos
    Example: "weather in landon" → "weather in london"
    """
    words = user_input.lower().split()
    corrected_words = []
    
    for word in words:
        # Remove punctuation
        clean_word = word.strip('.,!?;:')
        
        # Check if it's in our corrections dictionary
        if clean_word in PLACE_NAME_CORRECTIONS:
            corrected_words.append(PLACE_NAME_CORRECTIONS[clean_word])
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

# ============================================================================
# SPELLING VALIDATION - NO EXTERNAL LIBRARIES (USES ONLY LLM)
# ============================================================================

def validate_spelling_with_llm(user_input: str) -> dict:
    """
    ✅ NO EXTERNAL LIBRARIES NEEDED
    Uses only existing Groq LLM to validate spelling
    Falls back gracefully if spell check disabled
    """
    
    # Short inputs don't need checking
    if len(user_input.strip()) < 3 or not USE_SPELL_CHECK:
        return {
            'has_errors': False,
            'misspelled_words': [],
            'corrected_text': user_input,
            'corrections': {},
            'original_text': user_input
        }
    
    try:
        validation_prompt = f"""Check spelling in: "{user_input}"

Respond with ONLY this JSON (no markdown):
{{"has_errors": true or false, "misspelled_words": [], "corrected_text": "corrected version"}}"""
        
        response = spell_check_llm.invoke(validation_prompt)
        response_text = response.content.strip()
        
        # Try to extract JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result['original_text'] = user_input
            if 'corrections' not in result:
                result['corrections'] = {}
            return result
    except Exception as e:
        print(f"⚠️ Spell check error: {e}")
    
    # Fallback: no errors
    return {
        'has_errors': False,
        'misspelled_words': [],
        'corrected_text': user_input,
        'corrections': {},
        'original_text': user_input
    }

# ============================================================================
# STEP 1: LLM DECIDES WHICH TOOLS TO USE
# ============================================================================

def planner_node(state: AgentState) -> AgentState:
    """LLM analyzes the user query and decides which tools to use."""
    user_query = state["messages"][-1].content
    
    planning_prompt = f"""You are an intelligent tool selector for a multi-agent system.

Available tools:
1. WEATHER_TOOL - Get real-time weather data for any location (OpenWeather)
2. STOCK_TOOL - Get real-time stock prices for any ticker symbol (Alpha Vantage)
3. QA_TOOL - Search the web and answer questions with latest information (Tavily)
4. LLM_TOOL - Use general knowledge for answering questions

User Query: "{user_query}"

Respond with ONLY this JSON:
{{
    "weather": "location_name or null",
    "stock": "ticker_symbol or null",
    "qa": "search_query or null",
    "use_llm": true or false
}}"""

    response = llm.invoke(planning_prompt)
    response_text = response.content.strip()
    
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            tool_plan = json.loads(json_match.group())
        else:
            tool_plan = {"weather": None, "stock": None, "qa": None, "use_llm": True}
    except json.JSONDecodeError:
        print(f"⚠️ Failed to parse LLM response: {response_text}")
        tool_plan = {"weather": None, "stock": None, "qa": None, "use_llm": True}
    
    print(f"📋 Tool Plan: {tool_plan}")
    return {**state, "tool_plan": tool_plan}

# ============================================================================
# STEP 2: WEATHER AGENT
# ============================================================================

def weather_agent_node(state: AgentState) -> AgentState:
    """Fetch weather data from OpenWeather API"""
    tool_plan = state.get("tool_plan", {})
    location = tool_plan.get("weather")
    
    if not location:
        return {**state, "weather_data": ""}
    
    try:
        print(f"🌍 Fetching weather for: {location}")
        weather_result = tool1_weather.invoke(location)
        print(f"✅ Weather API called successfully")
        return {**state, "weather_data": weather_result}
    except Exception as e:
        error_msg = f"❌ Error fetching weather: {str(e)}"
        print(error_msg)
        return {**state, "weather_data": error_msg}

# ============================================================================
# STEP 3: STOCK AGENT
# ============================================================================

def stock_agent_node(state: AgentState) -> AgentState:
    """Fetch stock data from Alpha Vantage API"""
    tool_plan = state.get("tool_plan", {})
    ticker = tool_plan.get("stock")
    
    if not ticker:
        return {**state, "stock_data": ""}
    
    try:
        print(f"💰 Fetching stock price for: {ticker}")
        stock_result = tool2_stock.invoke(ticker)
        print(f"✅ Stock API called successfully")
        return {**state, "stock_data": stock_result}
    except Exception as e:
        error_msg = f"❌ Error fetching stock data: {str(e)}"
        print(error_msg)
        return {**state, "stock_data": error_msg}

# ============================================================================
# STEP 4: QA AGENT
# ============================================================================

def qa_search_agent_node(state: AgentState) -> AgentState:
    """Fetch answers and search results from Tavily API"""
    tool_plan = state.get("tool_plan", {})
    query = tool_plan.get("qa")
    
    if not query:
        return {**state, "qa_data": ""}
    
    try:
        print(f"🔍 Searching for: {query}")
        qa_result = tool3_qa.invoke(query)
        print(f"✅ Tavily API called successfully")
        return {**state, "qa_data": qa_result}
    except Exception as e:
        error_msg = f"❌ Error fetching answer: {str(e)}"
        print(error_msg)
        return {**state, "qa_data": error_msg}

# ============================================================================
# STEP 5: FINAL QA AGENT
# ============================================================================

def final_qa_agent_node(state: AgentState) -> AgentState:
    """Generate final answer using LLM with collected data"""
    user_input = state["messages"][-1].content
    past_messages = state.get("chat_history", [])
    
    conversation_history = ""
    if past_messages:
        conversation_history = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in past_messages[-6:]
        ])
    
    context_data = ""
    has_external_data = False
    
    if state.get("weather_data"):
        context_data += f"\n[WEATHER DATA]\n{state['weather_data']}\n"
        has_external_data = True
    if state.get("stock_data"):
        context_data += f"\n[STOCK DATA]\n{state['stock_data']}\n"
        has_external_data = True
    if state.get("qa_data"):
        context_data += f"\n[SEARCH RESULTS]\n{state['qa_data']}\n"
        has_external_data = True
    
    if has_external_data:
        qa_prompt = f"""You are a helpful assistant. The user asked: "{user_input}"

I have fetched real-time data from external APIs. Use this data in your answer.

{context_data}

{f"Conversation History:{conversation_history}" if conversation_history else ""}

User's Question: {user_input}"""
    else:
        qa_prompt = f"""You are a helpful assistant.

{f"Conversation History:{conversation_history}" if conversation_history else ""}

User's Question: {user_input}"""

    print(f"🤖 Generating final answer...")
    response = llm.invoke(qa_prompt)
    answer = response.content.strip()
    
    updated_history = past_messages + [
        HumanMessage(content=user_input),
        AIMessage(content=answer)
    ]
    
    return {
        **state,
        "final_answer": answer,
        "chat_history": updated_history,
        "execution_status": "completed"
    }

# ============================================================================
# ROUTERS
# ============================================================================

def route_from_planner(state: AgentState) -> str:
    tool_plan = state.get("tool_plan", {})
    if tool_plan.get("weather"):
        return "weather_agent"
    elif tool_plan.get("stock"):
        return "stock_agent"
    elif tool_plan.get("qa"):
        return "qa_agent"
    else:
        return "final_qa"

def route_from_weather(state: AgentState) -> str:
    tool_plan = state.get("tool_plan", {})
    if tool_plan.get("stock"):
        return "stock_agent"
    elif tool_plan.get("qa"):
        return "qa_agent"
    else:
        return "final_qa"

def route_from_stock(state: AgentState) -> str:
    tool_plan = state.get("tool_plan", {})
    if tool_plan.get("qa"):
        return "qa_agent"
    else:
        return "final_qa"

def route_from_qa(state: AgentState) -> str:
    return "final_qa"

# ============================================================================
# BUILD THE LANGGRAPH
# ============================================================================

graph = StateGraph(AgentState)
graph.set_entry_point("planner")

graph.add_node("planner", planner_node)
graph.add_node("weather_agent", weather_agent_node)
graph.add_node("stock_agent", stock_agent_node)
graph.add_node("qa_agent", qa_search_agent_node)
graph.add_node("final_qa", final_qa_agent_node)

graph.add_conditional_edges("planner", route_from_planner,
    {"weather_agent": "weather_agent", "stock_agent": "stock_agent",
     "qa_agent": "qa_agent", "final_qa": "final_qa"})

graph.add_conditional_edges("weather_agent", route_from_weather,
    {"stock_agent": "stock_agent", "qa_agent": "qa_agent", "final_qa": "final_qa"})

graph.add_conditional_edges("stock_agent", route_from_stock,
    {"qa_agent": "qa_agent", "final_qa": "final_qa"})

graph.add_conditional_edges("qa_agent", route_from_qa, {"final_qa": "final_qa"})

graph.add_edge("final_qa", END)

multiagent_app = graph.compile()

# ============================================================================
# INVOKE THE MULTIAGENT
# ============================================================================

def invoke_multiagent(user_input: str) -> str:
    """Main function to run the multiagent pipeline"""
    
    # ✅ FIX TYPOS IN PLACE NAMES FIRST
    user_input = correct_place_names(user_input)
    
    print(f"\n{'='*60}")
    print(f"📝 User Input: {user_input}")
    print(f"{'='*60}\n")
    
    memory.add_user_message(user_input)
    
    chat_messages = []
    for role, content in st.session_state.chat_history:
        if role == "user":
            chat_messages.append(HumanMessage(content=content))
        elif role == "bot":
            chat_messages.append(AIMessage(content=content))
    
    chat_messages.append(HumanMessage(content=user_input))
    
    initial_state = {
        "messages": chat_messages,
        "chat_history": chat_messages,
        "weather_data": "",
        "stock_data": "",
        "qa_data": "",
        "tool_plan": {}
    }
    
    result = multiagent_app.invoke(initial_state)
    final_answer = result.get("final_answer", "Sorry, I couldn't generate a response.")
    memory.add_ai_message(final_answer)
    
    print(f"\n✅ Final Answer Generated")
    print(f"{'='*60}\n")
    
    return final_answer

# ============================================================================
# STREAMLIT UI
# ============================================================================

if __name__ == "__main__":
    try:
        import streamlit.web.bootstrap
        IS_STREAMLIT = True
    except ImportError:
        IS_STREAMLIT = False

    if IS_STREAMLIT or "STREAMLIT_SERVER_HEADLESS" in os.environ:
        st.set_page_config(
            page_title="🧠 LangGraph Chatbot",
            layout="centered",
            initial_sidebar_state="collapsed"
        )
        
        st.markdown("""
        <style>
        .stChatMessage { padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; }
        .stChatMessage.user { background-color: #e3f2fd; border-left: 4px solid #2196F3; }
        .stChatMessage.assistant { background-color: #f5f5f5; border-left: 4px solid #666; }
        .spelling-error { background-color: #fff3cd; border-left: 4px solid #ffc107;
                          padding: 12px; border-radius: 4px; margin-bottom: 10px; }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("🧠 LangGraph Multi-Agent Chatbot")
        st.markdown("""**Smart tool selection with real API data + Spelling Validation**
        - 🌍 Weather from OpenWeather
        - 💰 Stocks from Alpha Vantage
        - 🔍 Search & QA from Tavily
        - ✅ LLM-based spelling validation + Place name typo correction
        """)
        
        user_input = st.chat_input("Ask me anything...")
        
        if user_input:
            # ✅ SPELLING VALIDATION
            spelling_result = validate_spelling_with_llm(user_input)
            
            if spelling_result['has_errors'] and spelling_result['corrections']:
                st.markdown('<div class="spelling-error"><strong>⚠️ Spelling Issues Detected</strong></div>',
                           unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Misspelled words:**")
                    for word in spelling_result['misspelled_words'][:5]:
                        st.write(f"• {word}")
                
                with col2:
                    st.write("**Original:** " + spelling_result['original_text'][:50] + "...")
                    st.write("**Corrected:** " + spelling_result['corrected_text'][:50] + "...")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Use Corrected", key="correct"):
                        st.session_state.chat_history.append(("user", spelling_result['corrected_text']))
                        with st.spinner("🧠 Thinking..."):
                            reply = invoke_multiagent(spelling_result['corrected_text'])
                        st.session_state.chat_history.append(("bot", reply))
                        st.rerun()
                
                with col2:
                    if st.button("✅ Use Original", key="original"):
                        st.session_state.chat_history.append(("user", user_input))
                        with st.spinner("🧠 Thinking..."):
                            reply = invoke_multiagent(user_input)
                        st.session_state.chat_history.append(("bot", reply))
                        st.rerun()
            else:
                st.session_state.chat_history.append(("user", user_input))
                with st.spinner("🧠 Thinking and fetching data..."):
                    reply = invoke_multiagent(user_input)
                st.session_state.chat_history.append(("bot", reply))
                st.rerun()
        
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)
    
    else:
        # CLI mode
        print("\n🤖 LangGraph Multi-Agent Chatbot (CLI Mode)")
        print("Type 'exit' or 'quit' to stop\n")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye! 👋")
                break
            if not user_input:
                continue
            
            spelling_result = validate_spelling_with_llm(user_input)
            
            if spelling_result['has_errors'] and spelling_result['corrections']:
                print(f"\n⚠️ Misspelled: {', '.join(spelling_result['misspelled_words'][:3])}")
                choice = input("Use [o]riginal or [c]orrected? (o/c): ").lower().strip()
                input_to_use = spelling_result['corrected_text'] if choice == 'c' else user_input
            else:
                input_to_use = user_input
            
            response = invoke_multiagent(input_to_use)
            print(f"\n🤖 Bot: {response}\n")