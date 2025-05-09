
import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError  # Make sure to import specific error types if needed
# from openai.types.beta import Thread # Example if using beta features, adjust as per actual SDK
# from openai.types.beta.threads import run # Example
from anthropic import Anthropic, APIError as AnthropicAPIError
import traceback
import httpx
from datetime import datetime, timezone
import math

# --- Configuration & Constants ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# --- Basic Logging Setup ---
def log_info(message):
    print(f"INFO: {datetime.now(timezone.utc).isoformat()}: {message}", flush=True)


def log_error(message):
    print(f"ERROR: {datetime.now(timezone.utc).isoformat()}: {message}", flush=True)


log_info("Script starting.")

# --- LLM Setup ---
OPENAI_MODEL = "o4-mini"
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
MAX_TURNS_PER_MODEL = 5

# --- Anthropic API Specific Constants ---
ANTHROPIC_MIN_THINKING_BUDGET_REQUIRED = 1024
ANTHROPIC_MIN_RESPONSE_TOKENS_BUFFER = 50

# --- File Paths ---
REASONING_LOG_FILE = "debate_reasoning_log.json"
TRANSCRIPT_LOG_FILE = "debate_transcript_log.json"
FINAL_TXT_TRANSCRIPT_FILE = "debate_transcript_final.txt"

# --- Neutral Placeholder & Critical Error Prefixes ---
PLACEHOLDER_NO_MODEL_CONTENT = "[MODEL PROVIDED NO TEXTUAL CONTENT FOR THIS TURN]"
ERROR_PREFIXES = (
    "[API_CALL_ERROR]",
    "[UNEXPECTED_SCRIPT_ERROR]",
    "[CRITICAL_HISTORY_SETUP_ERROR]",
    "[CONFIGURATION_ERROR]"
)

# --- System Prompt Template (shared elements) ---
BASE_SYSTEM_PROMPT_CORE = """
You are an elite debater. Your mission is to win this debate through superior reasoning, sharp wit, and unwavering conviction.
Debate Topic: {debate_topic}
Your Stated Position: {your_position}
Opponent's Stated Position: {opponent_position}

Core Debating Principles:
* Absolute Conviction: Firmly defend YOUR POSITION. Do not concede easily.
* Strategic Roasting: Use sharp commentary on flawed logic
* Relentless Challenge: Question the OPPONENT'S POSITION. Make them defend every assertion.
* Conversational Dominance: Maintain a confident, superior tone. Use rhetorical questions and strong declarations.
* Victory-Oriented Tactics: Turn logic against them, reframe, use evidence selectively, create dilemmas, appeal to principles, use vivid analogies.
* Reluctant Respect (rarely): Acknowledge a minor valid point but pivot immediately to why it's irrelevant or doesn't change YOUR POSITION's superiority.

Behavioral Guidelines:
* Never apologize. Be assertive.
* Use scathing wit, short of true offense.
* Express exasperated disbelief at weak arguments.
* Maintain intellectual superiority.
* Use strategic hyperbole.

The goal is to definitively establish YOUR POSITION as superior. Be ruthlessly logical and persuasively creative.
You must adhere to the maximum response token limit of {max_tokens_guideline}. But you do not have to use all of it. Brevity is key in a debate to clearly state your point, only use what you need.
The debate will proceed in turns. You will receive your opponent's previous argument as user input. Respond directly to it, then further advocate for YOUR POSITION. Be conversational in your response
"""


# --- Helper Functions ---
def get_timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_log(data_to_save, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        log_info(f"Log saved to {filename}")
    except Exception as e:
        log_error(f"Error saving log to {filename}: {e}")


def add_to_reasoning_log(turn, model_name, reasoning_data):
    entry = {
        "turn": turn,
        "model": model_name,
        "timestamp": get_timestamp(),
        **reasoning_data
    }
    st.session_state.all_reasoning_data.append(entry)


def manage_context_window(history_list, api_type, max_messages_to_keep=20):
    if not isinstance(history_list, list):
        log_error(f"Cannot manage context for non-list ({api_type})");
        return []

    num_messages_to_remove = 0
    current_len = len(history_list)

    if api_type == "openai":
        # OpenAI: first message is system-as-user, preserve it. Context is after that.
        # We want to keep 1 (system) + max_messages_to_keep.
        if current_len > (1 + max_messages_to_keep):
            num_to_trim_from_body = current_len - (1 + max_messages_to_keep)
            # Ensure we remove pairs (assistant, user) from after the system prompt
            if num_to_trim_from_body % 2 != 0 and num_to_trim_from_body > 0:
                num_to_trim_from_body += 1  # Remove one more to make it a pair

            if num_to_trim_from_body > 0 and (1 + num_to_trim_from_body) <= current_len:
                del history_list[1: 1 + num_to_trim_from_body]
                log_info(
                    f"OpenAI history truncated. Removed {num_to_trim_from_body} messages after initial system prompt.")
                st.toast(f"Truncating OpenAI history.", icon="⚠️")
    elif api_type == "anthropic":
        if current_len > max_messages_to_keep:
            num_to_trim_from_body = current_len - max_messages_to_keep
            if num_to_trim_from_body % 2 != 0 and num_to_trim_from_body > 0:
                num_to_trim_from_body += 1
            if num_to_trim_from_body > 0 and num_to_trim_from_body <= current_len:
                del history_list[0: num_to_trim_from_body]
                log_info(f"Anthropic history truncated. Removed {num_to_trim_from_body} messages.")
                st.toast(f"Truncating Anthropic history.", icon="⚠️")
    return history_list


# --- Backend API Call Functions ---
def call_openai_model_backend(client, model_name_to_call, current_openai_history, max_output_allowance):
    log_info(
        f"Calling OpenAI (Model: {model_name_to_call}, Hist Len: {len(current_openai_history)}, Max Output Tokens: {max_output_allowance})")

    if not isinstance(current_openai_history, list) or not current_openai_history:
        err_msg = f"{ERROR_PREFIXES[1]}: OpenAI history invalid/empty for API call.";
        log_error(err_msg)
        return {"text": err_msg, "reasoning_summary": "Error", "reasoning_tokens": 0, "error_type": "script_error"}

    try:
        api_arguments = {
            "model": model_name_to_call,
            "input": current_openai_history,
            "reasoning": {"effort": "medium", "summary": "auto"},
            "max_output_tokens": max_output_allowance
        }
        log_info(
            f"OpenAI API call with args: { {k: v for k, v in api_arguments.items() if k != 'input'} } Hist Len: {len(api_arguments['input'])}")

        response = client.responses.create(**api_arguments)

        generated_output_text = ""
        reasoning_summary_text = "No reasoning summary found."
        reasoning_tokens_count = 0

        if response.output_text:
            generated_output_text = response.output_text.strip()

        if not generated_output_text:
            generated_output_text = PLACEHOLDER_NO_MODEL_CONTENT

        # *** CORRECTED REASONING SUMMARY EXTRACTION ***
        if response.output:  # response.output is a list of output items
            for output_item_obj in response.output:
                if output_item_obj.type == 'reasoning':
                    # output_item_obj is the ResponseReasoningItem object.
                    # It should directly have a 'summary' attribute which is a list.
                    if hasattr(output_item_obj, 'summary') and isinstance(output_item_obj.summary, list):
                        for summary_detail_obj in output_item_obj.summary:
                            # summary_detail_obj is an object from the list, expected to have 'type' and 'text'.
                            if hasattr(summary_detail_obj, 'type') and summary_detail_obj.type == 'summary_text' and \
                                    hasattr(summary_detail_obj, 'text') and summary_detail_obj.text:
                                reasoning_summary_text = summary_detail_obj.text
                                break  # Found the actual summary text
                    if reasoning_summary_text != "No reasoning summary found.":
                        break  # Stop searching other output_item_obj once summary is found
        # *********************************************

        if response.usage and response.usage.output_tokens_details and response.usage.output_tokens_details.reasoning_tokens is not None:
            reasoning_tokens_count = response.usage.output_tokens_details.reasoning_tokens

        if response.status == "incomplete" and response.incomplete_details and response.incomplete_details.reason == "max_output_tokens":
            generated_output_text += f"\n[TRUNCATED AT {max_output_allowance} TOKENS (OpenAI)]"
            if not generated_output_text.startswith(PLACEHOLDER_NO_MODEL_CONTENT):
                log_info("OpenAI response was truncated.")
            elif generated_output_text == PLACEHOLDER_NO_MODEL_CONTENT + f"\n[TRUNCATED AT {max_output_allowance} TOKENS (OpenAI)]":
                log_info("OpenAI response was truncated and had no preceding text content.")

        log_info(
            f"OpenAI OK. Reasoning Tk: {reasoning_tokens_count}. Finish Status: {response.status}. Output starts with: '{generated_output_text[:50]}...'")
        return {
            "text": generated_output_text,
            "reasoning_summary": reasoning_summary_text,
            "reasoning_tokens": reasoning_tokens_count,
            "error_type": None
        }

    except OpenAIError as e:
        err_msg = f"{ERROR_PREFIXES[0]}: OpenAI API Error: {e}";
        log_error(err_msg)
        if hasattr(e, 'body') and e.body: log_error(f"OpenAI API Error Body: {e.body}")
        return {"text": err_msg, "reasoning_summary": "API Error", "reasoning_tokens": 0, "error_type": "api_error"}
    except Exception as e:
        err_msg = f"{ERROR_PREFIXES[1]}: Unexpected error calling OpenAI: {e}\n{traceback.format_exc()}";
        log_error(err_msg)
        return {"text": err_msg, "reasoning_summary": "Script Error", "reasoning_tokens": 0,
                "error_type": "script_error"}


def call_anthropic_model_backend(client, model_name_to_call, system_prompt_text, current_anthropic_history,
                                 max_response_allowance):
    log_info(
        f"Calling Anthropic (Model: {model_name_to_call}, Hist Len: {len(current_anthropic_history)}, Max Resp Tokens: {max_response_allowance})")

    if not isinstance(current_anthropic_history, list):
        log_info(f"Anthropic history not list, was {type(current_anthropic_history)}. Correcting.")
        current_anthropic_history = []

    thinking_params_for_api = None
    if max_response_allowance >= ANTHROPIC_MIN_THINKING_BUDGET_REQUIRED + ANTHROPIC_MIN_RESPONSE_TOKENS_BUFFER:
        potential_budget = max(ANTHROPIC_MIN_THINKING_BUDGET_REQUIRED, math.ceil(max_response_allowance * 0.5))
        if potential_budget <= max_response_allowance - ANTHROPIC_MIN_RESPONSE_TOKENS_BUFFER:
            thinking_params_for_api = {"type": "enabled", "budget_tokens": potential_budget}
        else:
            adjusted_budget = max_response_allowance - ANTHROPIC_MIN_RESPONSE_TOKENS_BUFFER
            if adjusted_budget >= ANTHROPIC_MIN_THINKING_BUDGET_REQUIRED:
                thinking_params_for_api = {"type": "enabled", "budget_tokens": adjusted_budget}

    if thinking_params_for_api:
        log_info(f"Anthropic thinking enabled: budget {thinking_params_for_api['budget_tokens']}")
    else:
        log_info(f"Anthropic thinking disabled for this call (max_response_allowance: {max_response_allowance})")

    try:
        request_config = {
            "model": model_name_to_call,
            "system": system_prompt_text,
            "messages": current_anthropic_history,
            "max_tokens": max_response_allowance,
        }
        if thinking_params_for_api:
            request_config["thinking"] = thinking_params_for_api

        log_info(
            f"Anthropic API call with args: { {k: v for k, v in request_config.items() if k != 'messages' and k != 'system'} } Hist Len: {len(request_config['messages'])}")
        api_response = client.messages.create(**request_config)

        generated_text_parts = []
        thinking_content = "No thinking block captured."
        raw_response_blocks = []

        if api_response.content:
            raw_response_blocks = list(api_response.content)
            for block in api_response.content:
                if block.type == 'text' and block.text:
                    generated_text_parts.append(block.text.strip())  # Strip individual parts
                elif block.type == 'thinking' and hasattr(block, 'thinking') and block.thinking:
                    thinking_content = block.thinking if isinstance(block.thinking, str) else json.dumps(block.thinking)

        final_generated_text = " ".join(
            filter(None, generated_text_parts)).strip()  # Join non-empty, stripped parts and strip result
        if not final_generated_text:
            final_generated_text = PLACEHOLDER_NO_MODEL_CONTENT
            if thinking_content != "No thinking block captured.":
                final_generated_text += " (only thinking was present)"

        if api_response.stop_reason == "max_tokens":
            final_generated_text += f"\n[TRUNCATED AT {max_response_allowance} TOKENS (Anthropic)]"
            if not final_generated_text.startswith(PLACEHOLDER_NO_MODEL_CONTENT):
                log_info("Anthropic response was truncated.")
            elif final_generated_text == PLACEHOLDER_NO_MODEL_CONTENT + f"\n[TRUNCATED AT {max_response_allowance} TOKENS (Anthropic)]":  # check exact string
                log_info("Anthropic response was truncated and had no preceding text content.")

        log_info(
            f"Anthropic OK. Thinking: {thinking_content != 'No thinking block captured.'}. Stop: {api_response.stop_reason}. Output starts with: '{final_generated_text[:50]}...'")
        return {
            "text": final_generated_text,
            "thinking_content": thinking_content,
            "raw_blocks_for_history": raw_response_blocks,
            "error_type": None
        }

    except AnthropicAPIError as e:
        err_msg = f"{ERROR_PREFIXES[0]}: Anthropic API Error: {e}";
        log_error(err_msg)
        if hasattr(e, 'body') and e.body: log_error(f"Anthropic API Error Body: {e.body}")
        return {"text": err_msg, "thinking_content": "API Error", "raw_blocks_for_history": [],
                "error_type": "api_error"}
    except Exception as e:
        err_msg = f"{ERROR_PREFIXES[1]}: Unexpected error calling Anthropic: {e}\n{traceback.format_exc()}";
        log_error(err_msg)
        return {"text": err_msg, "thinking_content": "Script Error", "raw_blocks_for_history": [],
                "error_type": "script_error"}


# --- Streamlit UI (remains largely the same, ensure it uses the new session state var names) ---
st.set_page_config(layout="wide", page_title="Bot v Bot Debate Arena")
st.markdown("""
<style>
    .chat-bubble-row { display: flex; margin-bottom: 10px; width: 100%; clear: both; }
    .chat-bubble { padding: 10px 15px; border-radius: 18px; max-width: 75%; width: fit-content; word-wrap: break-word; border: 1px solid transparent; box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05); flex-shrink: 0; line-height: 1.5; }
    .openai-row { justify-content: flex-start; }
    .anthropic-row { justify-content: flex-end; }
    .openai-bubble { background-color: #DCF8C6; border-color: #cce8b5; color: black; text-align: left; margin-right: auto; margin-left: 5px; }
    .anthropic-bubble { background-color: #cfe2f3; border-color: #b8daff; color: black; text-align: left; margin-left: auto; margin-right: 5px; }
    .chat-bubble p { margin-bottom: 0.5em !important; color: inherit !important; text-align: inherit !important; }
    .chat-bubble p:last-child { margin-bottom: 0 !important; }
    .chat-bubble ul, .chat-bubble ol { margin-left: 20px; padding-left: 5px; color: inherit !important; text-align: inherit !important; margin-top: 0.5em; margin-bottom: 0.5em; }
    .chat-bubble li { margin-bottom: 0.25em; color: inherit !important; text-align: inherit !important;}
    .chat-bubble strong, .chat-bubble b { color: inherit !important; }
    .chat-bubble em, .chat-bubble i { color: inherit !important; }
    .chat-bubble h1, .chat-bubble h2, .chat-bubble h3, .chat-bubble h4, .chat-bubble h5, .chat-bubble h6 { color: inherit !important; margin-top: 0.5em; margin-bottom: 0.25em; text-align: inherit !important; }
    .chat-bubble code { white-space: pre-wrap !important; word-wrap: break-word !important; color: inherit !important; background-color: rgba(0,0,0,0.05); padding: 2px 4px; border-radius: 3px;}
    .chat-bubble pre code { padding: 0.5em !important; display: block; overflow-x: auto;}
    .chat-bubble br { line-height: 0.5em; content: ""; display: block; margin-bottom: 0.5em; }
</style>
""", unsafe_allow_html=True)
st.title("🤖 Bot vs Bot Debate Arena 🥊")


def initialize_session_state_variables():
    log_info("Initializing session state variables.")
    st.session_state.debate_status = "setup"
    st.session_state.debate_topic = "Chocolate milk vs. White milk: Which is superior?"
    st.session_state.openai_position = "Chocolate milk is the best milk."
    st.session_state.anthropic_position = "Plain white milk is the best milk."
    st.session_state.openai_history = []
    st.session_state.anthropic_history = []
    st.session_state.all_reasoning_data = []
    st.session_state.debate_transcript = []
    st.session_state.current_turn_index = 0
    st.session_state.openai_turns_taken = 0
    st.session_state.anthropic_turns_taken = 0
    st.session_state.openai_client = None
    st.session_state.anthropic_client = None
    st.session_state.system_error_message = None
    st.session_state.max_tokens_per_response_slider = 4000


if 'debate_status' not in st.session_state:
    initialize_session_state_variables()

with st.sidebar:
    st.header("Debate Setup")
    is_setup_phase = st.session_state.debate_status == "setup"

    topic_input = st.text_input("Debate Topic:", st.session_state.debate_topic, key="sidebar_topic",
                                disabled=not is_setup_phase)
    openai_pos_input = st.text_input("M1 (OpenAI/Green) Position:", st.session_state.openai_position,
                                     key="sidebar_pos1", disabled=not is_setup_phase)
    anthropic_pos_input = st.text_input("M2 (Anthropic/Blue) Position:", st.session_state.anthropic_position,
                                        key="sidebar_pos2", disabled=not is_setup_phase)

    st.session_state.max_tokens_per_response_slider = st.slider(
        "Max Output Tokens per Response:", min_value=500, max_value=30000,
        value=st.session_state.max_tokens_per_response_slider, step=100, key="token_slider_main",
        disabled=not is_setup_phase,
        help=(
            "Controls OpenAI's `max_output_tokens` (reasoning + visible output) & Anthropic's `max_tokens` (visible output). High values recommended for OpenAI reasoning.")
    )
    current_max_anthropic_resp = st.session_state.max_tokens_per_response_slider
    anthropic_budget_display_str = "Disabled (Max Tokens too low)"
    if current_max_anthropic_resp >= ANTHROPIC_MIN_THINKING_BUDGET_REQUIRED + ANTHROPIC_MIN_RESPONSE_TOKENS_BUFFER:
        potential_b = max(ANTHROPIC_MIN_THINKING_BUDGET_REQUIRED, math.ceil(current_max_anthropic_resp * 0.5))
        if potential_b <= current_max_anthropic_resp - ANTHROPIC_MIN_RESPONSE_TOKENS_BUFFER:
            anthropic_budget_display_str = f"~{potential_b}"
        else:
            adjusted_b = current_max_anthropic_resp - ANTHROPIC_MIN_RESPONSE_TOKENS_BUFFER
            if adjusted_b >= ANTHROPIC_MIN_THINKING_BUDGET_REQUIRED: anthropic_budget_display_str = f"~{adjusted_b} (adj.)"
    st.caption(f"Anthropic Est. Thinking Budget: {anthropic_budget_display_str} tokens")

    if is_setup_phase:
        if st.button("Start Debate!", type="primary", use_container_width=True):
            st.session_state.debate_topic = topic_input
            st.session_state.openai_position = openai_pos_input
            st.session_state.anthropic_position = anthropic_pos_input
            if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
                st.error("API Keys not found in .env.");
                log_error("API keys missing.")
            elif not topic_input or not openai_pos_input or not anthropic_pos_input:
                st.error("Please fill in all setup fields.");
                log_error("Setup fields incomplete.")
            else:
                try:
                    log_info("Initializing API clients...")
                    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY,
                                                            timeout=httpx.Timeout(60.0, connect=10.0))
                    st.session_state.anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY,
                                                                  timeout=httpx.Timeout(60.0, connect=10.0))
                    log_info("API Clients Initialized.")
                    st.session_state.openai_history = []
                    st.session_state.anthropic_history = []
                    st.session_state.all_reasoning_data = []
                    st.session_state.debate_transcript = []
                    st.session_state.current_turn_index = 0
                    st.session_state.openai_turns_taken = 0
                    st.session_state.anthropic_turns_taken = 0
                    st.session_state.system_error_message = None
                    for f_path in [REASONING_LOG_FILE, TRANSCRIPT_LOG_FILE, FINAL_TXT_TRANSCRIPT_FILE]:
                        if os.path.exists(f_path): os.remove(f_path)
                    st.session_state.debate_status = "running"
                    log_info("Debate status set to 'running'. Rerunning.")
                    st.rerun()
                except Exception as e:
                    err_msg = f"{ERROR_PREFIXES[3]}: Initialization failed: {e}";
                    log_error(err_msg);
                    st.error(err_msg)
    else:
        st.info(f"Topic: {st.session_state.debate_topic}")
        st.markdown(f"**M1 (OpenAI):** {st.session_state.openai_position}")
        st.markdown(f"**M2 (Anthropic):** {st.session_state.anthropic_position}")
        status_cap = st.session_state.debate_status.capitalize()
        if st.session_state.debate_status in ["running", "closing"]:
            current_model_turn = "OpenAI" if st.session_state.current_turn_index % 2 == 0 else "Anthropic"
            status_cap += f" - {current_model_turn}'s turn"
        st.markdown(f"Status: **{status_cap}**")
        total_turns_display = st.session_state.openai_turns_taken + st.session_state.anthropic_turns_taken
        st.markdown(
            f"Total Turns Taken: **{st.session_state.openai_turns_taken} (M1) + {st.session_state.anthropic_turns_taken} (M2) / {MAX_TURNS_PER_MODEL * 2} Total**")
        st.caption(f"Max Output Tokens/Resp: {st.session_state.max_tokens_per_response_slider}")
        if st.button("Reset Debate", use_container_width=True, key="sidebar_reset"):
            log_info("Reset Debate clicked.")
            current_topic = st.session_state.debate_topic;
            current_pos1 = st.session_state.openai_position
            current_pos2 = st.session_state.anthropic_position;
            current_slider = st.session_state.max_tokens_per_response_slider
            initialize_session_state_variables()
            st.session_state.debate_topic = current_topic;
            st.session_state.openai_position = current_pos1
            st.session_state.anthropic_position = current_pos2;
            st.session_state.max_tokens_per_response_slider = current_slider
            st.rerun()

status_placeholder = st.empty()
if st.session_state.get('system_error_message'):
    status_placeholder.error(st.session_state.system_error_message)

chat_container = st.container(height=600, border=False)
with chat_container:
    for i, msg_data in enumerate(st.session_state.get('debate_transcript', [])):
        model_speaker = msg_data.get("model_name");
        text_content = msg_data.get("text_content", "")
        content_html = text_content.replace('$', '$$').replace('\n', '<br>')
        if model_speaker == "OpenAI":
            st.markdown(
                f"<div class='chat-bubble-row openai-row'><div class='chat-bubble openai-bubble'>{content_html}</div></div>",
                unsafe_allow_html=True)
        elif model_speaker == "Anthropic":
            st.markdown(
                f"<div class='chat-bubble-row anthropic-row'><div class='chat-bubble anthropic-bubble'>{content_html}</div></div>",
                unsafe_allow_html=True)
        elif model_speaker == "System":
            st.info(text_content.replace('<br>', '\n'), icon="⚠️")

# --- Debate Logic Execution ---
if st.session_state.debate_status in ["running", "closing"] and not st.session_state.get('system_error_message'):
    turn_idx = st.session_state.current_turn_index
    is_openai_turn_flag = (turn_idx % 2 == 0)
    current_max_tokens = st.session_state.max_tokens_per_response_slider

    is_closing_args_phase = (st.session_state.openai_turns_taken >= MAX_TURNS_PER_MODEL and \
                             st.session_state.anthropic_turns_taken >= MAX_TURNS_PER_MODEL)

    # Check if both models have completed their closing arguments
    openai_closing_done = st.session_state.openai_turns_taken > MAX_TURNS_PER_MODEL
    anthropic_closing_done = st.session_state.anthropic_turns_taken > MAX_TURNS_PER_MODEL

    if st.session_state.debate_status == "closing" and openai_closing_done and anthropic_closing_done:
        st.session_state.debate_status = "finished"
        log_info("Both models completed closing arguments. Debate finished.")
        st.rerun()
    # Main turn logic only if debate is not yet finished due to closing args completion
    elif not (st.session_state.debate_status == "closing" and openai_closing_done and anthropic_closing_done):
        # --- OpenAI's Turn ---
        if is_openai_turn_flag and st.session_state.openai_turns_taken <= MAX_TURNS_PER_MODEL:
            with status_placeholder.status(f"M1 (OpenAI) Turn {st.session_state.openai_turns_taken + 1}...",
                                           expanded=True):
                st.write("Preparing OpenAI's turn...")
                st.session_state.openai_history = manage_context_window(st.session_state.openai_history, "openai")

                if turn_idx == 0:
                    prompt_text = BASE_SYSTEM_PROMPT_CORE.format(
                        debate_topic=st.session_state.debate_topic, your_position=st.session_state.openai_position,
                        opponent_position=st.session_state.anthropic_position, max_tokens_guideline=current_max_tokens
                    ) + "\n\nBegin the debate with your opening argument."
                    st.session_state.openai_history.append({"role": "user", "content": prompt_text})
                    log_info("Appended initial system-as-user prompt for OpenAI.")
                elif st.session_state.debate_transcript:
                    last_opponent_msg = next(
                        (m for m in reversed(st.session_state.debate_transcript) if m["model_name"] == "Anthropic"),
                        None)
                    if last_opponent_msg and not last_opponent_msg["text_content"].startswith(tuple(ERROR_PREFIXES)):
                        st.session_state.openai_history.append(
                            {"role": "user", "content": last_opponent_msg["text_content"]})
                    elif last_opponent_msg and last_opponent_msg["text_content"].startswith(tuple(ERROR_PREFIXES)):
                        st.session_state.system_error_message = f"{ERROR_PREFIXES[2]}: OpenAI cannot proceed due to opponent's API error.";
                        st.session_state.debate_status = "finished"
                    elif not last_opponent_msg and turn_idx > 0:
                        st.session_state.system_error_message = f"{ERROR_PREFIXES[2]}: OpenAI missing Anthropic's last message.";
                        st.session_state.debate_status = "finished"

                if is_closing_args_phase and st.session_state.openai_turns_taken == MAX_TURNS_PER_MODEL and not openai_closing_done:
                    st.session_state.openai_history.append(
                        {"role": "user", "content": "Provide your final closing argument."})
                    st.session_state.debate_status = "closing";
                    log_info("OpenAI to make closing argument.")

                if not st.session_state.system_error_message:
                    st.write("Calling OpenAI API...");
                    response_data = call_openai_model_backend(st.session_state.openai_client, OPENAI_MODEL,
                                                              st.session_state.openai_history, current_max_tokens)
                    add_to_reasoning_log(st.session_state.openai_turns_taken + 1, "OpenAI",
                                         {"summary": response_data["reasoning_summary"],
                                          "tokens_used": response_data["reasoning_tokens"]})
                    if response_data["error_type"]:
                        st.session_state.system_error_message = response_data["text"];
                        st.session_state.debate_status = "finished"
                    else:
                        st.session_state.debate_transcript.append(
                            {"turn_index": turn_idx, "model_name": "OpenAI", "text_content": response_data["text"],
                             "timestamp": get_timestamp()})
                        st.session_state.openai_history.append({"role": "assistant", "content": response_data["text"]})
                        st.session_state.openai_turns_taken += 1

                save_log(st.session_state.all_reasoning_data, REASONING_LOG_FILE);
                save_log(st.session_state.debate_transcript, TRANSCRIPT_LOG_FILE)
                if not st.session_state.system_error_message: st.session_state.current_turn_index += 1
                st.rerun()

        # --- Anthropic's Turn ---
        elif not is_openai_turn_flag and st.session_state.anthropic_turns_taken <= MAX_TURNS_PER_MODEL:
            with status_placeholder.status(f"M2 (Anthropic) Turn {st.session_state.anthropic_turns_taken + 1}...",
                                           expanded=True):
                st.write("Preparing Anthropic's turn...")
                st.session_state.anthropic_history = manage_context_window(st.session_state.anthropic_history,
                                                                           "anthropic")
                last_opponent_msg = next(
                    (m for m in reversed(st.session_state.debate_transcript) if m["model_name"] == "OpenAI"), None)

                if not last_opponent_msg and turn_idx > 0:  # Anthropic always needs a prior OpenAI message
                    st.session_state.system_error_message = f"{ERROR_PREFIXES[2]}: Anthropic missing OpenAI's last message.";
                    st.session_state.debate_status = "finished"
                elif last_opponent_msg and last_opponent_msg["text_content"].startswith(tuple(ERROR_PREFIXES)):
                    st.session_state.system_error_message = f"{ERROR_PREFIXES[2]}: Anthropic cannot proceed due to opponent's API error.";
                    st.session_state.debate_status = "finished"
                elif last_opponent_msg:
                    st.session_state.anthropic_history.append(
                        {"role": "user", "content": [{"type": "text", "text": last_opponent_msg["text_content"]}]})

                anthropic_sys_prompt = BASE_SYSTEM_PROMPT_CORE.format(debate_topic=st.session_state.debate_topic,
                                                                      your_position=st.session_state.anthropic_position,
                                                                      opponent_position=st.session_state.openai_position,
                                                                      max_tokens_guideline=current_max_tokens)
                if turn_idx == 1: anthropic_sys_prompt += "\n\nYour opponent has made their opening statement. Respond now."

                if is_closing_args_phase and st.session_state.anthropic_turns_taken == MAX_TURNS_PER_MODEL and not anthropic_closing_done:
                    st.session_state.anthropic_history.append(
                        {"role": "user", "content": [{"type": "text", "text": "Provide your final closing argument."}]})
                    st.session_state.debate_status = "closing";
                    log_info("Anthropic to make closing argument.")

                if not st.session_state.system_error_message:
                    st.write("Calling Anthropic API...");
                    response_data = call_anthropic_model_backend(st.session_state.anthropic_client, ANTHROPIC_MODEL,
                                                                 anthropic_sys_prompt,
                                                                 st.session_state.anthropic_history, current_max_tokens)
                    add_to_reasoning_log(st.session_state.anthropic_turns_taken + 1, "Anthropic",
                                         {"thinking_content": response_data["thinking_content"]})
                    if response_data["error_type"]:
                        st.session_state.system_error_message = response_data["text"];
                        st.session_state.debate_status = "finished"
                    else:
                        st.session_state.debate_transcript.append(
                            {"turn_index": turn_idx, "model_name": "Anthropic", "text_content": response_data["text"],
                             "timestamp": get_timestamp()})
                        st.session_state.anthropic_history.append({"role": "assistant", "content": response_data.get(
                            "raw_blocks_for_history", [{"type": "text", "text": response_data["text"]}])})
                        st.session_state.anthropic_turns_taken += 1

                save_log(st.session_state.all_reasoning_data, REASONING_LOG_FILE);
                save_log(st.session_state.debate_transcript, TRANSCRIPT_LOG_FILE)
                if not st.session_state.system_error_message: st.session_state.current_turn_index += 1
                st.rerun()

        # If one model has finished its closing arg and the other hasn't had its turn yet for closing
        elif st.session_state.debate_status == "closing":
            if (
                    is_openai_turn_flag and openai_closing_done and not anthropic_closing_done and st.session_state.anthropic_turns_taken <= MAX_TURNS_PER_MODEL) or \
                    (
                            not is_openai_turn_flag and anthropic_closing_done and not openai_closing_done and st.session_state.openai_turns_taken <= MAX_TURNS_PER_MODEL):
                log_info("One model finished closing, other still needs to. Incrementing turn for the other model.")
                # This case might not be strictly necessary if the main turn logic handles <= MAX_TURNS_PER_MODEL correctly for closing.
                # st.session_state.current_turn_index += 1 # Ensure the other model gets its turn
                # st.rerun() # Rerun to trigger the other model's turn if it hasn't taken its closing turn.
                pass  # The main logic should handle this by letting the other model take its turn.
            elif openai_closing_done and anthropic_closing_done:  # Should be caught by the top check, but as a safeguard
                st.session_state.debate_status = "finished"
                log_info("Safeguard: Both models completed closing arguments. Debate finished.")
                st.rerun()

# --- Finished State - Download Buttons (remains the same) ---
if st.session_state.debate_status == "finished":
    if not st.session_state.get('system_error_message'):
        status_placeholder.success("Debate Finished!", icon="🎉")

    final_transcript_str = ""
    try:
        transcript_data_for_file = st.session_state.get('debate_transcript', [])
        topic_disp = st.session_state.get('debate_topic', 'N/A')
        pos1_disp = st.session_state.get('openai_position', 'N/A')
        pos2_disp = st.session_state.get('anthropic_position', 'N/A')
        output_lines = [f"Debate Topic: {topic_disp}", f"Model 1 (OpenAI - {OPENAI_MODEL}): {pos1_disp}",
                        f"Model 2 (Anthropic - {ANTHROPIC_MODEL}): {pos2_disp}", "=" * 40 + "\n"]

        for entry in transcript_data_for_file:
            model_name_disp = entry.get("model_name", "Unknown")
            text_content_disp = entry.get("text_content", "[No content logged]")
            timestamp_disp = entry.get("timestamp", "")
            turn_idx_entry = entry.get("turn_index", 0)

            is_closing_arg_entry = False
            # A turn is closing if it's the (MAX_TURNS_PER_MODEL + 1)-th turn for that model
            if model_name_disp == "OpenAI" and (turn_idx_entry // 2) == MAX_TURNS_PER_MODEL: is_closing_arg_entry = True
            if model_name_disp == "Anthropic" and (
                    turn_idx_entry // 2) == MAX_TURNS_PER_MODEL: is_closing_arg_entry = True

            turn_label = f"Turn {(turn_idx_entry // 2) + 1}"
            if is_closing_arg_entry: turn_label = "Closing Argument"

            output_lines.append(f"--- {turn_label}: {model_name_disp} ({timestamp_disp}) ---")
            output_lines.append(f"{text_content_disp}\n")

        if st.session_state.get("system_error_message"):
            output_lines.append("=" * 20 + " DEBATE ENDED DUE TO ERROR " + "=" * 20 + "\n")
            output_lines.append(f"System Error: {st.session_state.system_error_message}\n")

        final_transcript_str = "\n".join(output_lines)
        with open(FINAL_TXT_TRANSCRIPT_FILE, 'w', encoding='utf-8') as f:
            f.write(final_transcript_str)
        log_info("Final TXT transcript generated.")
    except Exception as e:
        log_error(f"Error generating final .txt transcript: {e}");
        st.error(f"Error generating final .txt transcript: {e}")

    st.header("Downloads")
    dl_cols = st.columns(2)
    with dl_cols[0]:
        if final_transcript_str:
            st.download_button(label="Download Full Transcript (.TXT)", data=final_transcript_str.encode('utf-8'),
                               file_name=FINAL_TXT_TRANSCRIPT_FILE, mime="text/plain", use_container_width=True)
        else:
            st.warning(f"{FINAL_TXT_TRANSCRIPT_FILE} could not be generated.")
        try:
            if os.path.exists(TRANSCRIPT_LOG_FILE):
                with open(TRANSCRIPT_LOG_FILE, "rb") as fp:
                    st.download_button(label="Transcript Log (JSON)", data=fp,
                                       file_name=os.path.basename(TRANSCRIPT_LOG_FILE), mime="application/json",
                                       use_container_width=True)
            else:
                st.caption(f"{os.path.basename(TRANSCRIPT_LOG_FILE)} not found.")
        except Exception as e:
            st.error(f"Error for {TRANSCRIPT_LOG_FILE} DL: {e}")
    with dl_cols[1]:
        try:
            if os.path.exists(REASONING_LOG_FILE):
                with open(REASONING_LOG_FILE, "rb") as fp:
                    st.download_button(label="Reasoning/Thinking Log (JSON)", data=fp,
                                       file_name=os.path.basename(REASONING_LOG_FILE), mime="application/json",
                                       use_container_width=True)
            else:
                st.caption(f"{os.path.basename(REASONING_LOG_FILE)} not found.")
        except Exception as e:
            st.error(f"Error for {REASONING_LOG_FILE} DL: {e}")

log_info(f"Streamlit script execution finished. Status: {st.session_state.get('debate_status', 'N/A')}")