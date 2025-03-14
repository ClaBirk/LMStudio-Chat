import sys
import json
import time
import re
import requests
from PyQt5 import QtWidgets, QtGui, QtCore
import markdown  # For markdown support
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# -------------------------------------------------------
# Global Variables
# -------------------------------------------------------
conversation_history = []  # Each element is a complete turn (including prefix)
selected_model = ""  # Will be populated when models are fetched
available_models = []  # List of model IDs
available_models_display = []  # List of display names for the UI
display_to_model_map = {}  # Maps from display names to actual model IDs
server_ip = "192.168.3.1"  # Default server IP
api_port = "1234"  # Default LM Studio API port
remote_port = "5051"  # Default remote server port for model unloading
lmstudio_server = f"http://{server_ip}:{api_port}"  # LM Studio API endpoint
remote_server = f"http://{server_ip}:{remote_port}"  # Remote server for model unloading
use_markdown = True  # Default to markdown mode
model_loading = False  # Flag to track if a model is currently loading
model_ready = False  # Flag to track if the currently selected model is ready
model_to_context_map = {}  # Global mapping of model names to context windows
model_to_type_map = {}  # Global mapping of model names to types (llm, vlm)
model_to_quant_map = {}  # Global mapping of model names to quantization types

# Heading size configuration - single tunable variable (values between 0.3-1.0 work well)
heading_size_scale = 0.65  # Base scale for headings - smaller values = smaller headings

# Memory usage thresholds for conversation history
MAX_CONVERSATION_SIZE = 500000  # 500KB threshold
TRIM_TARGET_SIZE = 300000  # 300KB target after trimming
LARGE_CONTENT_THRESHOLD = 100000  # 100KB threshold for disabling markdown

# -------------------------------------------------------
# 1) Model Fetching - Updated for LM Studio API
# -------------------------------------------------------
def fetch_models():
    global available_models, available_models_display, selected_model, lmstudio_server, model_to_type_map, model_to_quant_map, display_to_model_map
    try:
        # First try the LM Studio native API endpoint to get more model details
        # Properly construct the native API URL
        base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
        native_api_url = f"{base_url}/api/v0/models"
        print(f"DEBUG: Fetching models from native API: {native_api_url}")
        
        response = requests.get(native_api_url, timeout=10.0)  # Increased timeout
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                # Filter for LLM and VLM models and save context lengths and quantization
                models_list = []              # Actual model IDs
                display_models_list = []      # Display names for UI
                model_context_map = {}        # Context window info
                model_type_map = {}           # Model type info
                model_quant_map = {}          # Quantization info
                display_to_model_map.clear()  # Clear previous mappings
                
                for m in data["data"]:
                    # Include both LLM and VLM models
                    if m.get("type") in ["llm", "vlm"]:
                        model_id = m["id"]
                        # Save model type
                        model_type_map[model_id] = m.get("type")
                        # Save quantization information
                        if "quantization" in m:
                            model_quant_map[model_id] = m["quantization"]
                        # Save context length if available
                        if "max_context_length" in m:
                            context_length = m["max_context_length"]
                            model_context_map[model_id] = context_length
                            print(f"DEBUG: Model '{model_id}' ({m.get('type')}) has context window: {context_length}, quantization: {m.get('quantization', 'unknown')}")
                        else:
                            print(f"DEBUG: Model '{model_id}' ({m.get('type')}) has no context window information")
                        
                        # Create display name with model type, context, and quantization
                        type_indicator = "[VLM] " if m.get("type") == "vlm" else ""
                        quant = m.get("quantization", "")
                        quant_info = f" | {quant}" if quant else ""
                        
                        if "max_context_length" in m:
                            ctx = m["max_context_length"]
                            ctx_formatted = f"{ctx//1000}K" if ctx >= 1000 else str(ctx)
                            display_name = f"{type_indicator}{model_id} (ctx: {ctx_formatted}{quant_info})"
                        else:
                            display_name = f"{type_indicator}{model_id}{quant_info}"
                        
                        # Add to our lists
                        models_list.append(model_id)
                        display_models_list.append(display_name)
                        
                        # Map display name to model ID
                        display_to_model_map[display_name] = model_id
                        print(f"DEBUG: Mapped display name '{display_name}' to model ID '{model_id}'")
                
                # Save maps to global variables
                global model_to_context_map
                model_to_context_map = model_context_map
                model_to_type_map = model_type_map
                model_to_quant_map = model_quant_map
                
                # Sort models by context length (largest first), then alphabetically
                def sort_key(idx):
                    model_id = models_list[idx]
                    ctx = model_context_map.get(model_id, 0)
                    return (-ctx, model_id.lower())  # Negative context for descending order
                
                # Sort both lists together based on sort keys
                indices = list(range(len(models_list)))
                indices.sort(key=sort_key)
                
                # Reorder both lists using the sorted indices
                sorted_models = [models_list[i] for i in indices]
                sorted_display_models = [display_models_list[i] for i in indices]
                
                # Store both model IDs and display names in the global variables
                available_models = sorted_models
                available_models_display = sorted_display_models
                
                print(f"DEBUG: Populated available_models with {len(available_models)} models")
                print(f"DEBUG: Populated available_models_display with {len(available_models_display)} display names")
            else:
                available_models = []
                available_models_display = []
            
            # Handle empty model list or add selected model
            if not available_models:
                if selected_model:
                    available_models = [selected_model]
                    display_name = selected_model
                    available_models_display = [display_name]
                    display_to_model_map[display_name] = selected_model
                else:
                    available_models = ["<No models available>"]
                    available_models_display = ["<No models available>"]
                    display_to_model_map["<No models available>"] = "<No models available>"
            elif selected_model and selected_model not in available_models:
                display_name = selected_model
                available_models.insert(0, selected_model)
                available_models_display.insert(0, display_name)
                display_to_model_map[display_name] = selected_model
                
            # If we have models but no selection yet, select the first one
            if available_models and not available_models[0].startswith("<") and not selected_model:
                selected_model = available_models[0]
            
            # Return both model IDs and display names
            return True, available_models, available_models_display
        else:
            # Fall back to OpenAI-compatible endpoint
            openai_api_url = f"{base_url}/v1/models"
            print(f"DEBUG: Native API failed with {response.status_code}, trying OpenAI-compatible endpoint: {openai_api_url}")
            response = requests.get(openai_api_url, timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    # Extract model IDs from LM Studio response
                    model_ids = [m["id"] for m in data["data"]]
                    model_ids.sort()
                    available_models = model_ids
                    available_models_display = model_ids  # No context info in this case
                    
                    # Simple 1:1 mapping in this case
                    display_to_model_map.clear()
                    for model_id in model_ids:
                        display_to_model_map[model_id] = model_id
                else:
                    available_models = []
                    available_models_display = []
                
                if not available_models:
                    if selected_model:
                        available_models = [selected_model]
                        available_models_display = [selected_model]
                        display_to_model_map[selected_model] = selected_model
                    else:
                        available_models = ["<No models available>"]
                        available_models_display = ["<No models available>"]
                        display_to_model_map["<No models available>"] = "<No models available>"
                elif selected_model and selected_model not in available_models:
                    available_models.insert(0, selected_model)
                    available_models_display.insert(0, selected_model)
                    display_to_model_map[selected_model] = selected_model
                
                # If we have models but no selection yet, select the first one
                if available_models and not available_models[0].startswith("<") and not selected_model:
                    selected_model = available_models[0]
                
                return True, available_models, available_models_display
            else:
                print(f"DEBUG: Server returned error {response.status_code}: {response.text}")
                if selected_model:
                    available_models = [selected_model]
                    available_models_display = [selected_model]
                    display_to_model_map[selected_model] = selected_model
                else:
                    available_models = ["<Connection error>"]
                    available_models_display = ["<Connection error>"]
                    display_to_model_map["<Connection error>"] = "<Connection error>"
                return False, available_models, available_models_display
    except requests.exceptions.RequestException as e:
        print(f"DEBUG: Exception while fetching models: {e}")
        if selected_model:
            available_models = [selected_model]
            available_models_display = [selected_model]
            display_to_model_map[selected_model] = selected_model
        else:
            available_models = ["<Connection error>"]
            available_models_display = ["<Connection error>"]
            display_to_model_map["<Connection error>"] = "<Connection error>"
        return False, available_models, available_models_display

# -------------------------------------------------------
# Context Window Size Detection - New for LM Studio
# -------------------------------------------------------

def get_actual_context_length(model_id):
    """
    Get the actual context length configured for a loaded model (not just the max)
    from the models endpoint on port 5051.
    """
    global remote_server
    
    try:
        # Get the base URL from the remote server but use the models endpoint
        base_url = remote_server.rsplit(':', 1)[0]  # Remove port
        url = f"{base_url}:5051/models"
        print(f"DEBUG: Fetching actual context length from: {url}")
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            print(f"DEBUG: Loaded models data: {models_data}")
            
            # Find our model in the loaded models
            for model in models_data:
                # Check different identifiers since the API can use different naming
                model_identifiers = [
                    model.get("modelKey"),
                    model.get("identifier"),
                    model.get("displayName"),
                    model.get("path").split('/')[-1].split('.')[0] if model.get("path") else None
                ]
                
                # Try to find a match with our model_id
                if any(model_id == id for id in model_identifiers if id):
                    # Get the actual context length if available
                    if "contextLength" in model:
                        actual_context = model["contextLength"]
                        print(f"DEBUG: Found actual context length for '{model_id}': {actual_context}")
                        
                        # Update our global context map
                        global model_to_context_map
                        model_to_context_map[model_id] = actual_context
                        
                        return actual_context
                    elif "maxContextLength" in model:
                        # Fall back to max context length
                        max_context = model["maxContextLength"]
                        print(f"DEBUG: Found max context length for '{model_id}': {max_context}")
                        return max_context
            
            print(f"DEBUG: Model '{model_id}' not found in loaded models data")
    except Exception as e:
        print(f"DEBUG: Error getting actual context length: {e}")
    
    # If we couldn't get the actual context length, return 0 to indicate failure
    # We'll fall back to the previously detected max context length
    return 0

def get_context_window_size(model_id):
    """
    Get the context window size directly from the LM Studio API's model information.
    Updates the global context map as well.
    """
    global model_to_context_map
    
    print(f"DEBUG: Getting context window size for model '{model_id}'")
    
    # Check if we already have this information cached
    if model_id in model_to_context_map:
        cached_size = model_to_context_map[model_id]
        print(f"DEBUG: Using cached context length for '{model_id}': {cached_size}")
        return cached_size
    
    # Get the model info from the API v0 endpoint
    try:
        # Properly construct the native API URL
        base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
        native_api_url = f"{base_url}/api/v0/models"
        print(f"DEBUG: Fetching models from native API: {native_api_url}")
        
        response = requests.get(native_api_url, timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            if "data" in models_data:
                models = models_data["data"]
                
                # Update our global context map with all models
                for model in models:
                    if model["type"] in ["llm", "vlm"] and "max_context_length" in model:
                        model_to_context_map[model["id"]] = model["max_context_length"]
                
                # Find the matching model by ID
                if model_id in model_to_context_map:
                    context_length = model_to_context_map[model_id]
                    print(f"DEBUG: Found context length in model data: {context_length}")
                    return context_length
                
                print(f"DEBUG: Model '{model_id}' not found in API response or missing context length.")
        else:
            print(f"DEBUG: API returned status code {response.status_code}")
    except Exception as e:
        print(f"DEBUG: Error getting model info: {e}")
    
    # Fallback method: Use the OpenAI-compatible API error message
    print("DEBUG: Falling back to error message method...")
    try:
        # Send an intentionally too-large prompt to trigger an error that includes context size
        long_prompt = "a" * 100000  # 100,000 characters
        
        payload = {
            "model": model_id,
            "prompt": long_prompt,
            "max_tokens": 1,
            "temperature": 0.0
        }
        
        base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
        openai_api_url = f"{base_url}/v1/completions"
        print(f"DEBUG: Sending test prompt to: {openai_api_url}")
        
        response = requests.post(openai_api_url, json=payload, timeout=5)
        
        # If no error was thrown (unlikely), fall back to default
        if response.status_code == 200:
            print("DEBUG: Large prompt didn't trigger an error. Using default context size.")
            return 4096
        
        # Parse the error message
        error_message = response.text
        print(f"DEBUG: Error message: {error_message}")
        
        # Try different regex patterns to match various error formats
        patterns = [
            r"context length of only (\d+) tokens",
            r"maximum context length is (\d+) tokens",
            r"context window of (\d+) tokens",
            r"context size: (\d+)",
            r"max tokens: (\d+)",
            r"maximum context length \((\d+)\)",
            r"model's maximum context length \((\d+)\)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                context_window = int(match.group(1))
                print(f"DEBUG: Detected context window size: {context_window} tokens")
                # Update our context map
                model_to_context_map[model_id] = context_window
                return context_window
    except Exception as e:
        print(f"DEBUG: Error in fallback method: {e}")
    
    # If all else fails, use default value
    print("DEBUG: Could not determine context window size. Using default value of 4096 tokens.")
    return 4096

# -------------------------------------------------------
# Model Unloading - New for LM Studio
# -------------------------------------------------------
def unload_all_models():
    """
    Unload all models using the remote unload endpoint.
    Returns True if successful, False otherwise.
    """
    try:
        print("DEBUG: Unloading all models via remote endpoint...")
        url = f"{remote_server}/unload_all"
        
        # Set a reasonable timeout (10 seconds)
        response = requests.post(url, timeout=10)
        result = response.json()
        
        if result.get("status") == "success":
            print("DEBUG: Successfully unloaded all models.")
            return True
        else:
            print(f"DEBUG: Failed to unload models. Status: {result.get('status')}")
            print(f"DEBUG: Message: {result.get('message', 'No message')}")
            return False
    except requests.exceptions.Timeout:
        print("DEBUG: Timeout waiting for models to unload.")
        return False
    except Exception as e:
        print(f"DEBUG: Error unloading models: {e}")
        return False

# -------------------------------------------------------
# Custom Combo Box with Colored Items for Loaded Model
# -------------------------------------------------------
class ModelComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(ModelComboBox, self).__init__(parent)
        self.loaded_model = ""
        
    def setLoadedModel(self, model_name):
        """Set which model is currently loaded"""
        self.loaded_model = model_name
        self.update()
        
    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.Text))
        
        # Draw the combobox frame, focusrect and selected etc.
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt)
        
        # Draw the current item
        painter.drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt)
        
    def view(self):
        """Get the view of the combobox"""
        return super().view()
        
    def setView(self, view):
        """Set the view of the combobox"""
        view.setItemDelegate(ModelItemDelegate(self.loaded_model, self))
        super().setView(view)
        
class ModelItemDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, loaded_model, parent=None):
        super(ModelItemDelegate, self).__init__(parent)
        self.loaded_model = loaded_model
        self.parent_combo = parent
        
    def setLoadedModel(self, model_name):
        """Update which model is considered loaded"""
        self.loaded_model = model_name
        # Force update of the view
        if self.parent_combo and self.parent_combo.view():
            self.parent_combo.view().update()
        
    def paint(self, painter, option, index):
        # Get the text for this item
        item_text = index.data(QtCore.Qt.DisplayRole)
        
        # Check if this is the currently loaded model (strip any (ctx:) part)
        item_model = item_text.split(" (ctx:")[0] if " (ctx:" in item_text else item_text
        # Remove [VLM] prefix if present for comparison
        if item_model.startswith("[VLM] "):
            item_model = item_model[6:]
        is_loaded_model = (item_model == self.loaded_model)
        
        # Custom colors
        normal_text_color = QtGui.QColor("#FFFFFF")  # White text
        loaded_text_color = QtGui.QColor("#FF9500")  # Orange text for loaded model
        hover_bg_color = QtGui.QColor("#4A4A4A")     # Darker grey for hover/selection (more contrast)
        
        # Selected item has a different background
        if option.state & QtWidgets.QStyle.State_Selected:
            # Use custom dark grey instead of the default highlight color
            painter.fillRect(option.rect, hover_bg_color)
            # Keep text color the same when hovering
            text_color = loaded_text_color if is_loaded_model else normal_text_color
        else:
            # Use default background
            painter.fillRect(option.rect, option.palette.base())
            # Choose text color based on whether this is the loaded model
            text_color = loaded_text_color if is_loaded_model else normal_text_color
        
        # Set the text color
        painter.setPen(text_color)
        
        # Draw the text
        painter.drawText(option.rect.adjusted(5, 0, -5, 0), QtCore.Qt.AlignVCenter, item_text)

class CodeExtension(markdown.Extension):
    def extendMarkdown(self, md):
        md.registerExtension(self)
        md.preprocessors.register(CodeProcessor(md), 'highlight_code', 175)

class CodeProcessor(markdown.preprocessors.Preprocessor):
    def run(self, lines):
        new_lines = []
        in_code_block = False
        code_block_lines = []
        language = None
        block_id = 0
        
        for line in lines:
            # Check for code block start/end with language identifier
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    language = line.strip()[3:].strip() or 'text'
                    code_block_lines = []
                else:
                    in_code_block = False
                    code = '\n'.join(code_block_lines)
                    # Generate a unique ID for this code block
                    block_id += 1
                    
                    # Safely format the code for insertion into HTML
                    try:
                        if language.lower() in ['python', 'py']:
                            lexer = get_lexer_by_name('python', stripall=True)
                        else:
                            try:
                                lexer = get_lexer_by_name(language, stripall=True)
                            except:
                                lexer = get_lexer_by_name('text', stripall=True)
                        
                        # Format the code with syntax highlighting
                        formatter = HtmlFormatter(
                            style='monokai',
                            noclasses=True, 
                            nobackground=True,
                            linenos=False
                        )
                        highlighted_code = highlight(code, lexer, formatter)
                    except Exception as e:
                        print(f"DEBUG: Syntax highlighting error: {e}")
                        # Fallback to plain text with HTML escaping
                        highlighted_code = f"<pre style='margin: 0; color: #f8f8f2;'>{code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</pre>"
                    
                    # Create a code block without copy button
                    code_block_html = f"""<div class="code-block" style="margin: 0; padding: 0.5em; background-color: #000000; border-radius: 3px; font-family: Consolas, monospace; font-size: 0.95em; line-height: 1.2; position: relative; overflow-x: auto;">
<div style="margin: 0; background-color: #000000; white-space: pre-wrap; word-wrap: break-word;">{highlighted_code}</div>
</div>"""
                    
                    new_lines.append(code_block_html)
                    language = None
            elif in_code_block:
                code_block_lines.append(line)
            else:
                new_lines.append(line)
                
        return new_lines

# -------------------------------------------------------
# 2) Newline Collapsing
# -------------------------------------------------------
def collapse_newlines(text: str) -> str:
    """
    Replace 2+ consecutive newlines with a single newline
    to avoid huge vertical gaps.
    """
    return re.sub(r'\n{2,}', '\n', text)

# -------------------------------------------------------
# 3) Thinking Token Highlight
# -------------------------------------------------------
def highlight_thinking_tokens(text: str) -> str:
    """
    Replace <think> and </think> (and <thinking>, </thinking>) with
    a span so they display as literal text in bright yellow.
    """
    text = text.replace("<thinking>", '<span style="color: #FFD700;">&lt;thinking&gt;</span>')
    text = text.replace("</thinking>", '<span style="color: #FFD700;">&lt;/thinking&gt;</span>')
    text = text.replace("<think>", '<span style="color: #FFD700;">&lt;think&gt;</span>')
    text = text.replace("</think>", '<span style="color: #FFD700;">&lt;/think&gt;</span>')
    return text

# -------------------------------------------------------
# 3.5) Preprocess Markdown Headings
# -------------------------------------------------------
def preprocess_markdown_headings(text: str) -> str:
    """
    Preprocess markdown text to ensure all headings are at maximum h3 level.
    Converts top-level (#) and second-level (##) headings to third-level (###).
    """
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # If line starts with # or ## but not ###, ####, etc.
        if line.startswith('# '):
            line = '### ' + line[2:]
        elif line.startswith('## '):
            line = '### ' + line[3:]
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

# -------------------------------------------------------
# 4) Build HTML for Chat History - Modified for better Markdown
# -------------------------------------------------------
def build_html_chat_history(history=None):
    try:
        global use_markdown, conversation_history, heading_size_scale
        lines_html = []
        
        # Use provided history or global history
        if history is None:
            history = conversation_history
        
        # Calculate heading sizes based on scale - make h1 and h2 the same as h3
        base_size = heading_size_scale
        h3_size = base_size * 0.94  # This is our maximum heading size now
        h4_size = base_size * 0.91
        h5_size = base_size * 0.88
        h6_size = base_size * 0.85
        
        # CSS for better markdown styling
        markdown_css = f"""
        <style>
            .markdown-content {{
                line-height: 1.2;
                margin: 0;
                padding: 0;
            }}
            .markdown-content p {{
                margin: 0.3em 0;
            }}
            
            /* Force all headings to use h3 size */
            .markdown-content h1 {{
                font-size: {h3_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            
            /* Special color for User/AI headings */
            .markdown-content h1.user-heading, 
            .markdown-content h1.ai-heading {{
                color: orange !important;
                display: inline-block !important;
            }}
            
            /* Regular content headings */
            .markdown-content h2 {{
                font-size: {h3_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            
            .markdown-content h3 {{
                font-size: {h3_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            
            .markdown-content h4 {{
                font-size: {h4_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            
            .markdown-content h5 {{
                font-size: {h5_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            
            .markdown-content h6 {{
                font-size: {h6_size}em !important;
                font-weight: bold !important;
                margin: 0.3em 0 0.2em 0 !important;
                color: #e6e6e6 !important;
                line-height: 1.2 !important;
            }}
            
            /* Non-markdown User/AI labels should match h3 */
            .user-label, .ai-label {{
                color: orange !important;
                font-weight: bold !important;
                font-size: {h3_size}em !important;
            }}
            
            .markdown-content ul, .markdown-content ol {{
                margin: 0.3em 0 0.3em 1.5em;
                padding: 0;
            }}
            .markdown-content li {{
                margin: 0.1em 0;
            }}
            .markdown-content code {{
                background-color: #000000;
                color: #e6e6e6;
                border-radius: 3px;
                padding: 0.1em 0.3em;
                font-family: Consolas, monospace;
                font-size: 0.95em;
            }}
            .markdown-content pre {{
                margin: 0;
                padding: 0.5em;
                background-color: #000000;
                border-radius: 3px;
                font-family: Consolas, monospace;
                font-size: 0.95em;
                line-height: 1.2;
                overflow-x: auto;
            }}
            .markdown-content blockquote {{
                margin: 0.5em 0;
                padding: 0.3em 0.5em;
                border-left: 3px solid #537BA2;
                background-color: #323232;
            }}
            .markdown-content table {{
                border-collapse: collapse;
                margin: 0.5em 0;
                font-size: 0.95em;
            }}
            .markdown-content th, .markdown-content td {{
                padding: 0.3em 0.6em;
                border: 1px solid #444;
            }}
            .markdown-content th {{
                background-color: #373737;
            }}
            .markdown-content hr {{
                border: none;
                border-top: 1px solid #444;
                margin: 0.5em 0;
            }}
            /* Improved code block styling */
            .code-block {{
                position: relative;
            }}
            .code-block button {{
                opacity: 0.7;
                transition: opacity 0.2s ease;
            }}
            .code-block:hover button {{
                opacity: 1;
            }}
        </style>
        """
        
        chat_html = markdown_css
        
        for line in history:
            line = collapse_newlines(line)
            if line.startswith("User:\n"):
                content = line[len("User:\n"):]
                content = highlight_thinking_tokens(content)
                
                if use_markdown:
                    try:
                        # Create custom HTML for the User heading with orange styling as H3
                        user_heading = f"<h3 class='user-heading' style='color:#FF9500 !important;'>User:</h3>"
                        
                        # Append the content after the custom heading
                        # Apply our custom markdown extension along with standard extensions
                        md = markdown.Markdown(extensions=[
                            'fenced_code', 'tables', 'nl2br', CodeExtension()
                        ])
                        processed_content = md.convert(content)
                        content_html = f'<div class="markdown-content">{user_heading}{processed_content}</div>'
                    except Exception as e:
                        print(f"DEBUG: Markdown parsing error: {e}")
                        content_html = f'<div style="margin-top: 0.5em;"><span class="user-label" style="color:#FF9500;">User:</span><br/>{content.replace("<br/>", "\n").replace("\n", "<br/>")}</div>'
                    # Set consistent font size
                    line_html = f'<div style="font-size: 1em;">{content_html}</div>'
                else:
                    content = content.replace("\n", "<br/>")
                    line_html = f'<div style="margin-top: 0.5em;"><span class="user-label" style="color:#FF9500;">User:</span><br/>{content}</div>'
                
                lines_html.append(line_html)
            elif line.startswith("AI:\n"):
                content = line[len("AI:\n"):]
                content = highlight_thinking_tokens(content)
                
                if use_markdown:
                    try:
                        # Create custom HTML for the AI heading with orange styling as H3
                        ai_heading = f"<h3 class='ai-heading' style='color:#FF9500 !important;'>AI:</h3>"
                        
                        # Append the content after the custom heading
                        # Apply our custom markdown extension along with standard extensions
                        md = markdown.Markdown(extensions=[
                            'fenced_code', 'tables', 'nl2br', CodeExtension()
                        ])
                        processed_content = md.convert(content)
                        content_html = f'<div class="markdown-content">{ai_heading}{processed_content}</div>'
                    except Exception as e:
                        print(f"DEBUG: Markdown parsing error: {e}")
                        content_html = f'<div style="margin-top: 0.5em;"><span class="ai-label" style="color:#FF9500;">AI:</span><br/>{content.replace("<br/>", "\n").replace("\n", "<br/>")}</div>'
                    # Set consistent font size
                    line_html = f'<div style="font-size: 1em;">{content_html}</div>'
                else:
                    content = content.replace("\n", "<br/>")
                    line_html = f'<div style="margin-top: 0.5em;"><span class="ai-label" style="color:#FF9500;">AI:</span><br/>{content}</div>'
                
                lines_html.append(line_html)
            else:
                if use_markdown:
                    try:
                        line_html = highlight_thinking_tokens(line)
                        md = markdown.Markdown(extensions=[
                            'fenced_code', 'tables', 'nl2br', CodeExtension()
                        ])
                        line_html = md.convert(line_html)
                        line_html = f'<div class="markdown-content">{line_html}</div>'
                    except Exception:
                        line_html = highlight_thinking_tokens(line).replace("\n", "<br/>")
                else:
                    line_html = highlight_thinking_tokens(line).replace("\n", "<br/>")
                lines_html.append(line_html)
                
        joined_html = "\n".join(lines_html)
        final_html = f"<div style='line-height:1.1; margin:0; padding:0;'>{joined_html}</div>"
        return final_html
        
    except Exception as e:
        print(f"DEBUG: Error building HTML: {e}")
        # Fallback to simple non-markdown display
        simple_html = "<div>"
        for line in history or conversation_history:
            if line.startswith("User:\n"):
                line_html = f'<div style="margin-top: 0.5em;"><span style="color:#FF9500; font-weight:bold;">User:</span><br/>{line[6:].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")}</div>'
            elif line.startswith("AI:\n"):
                line_html = f'<div style="margin-top: 0.5em;"><span style="color:#FF9500; font-weight:bold;">AI:</span><br/>{line[4:].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")}</div>'
            else:
                line_html = line.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            simple_html += line_html
        simple_html += "</div>"
        return simple_html

# -------------------------------------------------------
# 5) Horizontal Separator
# -------------------------------------------------------
def create_separator():
    sep = QtWidgets.QFrame()
    sep.setFrameShape(QtWidgets.QFrame.HLine)
    sep.setFrameShadow(QtWidgets.QFrame.Sunken)
    sep.setLineWidth(1)
    sep.setStyleSheet("background-color: #262626;")
    return sep

# -------------------------------------------------------
# 6) Auto-resizing Text Edit
# -------------------------------------------------------
class AutoResizeTextEdit(QtWidgets.QTextEdit):
    enterPressed = QtCore.pyqtSignal()  # New signal for Enter key press
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.document().contentsChanged.connect(self.adjust_height)
        self.max_height = 1000  # Will be set dynamically
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(60)  # Minimum height for good usability
    
    def keyPressEvent(self, event):
        # Check if Enter was pressed without Shift modifier (normal Enter)
        if event.key() == QtCore.Qt.Key_Return and not (event.modifiers() & QtCore.Qt.ShiftModifier):
            self.enterPressed.emit()  # Emit signal to trigger send action
        else:
            # For all other keys including Shift+Enter, use default handling
            super().keyPressEvent(event)
        
    def adjust_height(self):
        # Calculate the height of the document
        doc_height = self.document().size().height()
        doc_margin = self.document().documentMargin()
        content_height = doc_height + 2 * doc_margin + 10  # Add a small buffer
        
        # Constrain to max_height
        new_height = min(content_height, self.max_height)
        
        # Ensure minimum height
        new_height = max(new_height, 60)
        
        # Apply the new height - Convert float to int
        if new_height != self.height():
            self.setFixedHeight(int(new_height))
    
    def set_max_height(self, height):
        self.max_height = max(height, 60)  # Ensure minimum reasonable height
        self.adjust_height()

# -------------------------------------------------------
# 7) Worker Thread for Streaming Responses - Updated for LM Studio
# -------------------------------------------------------
class ModelLoadingWorker(QtCore.QObject):
    """Worker for background model loading to prevent UI freezes"""
    modelLoaded = QtCore.pyqtSignal(bool, int)  # success, context_window
    modelError = QtCore.pyqtSignal(str)  # error message
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.should_stop = False
    
    def run(self):
        """Load model in background thread"""
        global lmstudio_server
        
        print(f"DEBUG: Background loading model: {self.model_name}")
        
        try:
            # First try to get the context window size
            context_size = get_context_window_size(self.model_name)
            
            if self.should_stop:
                print("DEBUG: Model loading cancelled")
                return
                
            # Now check if model is ready with a simple test
            try:
                # Simple test prompt
                test_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Write just one word: Hello"}
                ]
                
                payload = {
                    "model": self.model_name,
                    "messages": test_messages,
                    "max_tokens": 10,
                    "temperature": 0.0
                }
                
                response = requests.post(
                    f"{lmstudio_server}/v1/chat/completions", 
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60  # Extended timeout to 60 seconds for large models
                )
                
                if self.should_stop:
                    print("DEBUG: Model loading cancelled after API call")
                    return
                
                if response.status_code == 200:
                    # Check if we got a valid response with content
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content.strip():
                            # We got a response with actual content
                            print(f"DEBUG: Model {self.model_name} loaded successfully")
                            self.modelLoaded.emit(True, context_size)
                            return
                
                # If we get here, model is not fully ready
                print(f"DEBUG: Model {self.model_name} responded but may not be fully loaded")
                self.modelError.emit(f"Model loaded but may not be fully initialized")
            except Exception as e:
                print(f"DEBUG: Error checking model readiness: {e}")
                # We at least got the context size, so report partial success
                self.modelLoaded.emit(False, context_size)
        except Exception as e:
            print(f"DEBUG: Error loading model: {e}")
            self.modelError.emit(f"Error loading model: {str(e)}")

class RequestWorker(QtCore.QObject):
    newChunk = QtCore.pyqtSignal(str)
    tokenCountUpdate = QtCore.pyqtSignal(int, int)  # used_tokens, max_tokens
    finished = QtCore.pyqtSignal()
    connectionError = QtCore.pyqtSignal(str)  # Signal for connection errors
    stoppedByUser = QtCore.pyqtSignal()  # Signal for user-initiated stops

    def __init__(self, prompt, current_history, max_context):
        super().__init__()
        self.prompt = prompt
        self.current_history = current_history.copy()
        self.max_context = max_context
        self.prompt_eval = 0
        self.eval_count = 0
        self.ai_response = ""
        self.estimated_prompt_tokens = len(prompt) // 4
        self.accumulated_chunks = ""
        self.last_emit_time = time.time()
        self.should_stop = False  # Flag to indicate worker should stop
        self.stop_emitted = False  # Flag to track if stop signal was already emitted
        self.response = None  # Store the response object
        
    def run(self):
        global selected_model, lmstudio_server
        self.ai_response = ""
        self.stop_emitted = False
        start_time = time.time()
        max_generation_time = 180  # 3 minutes max to prevent UI lockup
        
        # Convert conversation history to messages format for LM Studio
        messages = []
        
        # LM Studio uses OpenAI format with system, user, assistant messages
        system_message_added = False
        
        for entry in self.current_history:
            if entry.startswith("User:\n"):
                messages.append({
                    "role": "user",
                    "content": entry[len("User:\n"):]
                })
            elif entry.startswith("AI:\n"):
                content = entry[len("AI:\n"):]
                if content.strip():  # Only add non-empty messages
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
        
        # Add a system message at the beginning if none exists
        if not system_message_added:
            messages.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant."
            })
        
        # LM Studio payload format
        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Initialize counters
        last_response_length = 0
        estimated_tokens = 0
        
        # Start with a blank token count
        self.tokenCountUpdate.emit(0, self.max_context)
        print(f"DEBUG: Initial token count cleared, will update at completion")

        for attempt in range(3):
            if self.should_stop:
                print("DEBUG: Worker stopping before request")
                if not self.stop_emitted:
                    self.stoppedByUser.emit()
                    self.stop_emitted = True
                    print("DEBUG: Stop signal emitted before request")
                self.finished.emit()
                return
                
            try:
                # Set a timeout to prevent UI lockup with large models
                self.response = requests.post(
                    f"{lmstudio_server}/v1/chat/completions", 
                    json=payload, 
                    stream=True, 
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
                
                if self.response.status_code == 200:
                    try:
                        for line in self.response.iter_lines():
                            # Check timeout to avoid UI lockup
                            current_time = time.time()
                            if current_time - start_time > max_generation_time:
                                print("DEBUG: Generation timeout reached")
                                self.connectionError.emit("Generation timeout reached")
                                if self.response and hasattr(self.response, 'close'):
                                    self.response.close()
                                self.finished.emit()
                                return
                            
                            # Check if we should stop
                            if self.should_stop:
                                print("DEBUG: Worker stopping due to should_stop flag")
                                
                                # Safely close response
                                try:
                                    if self.response and hasattr(self.response, 'close'):
                                        self.response.close()
                                        print("DEBUG: Closed response connection during stop")
                                except Exception as e:
                                    print(f"DEBUG: Non-critical error closing response: {e}")
                                
                                if not self.stop_emitted:
                                    self.stoppedByUser.emit()
                                    self.stop_emitted = True
                                    print("DEBUG: Stop signal emitted")
                                    
                                self.finished.emit()
                                return
                                
                            # Process events to keep UI responsive
                            QtWidgets.QApplication.processEvents()

                            if line:
                                try:
                                    # Handle SSE format from LM Studio
                                    line_text = line.decode('utf-8')
                                    if line_text == "data: [DONE]":
                                        # End of the stream
                                        # Emit any remaining accumulated chunks
                                        if self.accumulated_chunks:
                                            self.newChunk.emit(self.accumulated_chunks)
                                        
                                        # Rough estimate for token usage
                                        # This is approximate - we're using the length/4 method
                                        input_estimate = sum(len(msg["content"]) for msg in messages) // 4
                                        output_estimate = len(self.ai_response) // 4
                                        total_estimate = input_estimate + output_estimate
                                        
                                        print(f"DEBUG: Estimated token usage: {total_estimate}")
                                        self.tokenCountUpdate.emit(total_estimate, self.max_context)
                                        
                                        # Safely close response
                                        try:
                                            if self.response and hasattr(self.response, 'close'):
                                                self.response.close()
                                        except Exception as e:
                                            print(f"DEBUG: Error while closing response: {e}")
                                            
                                        self.finished.emit()
                                        return
                                    
                                    # Parse data line
                                    if line_text.startswith("data: "):
                                        data_json = line_text[6:]  # Skip "data: " prefix
                                        data = json.loads(data_json)
                                        
                                        # Extract the content delta - LM Studio uses OpenAI format
                                        choices = data.get("choices", [])
                                        if choices and len(choices) > 0:
                                            delta = choices[0].get("delta", {})
                                            content = delta.get("content", "")
                                            
                                            if content:
                                                # Process content and add to response
                                                self.ai_response += content
                                                
                                                # Accumulate chunks and emit less frequently for smoother UI
                                                self.accumulated_chunks += content
                                                current_time = time.time()
                                                
                                                # Emit chunks if enough time has passed or we have enough content
                                                if current_time - self.last_emit_time > 0.3 or len(self.accumulated_chunks) > 50:
                                                    self.newChunk.emit(self.accumulated_chunks)
                                                    self.accumulated_chunks = ""
                                                    self.last_emit_time = current_time
                                                    
                                                    # Process events for responsiveness - but limit to reduce overload
                                                    QtWidgets.QApplication.processEvents()
                                            
                                except json.JSONDecodeError as e:
                                    print(f"DEBUG: Failed to parse chunk: {line.decode('utf-8')}")
                                    # Skip problematic chunks but continue processing
                                    continue
                                except Exception as e:
                                    print(f"DEBUG: Error processing chunk: {str(e)}")
                                    continue
                                    
                    except Exception as e:
                        # Handle any exception during response reading
                        if self.should_stop:
                            print(f"DEBUG: Exception during stop (expected): {str(e)}")
                            if not self.stop_emitted:
                                self.stoppedByUser.emit()
                                self.stop_emitted = True
                                print("DEBUG: Stop signal emitted after exception")
                        else:
                            error_msg = f"Error during response: {str(e)}"
                            print(f"DEBUG: {error_msg}")
                            self.connectionError.emit(error_msg)
                        
                        # Safely close response
                        try:
                            if self.response and hasattr(self.response, 'close'):
                                self.response.close()
                                print("DEBUG: Response closed at exception")
                        except Exception as e:
                            print(f"DEBUG: Error while closing response: {e}")
                    
                        self.finished.emit()
                        return
                else:
                    # Handle HTTP errors
                    error_msg = f"Server error: {self.response.status_code}"
                    print(f"DEBUG: {error_msg}")
                    if attempt == 2:  # Last attempt
                        self.connectionError.emit(error_msg)
                        self.finished.emit()
                        return
            except requests.exceptions.ConnectionError:
                error_msg = "Connection error"
                print(f"DEBUG: {error_msg}")
                time.sleep(1)
                if attempt == 2:  # Last attempt
                    self.connectionError.emit(error_msg)
                    self.finished.emit()
                    return
            except requests.exceptions.Timeout:
                error_msg = "Request timeout"
                print(f"DEBUG: {error_msg}")
                self.connectionError.emit(error_msg)
                self.finished.emit()
                return
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(f"DEBUG: {error_msg}")
                self.connectionError.emit(error_msg)
                self.finished.emit()
                return
                
    def __del__(self):
        """Clean up resources when the worker is deleted"""
        if hasattr(self, 'response') and self.response:
            try:
                if hasattr(self.response, 'close'):
                    self.response.close()
                    print("DEBUG: Response closed in __del__")
            except:
                pass

# -------------------------------------------------------
# 8) Main Chat Window
# -------------------------------------------------------
class ChatWindow(QtWidgets.QWidget):
    def update_model_status_indicator(self, status):
        """Update the model status indicator color and tooltip
        status can be: 'unloaded', 'loading', 'loaded', 'error'
        """
        if status == 'unloaded':
            self.model_status_indicator.setStyleSheet("""
                background-color: #000000;
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 2px;
            """)
            self.model_status_indicator.setToolTip("Model Status: Unloaded")
        elif status == 'loading':
            self.model_status_indicator.setStyleSheet("""
                background-color: #FF9500;
                border: 1px solid #FFA530;
                border-radius: 8px;
                margin: 2px;
            """)
            self.model_status_indicator.setToolTip("Model Status: Loading")
        elif status == 'loaded':
            self.model_status_indicator.setStyleSheet("""
                background-color: #2A5E2A;
                border: 1px solid #3E8E3E;
                border-radius: 8px;
                margin: 2px;
            """)
            self.model_status_indicator.setToolTip("Model Status: Loaded")
        elif status == 'error':
            self.model_status_indicator.setStyleSheet("""
                background-color: #8B0000;
                border: 1px solid #A00000;
                border-radius: 8px;
                margin: 2px;
            """)
            self.model_status_indicator.setToolTip("Model Status: Error loading model")
    
    def cleanup_resources(self):
        """Clean up resources to prevent memory leaks"""
        # Stop any running worker thread
        if self.thread is not None and self.thread.isRunning():
            try:
                if self.worker is not None:
                    self.worker.should_stop = True
                    
                    # Close response connection if it exists
                    if hasattr(self.worker, 'response') and self.worker.response:
                        try:
                            if hasattr(self.worker.response, 'close'):
                                self.worker.response.close()
                        except Exception as e:
                            print(f"DEBUG: Error closing response: {e}")
                
                # Wait for thread to finish with timeout
                self.thread.quit()
                if not self.thread.wait(500):  # Wait up to 500ms
                    print("DEBUG: Thread did not terminate in time, forcing termination")
                    self.thread.terminate()
                    self.thread.wait()
                    
            except Exception as e:
                print(f"DEBUG: Error cleaning up thread: {e}")
        
        # Clean up model loading thread
        self.stop_model_loading_thread()
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        print("DEBUG: Resources cleaned up")
            
    def load_selected_model(self):
        """Load the selected model in a background thread to avoid freezing the UI"""
        global selected_model, model_loading, model_ready, display_to_model_map
        
        # First clean up any existing resources
        self.cleanup_resources()
        
        # Get the currently selected display name from the combo box
        display_name = self.model_combo.currentText()
        
        # Look up the actual model ID from the display name
        if display_name in display_to_model_map:
            model_name = display_to_model_map[display_name]
            print(f"DEBUG: Loading selected model: display='{display_name}', model ID='{model_name}'")
        else:
            # Fallback if display name not found in map
            model_name = display_name
            if model_name.startswith("[VLM] "):
                model_name = model_name[6:]  # Remove "[VLM] " prefix
            if " (ctx:" in model_name:
                model_name = model_name.split(" (ctx:")[0]  # Remove context window info
                
            print(f"DEBUG: Display name '{display_name}' not found in map, using as model ID: '{model_name}'")
        
        # Skip if invalid model
        if not model_name or model_name.startswith("<"):
            self.status_label.setText("❌ No valid model selected!")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return
                
        # Check if we're already loading this model
        if self.model_loading_in_progress and selected_model == model_name:
            self.status_label.setText(f"⏳ Already loading model: {model_name}")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return
                
        # Set loading state
        model_loading = True
        model_ready = False
        self.model_loading_in_progress = True
        self.update_ui_state()
        self.update_model_status_indicator('loading')
        
        # Update the selected model
        selected_model = model_name
        print(f"DEBUG: Loading model: {model_name}")
        
        # Show status message
        self.status_label.setText(f"⏳ Loading model: {model_name}")
        QtWidgets.QApplication.processEvents()
        
        # Stop any existing thread safely
        self.stop_model_loading_thread()
                
        # Create new thread for model loading
        self.model_loading_thread = QtCore.QThread()
        self.model_loading_worker = ModelLoadingWorker(model_name)
        self.model_loading_worker.moveToThread(self.model_loading_thread)
        
        # Connect signals
        self.model_loading_worker.modelLoaded.connect(self.handle_model_loaded)
        self.model_loading_worker.modelError.connect(self.handle_model_error)
        self.model_loading_thread.started.connect(self.model_loading_worker.run)
        
        # Cleanup
        self.model_loading_worker.modelLoaded.connect(self.model_loading_thread.quit)
        self.model_loading_worker.modelError.connect(self.model_loading_thread.quit)
        
        # We need to use a lambda to ensure reference safety
        self.model_loading_thread.finished.connect(
            lambda: self.cleanup_model_loading_thread())
        
        # Start the thread
        self.model_loading_thread.start()
    
    def stop_model_loading_thread(self):
        """Safely stop the model loading thread if it's running"""
        try:
            if self.model_loading_worker is not None:
                self.model_loading_worker.should_stop = True
                
            if self.model_loading_thread is not None and self.model_loading_thread.isRunning():
                print("DEBUG: Stopping previous model loading thread")
                self.model_loading_thread.quit()
                self.model_loading_thread.wait(1000)  # Wait up to 1 second
                print("DEBUG: Previous model loading thread stopped")
        except Exception as e:
            print(f"DEBUG: Error stopping model loading thread: {e}")
    
    def cleanup_model_loading_thread(self):
        """Clean up thread and worker objects"""
        print("DEBUG: Cleaning up model loading thread")
        try:
            if self.model_loading_worker is not None:
                self.model_loading_worker.deleteLater()
                self.model_loading_worker = None
                
            if self.model_loading_thread is not None:
                self.model_loading_thread.deleteLater()
                self.model_loading_thread = None
        except Exception as e:
            print(f"DEBUG: Error during cleanup: {e}")
        
    def handle_model_loaded(self, is_ready, context_size):
        """Handle the response from the model loading thread"""
        global model_loading, model_ready, selected_model
        
        # First set the context size from the model loading process
        self.current_model_context = context_size
        self.token_count_label.setText(f"Tokens: {self.last_token_count} / {self.current_model_context}")
        
        if is_ready:
            model_ready = True
            self.update_model_status_indicator('loaded')
            
            # Now check for the actual context length of the loaded model
            actual_context = get_actual_context_length(selected_model)
            if actual_context > 0:
                self.current_model_context = actual_context
                self.token_count_label.setText(f"Tokens: {self.last_token_count} / {self.current_model_context}")
                self.status_label.setText(f"✓ Model loaded with {self.current_model_context} context window (actual configured value)")
            else:
                self.status_label.setText(f"✓ Model loaded with {self.current_model_context} context window (max value)")
                
            # Update the loaded model in the combo box for highlighting
            self.loaded_model = selected_model
            # Update the combo box delegate to highlight the loaded model
            if hasattr(self.model_combo, 'view') and self.model_combo.view():
                delegate = self.model_combo.view().itemDelegate()
                if hasattr(delegate, 'setLoadedModel'):
                    delegate.setLoadedModel(selected_model)
            self.model_combo.setLoadedModel(selected_model)
        else:
            model_ready = False
            self.update_model_status_indicator('error')
            self.status_label.setText(f"⚠️ Got context size ({self.current_model_context}) but model may not be fully ready")
        
        # Reset loading state and update UI
        model_loading = False
        self.model_loading_in_progress = False
        self.update_ui_state()
        
        # Keep status visible for a little longer
        QtCore.QTimer.singleShot(5000, lambda: self.status_label.setText(""))
    
    def handle_model_error(self, error_msg):
        """Handle errors from the model loading thread"""
        global model_loading, model_ready
        
        model_ready = False
        self.update_model_status_indicator('error')
        self.status_label.setText(f"❌ {error_msg}")
        
        # Reset loading state and update UI
        model_loading = False
        self.model_loading_in_progress = False
        self.update_ui_state()
        
        # Keep status visible for a little longer
        QtCore.QTimer.singleShot(5000, lambda: self.status_label.setText(""))
    
    def update_token_count(self, used: int, maximum: int):
        """Update the token count display with the current usage"""
        # Update the token count display
        if used > 0:  # Only show non-zero token counts (final result)
            print(f"DEBUG: Updating token count: {used} / {maximum}")
            self.last_token_count = used  # Store the token count for future reference
            self.token_count_label.setText(f"Tokens: {used} / {maximum}")
            
            # Process events to ensure the UI updates immediately
            QtWidgets.QApplication.processEvents()
        
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LM Studio Chat")
        self.resize(900, 650)
        self.current_ai_text = ""
        self.current_ai_index = -1
        self.thread = None
        self.worker = None
        self.last_token_count = 0  # Store the last known token count
        self.api_connected = False  # Track API connection status
        self.current_model_context = 4096  # Default context size
        self.stop_requested = False  # Track if stop was requested
        self.auto_reconnect_enabled = False  # Flag to control automatic reconnection
        self.last_ui_update_time = 0  # Track last UI update time
        self.showed_markdown_warning = False  # Track if we showed the markdown warning
        
        # Initialize model loading thread references to None
        self.model_loading_thread = None
        self.model_loading_worker = None
        self.model_loading_in_progress = False
        
        # Keep track of currently loaded model for highlighting
        self.loaded_model = ""
        
        # Set up the UI
        self.setupUI()
        
        # Setup API connection checker timer
        self.api_check_timer = QtCore.QTimer(self)
        self.api_check_timer.timeout.connect(self.check_api_connection)
        
        # Add tooltips to explain UI elements
        self.send_button.setToolTip("Send message (only enabled when model is fully loaded)")
        self.load_model_button.setToolTip("Load the selected model")
        self.ip_input.setToolTip("IP address of the LM Studio server")
        self.api_port_input.setToolTip("Port for LM Studio API (default: 1234)")
        self.remote_port_input.setToolTip("Port for remote unload server (default: 5051)")
        
        # Important: Connect the signal AFTER UI setup to avoid issues
        QtCore.QTimer.singleShot(100, self.connectModelSignal)
        
        # Set an initial status message to guide the user
        self.status_label.setText("Connect to LM Studio API server to begin")
        QtCore.QTimer.singleShot(5000, lambda: self.status_label.setText(""))
        
    def connectModelSignal(self):
        """Connect model selection changed signal after UI is fully initialized"""
        try:
            # First try to disconnect in case it's already connected
            try:
                self.model_combo.currentTextChanged.disconnect(self.change_model)
            except Exception:
                pass
                
            # Now connect the signal
            self.model_combo.currentTextChanged.connect(self.change_model)
            print("DEBUG: Model selection signal connected")
        except Exception as e:
            print(f"DEBUG: Error connecting model signal: {e}")
        
    def closeEvent(self, event):
        # Show closing status
        self.status_label.setText("⏳ Closing application and unloading models...")
        QtWidgets.QApplication.processEvents()
        
        # Stop all threads
        self.stop_model_loading_thread()
        
        # Stop the timer before closing
        if hasattr(self, 'api_check_timer') and self.api_check_timer.isActive():
            self.api_check_timer.stop()
        
        # Clean up any running thread
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(1000)  # Wait up to 1 second
        
        # Unload all models before exiting
        if self.api_connected:
            try:
                unload_all_models()
                print("DEBUG: Models unloaded during application exit")
                
                # Update model status indicator for visual feedback
                self.update_model_status_indicator('unloaded')
                QtWidgets.QApplication.processEvents()
            except Exception as e:
                print(f"DEBUG: Error unloading models on exit: {e}")
        
        # Accept the close event
        event.accept()
        
    def toggle_markdown(self):
        """Toggle between markdown and plain text formatting modes."""
        global use_markdown
        use_markdown = not use_markdown
        
        # Update button text
        if use_markdown:
            self.markdown_button.setText("Markdown: ON")
        else:
            self.markdown_button.setText("Markdown: OFF")
        
        # Update display immediately
        self.update_chat_history()
        
        # Show status message
        status_msg = "✓ Markdown formatting enabled." if use_markdown else "✓ Plain text formatting enabled."
        self.status_label.setText(status_msg)
        QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
    
    def change_model(self, display_name):
        """Handle model selection changes from the dropdown - just update selected model, don't load it"""
        global selected_model, display_to_model_map
        
        # Skip if display_name is empty (happens during combo box clear)
        if not display_name:
            print("DEBUG: Empty display name in change_model - skipping")
            return
                
        print(f"DEBUG: Dropdown selection changed to: '{display_name}'")
        
        # Look up the actual model ID from the display name
        if display_name in display_to_model_map:
            model_name = display_to_model_map[display_name]
            print(f"DEBUG: Found model ID '{model_name}' for display name '{display_name}'")
        else:
            # Fallback if display name not found in map
            model_name = display_name.split(" (ctx:")[0]  # Try to extract model name
            if model_name.startswith("[VLM] "):
                model_name = model_name[6:]  # Remove "[VLM] " prefix
                    
            print(f"DEBUG: Display name '{display_name}' not found in map, using extracted ID: '{model_name}'")
        
        if model_name == selected_model or model_name.startswith("<") or not model_name:
            print(f"DEBUG: Skipping model change (same model or invalid)")
            return  # Skip if same model or placeholder
        
        # Update the selected model (but don't load it - that happens with the Load button)
        selected_model = model_name
        print(f"DEBUG: Selected model changed to: {model_name}")
        
        # Update status message
        self.status_label.setText(f"Model selected: {model_name} (click 'Load Model' to load)")
        QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
    
    def check_model_ready(self, model_name, set_loading_state=True):
        """
        Check if the model is fully loaded and ready to use.
        Performs a more thorough check to ensure model can actually generate text.
        """
        global model_loading, model_ready
        
        if set_loading_state:
            # Set loading state and update UI
            model_loading = True
            model_ready = False  # Reset ready state
            self.update_ui_state()
            self.status_label.setText("⏳ Checking if model is ready...")
            QtWidgets.QApplication.processEvents()
        
        # Ensure we have a valid model name
        if not model_name or model_name.startswith("<"):
            if set_loading_state:
                model_loading = False
                model_ready = False
                self.update_ui_state()
                self.status_label.setText("⚠️ No valid model selected")
                QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
            return False
        
        try:
            # More detailed test prompt that requires some generation
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write just one word: Hello"}
            ]
            
            payload = {
                "model": model_name,
                "messages": test_messages,
                "max_tokens": 10,
                "temperature": 0.0
            }
            
            # First check if we can connect to the model - use a much longer timeout for large models
            self.status_label.setText("⏳ Loading and testing model (may take several minutes for large models)...")
            QtWidgets.QApplication.processEvents()
            
            response = requests.post(
                f"{lmstudio_server}/v1/chat/completions", 
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Extended timeout to 60 seconds for large models
            )
            
            if response.status_code == 200:
                # Check if we got a valid response with content
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    if content.strip():
                        # We got a response with actual content
                        if set_loading_state:
                            model_loading = False
                            model_ready = True
                            self.update_ui_state()
                            self.status_label.setText("✓ Model is loaded and ready")
                            QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
                        return True
                
                # If we got a response but no valid content, model might not be fully initialized
                print(f"DEBUG: Model responded but returned empty or invalid content")
                if set_loading_state:
                    model_loading = False
                    model_ready = False
                    self.update_ui_state()
                    self.status_label.setText("⚠️ Model responded but may not be fully loaded")
                    QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
                return False
            else:
                # Non-200 response, model is not ready
                print(f"DEBUG: Model not ready. Status code: {response.status_code}")
                if set_loading_state:
                    model_loading = False
                    model_ready = False
                    self.update_ui_state()
                    self.status_label.setText("⚠️ Model not ready (API error)")
                    QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
                return False
                
        except requests.exceptions.Timeout:
            print("DEBUG: Request timed out - model may be very large or still loading")
            if set_loading_state:
                model_loading = False
                model_ready = False
                self.update_ui_state()
                self.status_label.setText("⚠️ Model loading timed out - may need more time")
                QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
            return False
        except Exception as e:
            # Exception occurred, model is not ready
            print(f"DEBUG: Error checking model readiness: {e}")
            if set_loading_state:
                model_loading = False
                model_ready = False
                self.update_ui_state()
                self.status_label.setText("⚠️ Error connecting to model")
                QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
            return False
        
    def fetch_model_context(self, model_name: str):
        """Fetch the model's context window size"""
        global model_loading
        
        # Set loading state and update UI
        model_loading = True
        self.update_ui_state()
        self.status_label.setText("⏳ Detecting context window size...")
        QtWidgets.QApplication.processEvents()
        
        try:
            context_size = get_context_window_size(model_name)
            self.current_model_context = context_size
            self.token_count_label.setText(f"Tokens: {self.last_token_count} / {self.current_model_context}")
            
            # Don't reset model_loading here, as we still need to check if model is ready
            self.status_label.setText(f"✓ Context window: {context_size} tokens")
            return True
        except Exception as e:
            print(f"DEBUG: Error fetching context window: {e}")
            # Don't reset model_loading here, as we still need to check if model is ready
            self.status_label.setText("⚠️ Could not detect context size")
            return False

    def update_ui_state(self):
        """Update UI element states based on connection and loading status"""
        # First check if we're connected
        if not self.api_connected:
            # If not connected, disable all buttons except connect
            self.send_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.markdown_button.setEnabled(False)
            self.load_model_button.setEnabled(False)
            self.model_combo.setEnabled(False)
            
            # Reset model status indicator
            self.update_model_status_indicator('unloaded')
            return
            
        # If connected, enable model selection and load button
        self.model_combo.setEnabled(True)
        self.load_model_button.setEnabled(True)
            
        # Now check if we're loading a model
        if model_loading:
            # If loading, disable all buttons except connect and model selection
            self.send_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.markdown_button.setEnabled(False)
            self.load_model_button.setEnabled(False)
        else:
            # Enable general UI buttons
            self.reset_button.setEnabled(True)
            self.markdown_button.setEnabled(True)
            self.load_model_button.setEnabled(True)
            
            # Only enable send button if model is ready
            self.send_button.setEnabled(model_ready)
            
            # Update the send button appearance
            if model_ready:
                self.send_button.setStyleSheet("")  # Normal style
            else:
                # Dimmed style for not-ready model
                self.send_button.setStyleSheet("""
                    background-color: #333333;
                    color: #777777;
                """)
    
    def set_loading_buttons(self, enabled):
        """Helper to set button states during loading operations"""
        # We always disable the send button during operations
        self.send_button.setEnabled(False)
        
        # Reset and markdown buttons can be toggled
        self.reset_button.setEnabled(enabled)
        self.markdown_button.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.load_model_button.setEnabled(enabled)
        QtWidgets.QApplication.processEvents()
    
    def setupUI(self):
        main_layout = QtWidgets.QVBoxLayout()

        bold_label_font = QtGui.QFont()
        bold_label_font.setPointSize(11)
        bold_label_font.setBold(True)
        
        # --- Server Connection ---
        server_label = QtWidgets.QLabel("Server Connection:")
        server_label.setAlignment(QtCore.Qt.AlignLeft)
        server_label.setFont(bold_label_font)
        main_layout.addWidget(server_label)

        row_server_layout = QtWidgets.QHBoxLayout()
        
        # IP Address input
        ip_label = QtWidgets.QLabel("IP:")
        row_server_layout.addWidget(ip_label)
        self.ip_input = QtWidgets.QLineEdit(server_ip)
        self.ip_input.setFixedWidth(120)
        self.ip_input.setToolTip("Server IP address")
        row_server_layout.addWidget(self.ip_input)
        
        # API Port input
        api_port_label = QtWidgets.QLabel("API Port:")
        row_server_layout.addWidget(api_port_label)
        self.api_port_input = QtWidgets.QLineEdit(api_port)
        self.api_port_input.setFixedWidth(60)
        self.api_port_input.setToolTip("LM Studio API port (default: 1234)")
        row_server_layout.addWidget(self.api_port_input)
        
        # Remote Port input
        remote_port_label = QtWidgets.QLabel("Remote Port:")
        row_server_layout.addWidget(remote_port_label)
        self.remote_port_input = QtWidgets.QLineEdit(remote_port)
        self.remote_port_input.setFixedWidth(60)
        self.remote_port_input.setToolTip("Remote unload server port (default: 5051)")
        row_server_layout.addWidget(self.remote_port_input)
        
        # Connection status indicator (circular)
        self.connection_indicator = QtWidgets.QLabel()
        self.connection_indicator.setFixedSize(16, 16)
        self.connection_indicator.setStyleSheet("""
            background-color: #000000;
            border-radius: 8px;
            margin: 2px;
        """)
        self.connection_indicator.setToolTip("Disconnected")
        row_server_layout.addWidget(self.connection_indicator)
        
        # Connect/Disconnect button
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        row_server_layout.addWidget(self.connect_button)
        
        row_server_layout.addStretch(1)
        main_layout.addLayout(row_server_layout)
        main_layout.addWidget(create_separator())

        # --- Select Model ---
        model_label = QtWidgets.QLabel("Select Model:")
        model_label.setAlignment(QtCore.Qt.AlignLeft)
        model_label.setFont(bold_label_font)
        main_layout.addWidget(model_label)

        row_model_layout = QtWidgets.QHBoxLayout()
        # Use our custom combo box instead of standard one
        self.model_combo = ModelComboBox()
        self.model_combo.setFixedWidth(450)
        
        # Set up the custom view with our delegate
        list_view = QtWidgets.QListView()
        self.model_combo.setView(list_view)
        
        row_model_layout.addWidget(self.model_combo)
        
        # Load Model button
        self.load_model_button = QtWidgets.QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_selected_model)
        self.load_model_button.setToolTip("Load the selected model")
        row_model_layout.addWidget(self.load_model_button)
        
        # Model status indicator (circular)
        self.model_status_indicator = QtWidgets.QLabel()
        self.model_status_indicator.setFixedSize(16, 16)
        self.model_status_indicator.setStyleSheet("""
            background-color: #000000;
            border-radius: 8px;
            margin: 2px;
        """)
        self.model_status_indicator.setToolTip("Model Status: Unloaded")
        row_model_layout.addWidget(self.model_status_indicator)
        
        self.token_count_label = QtWidgets.QLabel("Tokens: 0 / 0")
        self.token_count_label.setFont(bold_label_font)
        row_model_layout.addWidget(self.token_count_label)
        
        row_model_layout.addStretch(1)
        main_layout.addLayout(row_model_layout)
        main_layout.addWidget(create_separator())

        # --- Chat History ---
        chat_label = QtWidgets.QLabel("Chat History:")
        chat_label.setAlignment(QtCore.Qt.AlignLeft)
        chat_label.setFont(bold_label_font)
        main_layout.addWidget(chat_label)

        # Create a container widget for the chat and prompt areas
        self.chat_prompt_container = QtWidgets.QWidget()
        chat_prompt_layout = QtWidgets.QVBoxLayout(self.chat_prompt_container)
        chat_prompt_layout.setContentsMargins(0, 0, 0, 0)
        
        # Use standard QTextEdit for chat history
        self.chat_history_widget = QtWidgets.QTextEdit()
        self.chat_history_widget.setObjectName("ChatHistory")
        self.chat_history_widget.setReadOnly(True)
        self.chat_history_widget.setWordWrapMode(QtGui.QTextOption.WordWrap)
        
        # Set scrollbar stylesheet to ensure consistent appearance
        self.chat_history_widget.setStyleSheet("""
        QScrollBar:vertical {
            background-color: #3b3b3b;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #555555;
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            background: none;
            border: none;
            height: 0px;
        }
        QScrollBar:horizontal {
            background-color: #3b3b3b;
            height: 12px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #555555;
            min-width: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            background: none;
            border: none;
            width: 0px;
        }
        """)
        
        # --- Prompt ---
        prompt_label = QtWidgets.QLabel("Enter your prompt:")
        prompt_label.setAlignment(QtCore.Qt.AlignLeft)
        prompt_label.setFont(bold_label_font)
        
        # Auto-resizing prompt input
        self.prompt_input = AutoResizeTextEdit()
        self.prompt_input.setObjectName("PromptInput")
        self.prompt_input.setWordWrapMode(QtGui.QTextOption.WordWrap)
        self.prompt_input.enterPressed.connect(self.send_message)  # Connect Enter key to send_message
        self.prompt_input.setStyleSheet("""
        QScrollBar:vertical {
            background-color: #3b3b3b;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #555555;
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            background: none;
            border: none;
            height: 0px;
        }
        QScrollBar:horizontal {
            background-color: #3b3b3b;
            height: 12px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #555555;
            min-width: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            background: none;
            border: none;
            width: 0px;
        }
        """)
        
        # Add widgets to the chat_prompt_layout
        chat_prompt_layout.addWidget(self.chat_history_widget, 1)  # Chat takes remaining space
        chat_prompt_layout.addWidget(create_separator())
        chat_prompt_layout.addWidget(prompt_label)
        chat_prompt_layout.addWidget(self.prompt_input, 0)  # Prompt has no stretch factor (sized by content)
        
        # Add the container to the main layout
        main_layout.addWidget(self.chat_prompt_container, 1)
        main_layout.addWidget(create_separator())

        # Status indicator label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #FF9500; font-weight: bold;")
        main_layout.addWidget(self.status_label)
        
        buttons_layout = QtWidgets.QHBoxLayout()
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_chat)
        self.markdown_button = QtWidgets.QPushButton("Markdown: ON")  # Default text
        self.markdown_button.clicked.connect(self.toggle_markdown)
        self.exit_button = QtWidgets.QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.send_button)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.markdown_button)
        buttons_layout.addWidget(self.exit_button)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        self.ip_input.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.api_port_input.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.remote_port_input.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.model_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        
        # Set initial prompt height
        self.updatePromptHeight()

    def showEvent(self, event):
        super().showEvent(event)
        # Set button height
        button_height = self.connect_button.sizeHint().height()
        self.ip_input.setFixedHeight(button_height)
        self.api_port_input.setFixedHeight(button_height)
        self.remote_port_input.setFixedHeight(button_height)
        self.model_combo.setFixedHeight(button_height)
        self.load_model_button.setFixedHeight(button_height)
        
        # Initialize model status indicator to unloaded
        self.update_model_status_indicator('unloaded')
        
        # Initial resize of prompt input
        QtCore.QTimer.singleShot(100, self.updatePromptHeight)
    
    def resizeEvent(self, event):
            super().resizeEvent(event)
            # When window is resized, update the prompt input max height
            self.updatePromptHeight()
    
    def updatePromptHeight(self):
        """Update the prompt input height based on the golden ratio"""
        if hasattr(self, 'chat_prompt_container') and hasattr(self.prompt_input, 'set_max_height'):
            container_height = self.chat_prompt_container.height()
            max_prompt_height = int(container_height * 0.382)  # Golden ratio proportion
            self.prompt_input.set_max_height(max_prompt_height)
            # Also trigger adjustment to current content
            self.prompt_input.adjust_height()
    
    def check_api_connection(self):
        """Check if the LM Studio API server is accessible and update UI accordingly"""
        global lmstudio_server, model_ready
        
        # Skip check if auto-reconnect is disabled and we're not connected
        if not self.auto_reconnect_enabled and not self.api_connected:
            return
            
        # Don't perform check if a generation is in progress
        if not self.send_button.isEnabled() and self.thread is not None and self.thread.isRunning():
            return
            
        try:
            # Properly construct the OpenAI-compatible API URL
            base_url = lmstudio_server.split('/v1')[0] if '/v1' in lmstudio_server else lmstudio_server
            models_url = f"{base_url}/v1/models"
            
            # Use a short timeout to avoid blocking the UI for too long
            response = requests.get(models_url, timeout=2.0)
            if response.status_code == 200:
                if not self.api_connected:
                    self.api_connected = True
                    self.update_connect_button()
                    print("DEBUG: API connection established")
                    # Update models since we're now connected
                    success, _, _ = fetch_models()
                    if success:
                        # Update the model dropdown with context information
                        self.update_model_combo()
                        # Set model status to unloaded
                        # Note: We don't auto-load the model here anymore
                        model_ready = False
                        self.update_model_status_indicator('unloaded')
                        self.update_ui_state()
            else:
                if self.api_connected:
                    self.api_connected = False
                    self.update_connect_button()
                    print(f"DEBUG: API connection lost (status {response.status_code})")
        except requests.exceptions.RequestException:
            if self.api_connected:
                self.api_connected = False
                self.update_connect_button()
                print("DEBUG: API connection lost (connection error)")
    
    def update_connect_button(self):
        """Update the connect button appearance based on connection status"""
        if self.api_connected:
            # Connected - green indicator and Disconnect button
            self.connection_indicator.setStyleSheet("""
                background-color: #2A5E2A;
                border: 1px solid #3E8E3E;
                border-radius: 8px;
                margin: 2px;
            """)
            self.connection_indicator.setToolTip("Connected to LM Studio API")
            self.connect_button.setText("Disconnect")
        else:
            # Not connected - black indicator and Connect button
            self.connection_indicator.setStyleSheet("""
                background-color: #000000;
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 2px;
            """)
            self.connection_indicator.setToolTip("Disconnected")
            self.connect_button.setText("Connect")
        
        # Update UI elements based on connection and loading state
        self.update_ui_state()
    
    def toggle_connection(self):
        """Toggle between connecting and disconnecting from the server"""
        global model_ready
        
        if self.api_connected:
            # Currently connected, so disconnect
            # Show disconnecting status
            self.status_label.setText("⏳ Disconnecting and unloading models...")
            QtWidgets.QApplication.processEvents()
            
            # Try to unload all models first
            try:
                unload_all_models()
                print("DEBUG: Models unloaded during disconnect")
            except Exception as e:
                print(f"DEBUG: Error unloading models on disconnect: {e}")
            
            self.api_connected = False
            self.auto_reconnect_enabled = False  # Disable auto reconnect
            self.api_check_timer.stop()  # Stop the connection check timer
            
            # Reset model ready state
            model_ready = False
            self.update_model_status_indicator('unloaded')
            
            self.update_connect_button()
            self.status_label.setText("✓ Disconnected from server")
            QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
            print("DEBUG: Manually disconnected from server")
        else:
            # Currently disconnected, so connect
            self.connect_server()
            
    def connect_server(self):
        """Connect to the specified LM Studio server"""
        global lmstudio_server, remote_server, server_ip, api_port, remote_port, model_ready
        
        # Reset model ready state
        model_ready = False
        self.update_ui_state()
        
        # Update server settings from UI inputs
        new_ip = self.ip_input.text().strip()
        new_api_port = self.api_port_input.text().strip()
        new_remote_port = self.remote_port_input.text().strip()
        
        if new_ip:
            server_ip = new_ip
        if new_api_port:
            api_port = new_api_port
        if new_remote_port:
            remote_port = new_remote_port
            
        # Update server URLs
        lmstudio_server = f"http://{server_ip}:{api_port}"
        remote_server = f"http://{server_ip}:{remote_port}"
        
        # Show connecting status
        self.status_label.setText(f"⏳ Connecting to {server_ip}...")
        QtWidgets.QApplication.processEvents()
            
        # Reset connection status 
        self.api_connected = False
        self.update_connect_button()
        
        # Enable auto reconnect and start the timer
        self.auto_reconnect_enabled = True
        self.api_check_timer.start(3000)  # Check every 3 seconds
        
        # Do an immediate check
        self.check_api_connection()
            
    def update_model_combo(self):
        """Update the model dropdown with available models and context window sizes"""
        global available_models, available_models_display
        
        try:
            print("DEBUG: Updating model dropdown with display names")
            
            # Remember current selection
            current_text = self.model_combo.currentText()
            
            # Temporarily disconnect signal to avoid triggering change_model
            try:
                self.model_combo.currentTextChanged.disconnect(self.change_model)
            except Exception:
                pass
                
            self.model_combo.clear()
            
            # Add models with their display names that include context windows
            self.model_combo.addItems(available_models_display)
            
            # Try to restore the previous selection
            index = self.model_combo.findText(current_text)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                print(f"DEBUG: Restored previous selection '{current_text}'")
            elif available_models_display and not available_models_display[0].startswith("<"):
                # If previous model not found but we have valid models, select the first one
                self.model_combo.setCurrentIndex(0)
                print(f"DEBUG: Selected first model '{available_models_display[0]}'")
                
            # Reconnect signal
            self.model_combo.currentTextChanged.connect(self.change_model)
            
        except Exception as e:
            print(f"DEBUG: Error updating model dropdown: {e}")
            # Fallback to simple update if something went wrong
            self.model_combo.clear()
            self.model_combo.addItems(available_models_display)
    
    def check_conversation_size(self):
        """Check if conversation is too large and trim if needed"""
        global conversation_history
        
        # Calculate total size of conversation history
        total_length = sum(len(msg) for msg in conversation_history)
        
        if total_length > MAX_CONVERSATION_SIZE:
            print(f"DEBUG: Conversation size ({total_length} bytes) exceeds threshold, trimming...")
            
            # Trim older messages to reduce memory usage
            while total_length > TRIM_TARGET_SIZE and len(conversation_history) > 4:
                removed = conversation_history.pop(0)
                total_length -= len(removed)
                # If we removed a user message, also remove the corresponding AI response
                if len(conversation_history) > 0 and conversation_history[0].startswith("AI:"):
                    removed = conversation_history.pop(0)
                    total_length -= len(removed)
            
            # Add a note that history was trimmed
            conversation_history.insert(0, "AI:\n[Older messages have been removed to improve performance]")
            
            # Show status message
            self.status_label.setText("ℹ️ Conversation history trimmed for performance")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            
            return True
        
        return False
        
    def send_message(self):
        global conversation_history, selected_model, model_ready
        
        # If worker is running, stop it instead
        if self.thread is not None and self.thread.isRunning():
            if hasattr(self, 'send_button') and self.send_button.text() == "Stop":
                # Check if we're already stopping
                if not self.stop_requested:
                    # Stop the worker
                    self.stop_requested = True  # Track that we requested a stop
                    self.stop_worker()
                else:
                    # We're already in the process of stopping, just show a message
                    self.status_label.setText("⏳ Already stopping, please wait...")
                    QtCore.QTimer.singleShot(2000, lambda: self.status_label.setText(""))
                return
            
        # If not connected, show error and return
        if not self.api_connected:
            self.status_label.setText("❌ Not connected to API server!")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return
        
        # If no model is selected or model is not ready, show error and return
        if not selected_model or selected_model.startswith("<") or not model_ready:
            self.status_label.setText("❌ Model is not ready! Please wait for model to load.")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            return
            
        prompt = self.prompt_input.toPlainText().strip()
        if prompt:
            # Show the previous token count with a "+" indicator
            if self.last_token_count > 0:
                self.token_count_label.setText(f"Tokens: {self.last_token_count}+ / {self.current_model_context}")
            else:
                self.token_count_label.setText(f"Tokens: 0 / {self.current_model_context}")
            
            # First check if conversation is too large and trim if needed
            self.check_conversation_size()
                
            # Make sure the prompt is properly formatted for markdown rendering
            if use_markdown and "\n```" in prompt and not prompt.endswith("```"):
                # Ensure code blocks are closed for proper rendering
                count = prompt.count("```")
                if count % 2 == 1:  # Odd number of ``` means unclosed code block
                    prompt += "\n```"
                        
            # Append user message and AI placeholder
            conversation_history.append(f"User:\n{prompt}")
            conversation_history.append("AI:\n")
            self.current_ai_index = len(conversation_history) - 1
            
            # Immediately update UI to show user's message
            self.update_chat_history()
            
            self.current_ai_text = ""
            self.prompt_input.clear()
            
            # Set focus to the chat history widget to prevent marking the API address
            self.chat_history_widget.setFocus()
            
            # Change the Send button to Stop
            self.send_button.setText("Stop")
            
            # Disable other buttons while generating response
            self.reset_button.setEnabled(False)
            self.markdown_button.setEnabled(False)
            self.exit_button.setEnabled(False)
            self.load_model_button.setEnabled(False)
            
            # Reset stop requested flag
            self.stop_requested = False
            
            # Update status indicator
            self.status_label.setText("⟳ Generating response... Please wait.")
            
            self.start_worker(prompt, conversation_history[:])
    
    def stop_worker(self):
        """Stop the current worker thread safely"""
        if self.thread is not None and self.thread.isRunning():
            print("DEBUG: Stopping response generation...")
            self.status_label.setText("⏳ Stopping generation...")
            
            try:
                # Set the stop flag first (important for signal flow control)
                if self.worker is not None:
                    self.worker.should_stop = True
                    print("DEBUG: Set should_stop flag")
                    
                    # Close the response connection if it exists (similar to AbortController)
                    response_obj = getattr(self.worker, 'response', None)
                    if response_obj is not None and hasattr(response_obj, 'close'):
                        try:
                            # This will trigger an exception in the response.iter_lines() loop
                            # which will be caught and handled properly
                            response_obj.close()
                            print("DEBUG: Closed response connection")
                        except Exception as e:
                            # Just log errors during closure - they're expected
                            print(f"DEBUG: Non-critical error closing response: {e}")
                    
                # Process events to keep UI responsive
                QtWidgets.QApplication.processEvents()
                
            except Exception as e:
                # Only log errors, don't show them to users
                print(f"DEBUG: Error during stopping: {e}")
            
            print("DEBUG: Stop signal sent to worker")
        
    def start_worker(self, prompt, current_history):
        # Check if previous thread is still running and properly clean up
        if self.thread is not None:
            if self.thread.isRunning():
                print("DEBUG: Stopping running thread.")
                self.thread.quit()
                self.thread.wait(1000)  # Wait up to 1 second
                print("DEBUG: Stopped and cleaned up the thread.")
            self.thread = None
            self.worker = None

        # Create new thread and worker
        print("DEBUG: Creating a new thread and worker.")
        self.thread = QtCore.QThread()
        self.worker = RequestWorker(prompt, current_history, self.current_model_context)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.worker.tokenCountUpdate.connect(self.update_token_count)
        self.worker.newChunk.connect(self.handle_new_chunk)
        self.worker.finished.connect(self.handle_finished)
        self.worker.connectionError.connect(self.handle_connection_error)
        self.worker.stoppedByUser.connect(self.handle_stopped_by_user)  # New signal
        self.thread.started.connect(self.worker.run)

        # Cleanup connections
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Reset references when objects are destroyed
        self.thread.destroyed.connect(lambda: setattr(self, 'thread', None))
        self.worker.destroyed.connect(lambda: setattr(self, 'worker', None))

        # Start the thread
        self.thread.start()
        print("DEBUG: Thread started.")

    def handle_connection_error(self, error_msg):
        # Update connection status on error
        self.api_connected = False
        self.update_connect_button()
        # Reset the send button
        self.send_button.setText("Send")
        # Re-enable buttons
        self.reset_button.setEnabled(True)
        self.markdown_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        # Show error in status bar (not in chat)
        self.status_label.setText(f"❌ {error_msg}")
        QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
        
        # Only add an error message to chat if this wasn't a user-requested stop
        if not self.stop_requested and self.current_ai_index < len(conversation_history):
            if not self.current_ai_text:
                self.current_ai_text = "I'm sorry, I encountered a connection error. Please try again."
            else:
                # We keep the content but don't add any special message about interruption
                pass
                
            conversation_history[self.current_ai_index] = "AI:\n" + self.current_ai_text
            self.update_chat_history()
        
        # Only trigger immediate API check if auto reconnect is enabled
        if self.auto_reconnect_enabled:
            QtCore.QTimer.singleShot(500, self.check_api_connection)

    def handle_stopped_by_user(self):
        """Handle when a response was intentionally stopped by user"""
        # Don't add a stop message to the chat history at all
        # Just show it in the status bar
        
        # Update UI immediately
        self.status_label.setText("✓ Generation stopped by user.")
        QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
        
        # Re-enable all buttons
        self.send_button.setText("Send")
        self.reset_button.setEnabled(True)
        self.markdown_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        
        # Reset the stop requested flag
        self.stop_requested = False

    def handle_new_chunk(self, chunk):
        # Store current scroll position before updating
        scrollbar = self.chat_history_widget.verticalScrollBar()
        current_scroll = scrollbar.value()
        
        # Update the conversation history
        self.current_ai_text += chunk
        conversation_history[self.current_ai_index] = "AI:\n" + self.current_ai_text
        
        # Only update display periodically for large chunks
        current_time = time.time()
        if not hasattr(self, 'last_ui_update_time'):
            self.last_ui_update_time = 0
        
        # Update display at most every 0.3 seconds for better performance
        if current_time - self.last_ui_update_time > 0.3 or len(chunk) < 50:
            self.update_chat_history(preserve_scroll=current_scroll)
            self.last_ui_update_time = current_time
            
            # Process events but don't overdo it
            QtWidgets.QApplication.processEvents()
            
    def handle_finished(self):
        # Re-enable all buttons when response is complete
        self.send_button.setText("Send")  # Reset button text
        self.reset_button.setEnabled(True)
        self.markdown_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        
        # Clear status indicator if not showing an error
        if not self.status_label.text().startswith("❌") and not self.status_label.text().startswith("⚠️"):
            self.status_label.setText("")
        
        # Reset stop requested flag
        self.stop_requested = False

    def reset_chat(self):
        global conversation_history
        conversation_history = []
        self.chat_history_widget.clear()
        self.last_token_count = 0  # Reset the token count on chat reset
        self.token_count_label.setText(f"Tokens: 0 / {self.current_model_context}")
        # Also reset markdown warning flag
        self.showed_markdown_warning = False

    def update_chat_history(self, preserve_scroll=None):
        """Update the chat history display with the latest content"""
        try:
            global use_markdown
            
            # For very large content, temporarily disable markdown
            temp_use_markdown = use_markdown
            total_text_length = sum(len(msg) for msg in conversation_history)
            
            if temp_use_markdown and total_text_length > LARGE_CONTENT_THRESHOLD:  # 100KB threshold
                temp_use_markdown = False
                if not self.showed_markdown_warning:
                    self.status_label.setText("⚠️ Markdown disabled for large content")
                    self.showed_markdown_warning = True
                    QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
                    
            # Get the current content to be displayed
            current_ai_text = ""
            if self.current_ai_index >= 0 and self.current_ai_index < len(conversation_history):
                current_content = conversation_history[self.current_ai_index]
                if current_content.startswith("AI:\n"):
                    current_ai_text = current_content[4:]  # Skip "AI:\n" prefix
            
            # Copy the conversation history
            temp_history = conversation_history[:]
            
            # If we're still generating the AI response, add special handling for code blocks
            if self.thread is not None and self.thread.isRunning() and self.current_ai_index >= 0:
                # Look for unfinished code blocks
                if "```" in current_ai_text and current_ai_text.count("```") % 2 == 1:
                    # Add a temporary closing backticks to make the markdown formatter happy
                    temp_history[self.current_ai_index] = temp_history[self.current_ai_index] + "\n```"
            
            # If HTML content is extremely large, use a simplified renderer
            if total_text_length > 200000:  # 200KB threshold
                # Create a simplified version for display
                simplified_html = "<div style='line-height:1.2;'>"
                simplified_html += "<h3 style='color:#FF9500;'>Content too large for full rendering</h3>"
                simplified_html += "<p>Showing last few messages only:</p>"
                
                # Get the last few messages only
                last_msgs = []
                for i in range(len(temp_history)-1, max(0, len(temp_history)-6), -1):
                    last_msgs.insert(0, temp_history[i])
                
                # Simple HTML for each message
                for msg in last_msgs:
                    if msg.startswith("User:\n"):
                        simplified_html += f"<div><h3 style='color:#FF9500;'>User:</h3><p>{msg[6:].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')}</p></div>"
                    elif msg.startswith("AI:\n"):
                        simplified_html += f"<div><h3 style='color:#FF9500;'>AI:</h3><p>{msg[4:].replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')}</p></div>"
                
                simplified_html += "</div>"
                self.chat_history_widget.setHtml(simplified_html)
            else:
                # Process markdown ahead of time for context window
                if temp_use_markdown:
                    # Pre-process all content for markdown to ensure proper rendering
                    for i, line in enumerate(temp_history):
                        if line.startswith("User:\n") or line.startswith("AI:\n"):
                            prefix = line.split("\n")[0] + "\n"
                            content = line[len(prefix):]
                            
                            try:
                                # Preprocess headings to limit to h3 at maximum
                                content = preprocess_markdown_headings(content)
                                
                                # Update the line with preprocessed content
                                temp_history[i] = prefix + content
                                
                                # Apply our custom markdown extension along with standard extensions
                                md = markdown.Markdown(extensions=[
                                    'fenced_code', 'tables', 'nl2br', CodeExtension()
                                ])
                                # Just parse the content to ensure it's valid markdown
                                md.convert(content)
                            except Exception as e:
                                print(f"DEBUG: Pre-processing markdown error at line {i}: {e}")
                                # If there's an error in markdown parsing, don't modify the line
                                pass
                
                # Generate HTML content using the temporary history and the temp_use_markdown value
                html_content = build_html_chat_history(temp_history)
                
                # Update the chat history text
                self.chat_history_widget.setHtml(html_content)
            
            # Restore scroll position if specified
            if preserve_scroll is not None:
                scrollbar = self.chat_history_widget.verticalScrollBar()
                scrollbar.setValue(preserve_scroll)
            else:
                # Auto-scroll to the bottom if no position specified
                scrollbar = self.chat_history_widget.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                
            # Process events to keep UI responsive
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            print(f"DEBUG: Error updating chat history: {e}")
            # Show a fallback message in the status bar
            self.status_label.setText("⚠️ Error updating display")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(""))
            
# -------------------------------------------------------
# 9) Dark Mode Styling
# -------------------------------------------------------
def apply_dark_mode(app):
    QtWidgets.QApplication.setStyle("Fusion")

    dark_palette = QtGui.QPalette()
    
    base_window_color = QtGui.QColor("#2f2f2f")
    chat_bg_color     = QtGui.QColor("#2a2a2a")
    alt_base_color    = QtGui.QColor("#3b3b3b")
    text_color        = QtGui.QColor("#ffffff")
    button_color      = QtGui.QColor("#3e3e3e")
    highlight_color   = QtGui.QColor("#537BA2")
    border_color      = QtGui.QColor("#4f4f4f")

    dark_palette.setColor(QtGui.QPalette.Window, base_window_color)
    dark_palette.setColor(QtGui.QPalette.WindowText, text_color)
    dark_palette.setColor(QtGui.QPalette.Base, alt_base_color)
    dark_palette.setColor(QtGui.QPalette.AlternateBase, base_window_color)
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, text_color)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, text_color)
    dark_palette.setColor(QtGui.QPalette.Text, text_color)
    dark_palette.setColor(QtGui.QPalette.Button, button_color)
    dark_palette.setColor(QtGui.QPalette.ButtonText, text_color)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, highlight_color)
    dark_palette.setColor(QtGui.QPalette.Highlight, highlight_color)
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

    app.setPalette(dark_palette)

    app.setStyleSheet(f"""
        QWidget {{
            font-size: 10pt;
            color: {text_color.name()};
        }}
        QToolTip {{
            color: #ffffff;
            background-color: {highlight_color.name()};
            border: 1px solid {text_color.name()};
        }}
        QPushButton {{
            border: 1px solid {border_color.name()};
            background-color: {button_color.name()};
            padding: 6px;
        }}
        QPushButton:hover {{
            background-color: #4a4a4a;
        }}
        QPushButton:pressed {{
            background-color: #5a5a5a;
        }}
        QPushButton:disabled {{
            background-color: #282828;
            color: #606060;
            border: 1px solid #404040;
        }}
        QLineEdit, QComboBox {{
            background-color: {alt_base_color.name()};
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QComboBox::drop-down {{
            border-left: 1px solid {border_color.name()};
        }}
        QTextEdit#ChatHistory {{
            background-color: #2a2a2a;
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QTextEdit#PromptInput {{
            background-color: {alt_base_color.name()};
            border: 1px solid {border_color.name()};
            color: {text_color.name()};
        }}
        QScrollBar:vertical {{
            background-color: {alt_base_color.name()};
            width: 12px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background-color: #555555;
            min-height: 20px;
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            background: none;
            border: none;
            height: 0px;
        }}
    """)

# -------------------------------------------------------
# 10) Main Entry
# -------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_mode(app)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())