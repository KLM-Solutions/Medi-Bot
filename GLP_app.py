import streamlit as st
import requests
import json
from typing import Dict, Any, Optional, Iterator

class GLP1Bot:
    def __init__(self):
        """Initialize the GLP1Bot with PPLX client and system prompts"""
        if 'pplx' not in st.secrets:
            raise ValueError("PPLX API key not found in secrets")
            
        self.pplx_api_key = st.secrets["pplx"]["api_key"]
        self.pplx_model = st.secrets["pplx"].get("model", "medical-pplx")  
        
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json"
        }
        
        self.pplx_system_prompt = """
You are a specialized medical information assistant focused EXCLUSIVELY on GLP-1 medications (such as Ozempic, Wegovy, Mounjaro, etc.). You must:

1. ONLY provide information about GLP-1 medications and directly related topics
2. For any query not specifically about GLP-1 medications or their direct effects, respond with:
   "I apologize, but I can only provide information about GLP-1 medications and related topics. Your question appears to be about something else. Please ask a question specifically about GLP-1 medications, their usage, effects, or related concerns."

3. For valid GLP-1 queries, structure your response with:
   - An empathetic opening acknowledging the patient's situation
   - Clear, validated medical information about GLP-1 medications
   - Important safety considerations or disclaimers
   - An encouraging closing that reinforces their healthcare journey
   - Include relevant sources for the information provided, using the format: [Source: Title or description (Year if available)]

Remember: You must NEVER provide information about topics outside of GLP-1 medications and their direct effects.
Each response must include relevant medical disclaimers and encourage consultation with healthcare providers.
Always cite your sources for medical claims and information.
"""

    def get_pplx_stream(self, query: str) -> Iterator[Dict[str, Any]]:
        """Get streaming response from PPLX API"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": f"{query}\n\nPlease include sources for the information provided."}
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
                "stream": True  # Enable streaming
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload,
                stream=True
            )
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    if line.strip() == b"data: [DONE]":
                        break
                    if line.startswith(b"data: "):
                        json_str = line[6:].decode('utf-8')
                        try:
                            json_data = json.loads(json_str)
                            content = json_data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield f"Error: {str(e)}"

    def process_streaming_query(self, user_query: str, placeholder) -> Dict[str, Any]:
        """Process user query with streaming response"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            # Initialize response container
            response_container = {
                "status": "success",
                "query_category": self.categorize_query(user_query),
                "original_query": user_query,
                "response": "",
                "sources": ""
            }
            
            full_response = []
            sources_started = False
            sources = []
            
            # Stream the response
            for chunk in self.get_pplx_stream(user_query):
                if "Sources:" in chunk:
                    sources_started = True
                    continue
                
                if sources_started:
                    sources.append(chunk)
                else:
                    full_response.append(chunk)
                
                # Update the placeholder with current content
                current_content = "".join(full_response)
                if sources_started:
                    current_content += "\n\nSources:\n" + "".join(sources)
                
                # Add disclaimer
                formatted_content = current_content + "\n\nDisclaimer: Always consult your healthcare provider before making any changes to your medication or treatment plan."
                
                # Update the placeholder
                placeholder.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="category-tag">{response_container["query_category"].upper()}</div><br>
                    <b>Response:</b><br>{formatted_content}
                </div>
                """, unsafe_allow_html=True)
            
            # Store final response
            response_container["response"] = "".join(full_response)
            response_container["sources"] = "".join(sources)
            
            return response_container
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

    def categorize_query(self, query: str) -> str:
        """Categorize the user query"""
        categories = {
            "dosage": ["dose", "dosage", "how to take", "when to take", "injection", "administration"],
            "side_effects": ["side effect", "adverse", "reaction", "problem", "issues", "symptoms"],
            "benefits": ["benefit", "advantage", "help", "work", "effect", "weight", "glucose"],
            "storage": ["store", "storage", "keep", "refrigerate", "temperature"],
            "lifestyle": ["diet", "exercise", "lifestyle", "food", "alcohol", "eating"],
            "interactions": ["interaction", "drug", "medication", "combine", "mixing"],
            "cost": ["cost", "price", "insurance", "coverage", "afford"]
        }
        
        query_lower = query.lower()
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general"

def main():
    """Main application function"""
    try:
        st.set_page_config(
            page_title="GLP-1 Medication Assistant",
            page_icon="💊",
            layout="wide"
        )
        
        set_page_style()  # Keep your existing style function
        
        if 'pplx' not in st.secrets:
            st.error('Required PPLX API key not found. Please configure the PPLX API key in your secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        <div class="info-box">
        Get accurate, validated information specifically about GLP-1 medications, their usage, benefits, and side effects.
        Our assistant specializes exclusively in GLP-1 medications and related topics.
        
        <em>Please note: This assistant provides general information about GLP-1 medications only. Always consult your healthcare provider for medical advice.</em>
        </div>
        """, unsafe_allow_html=True)
        
        bot = GLP1Bot()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        with st.container():
            user_input = st.text_input(
                "Ask your question about GLP-1 medications:",
                key="user_input",
                placeholder="e.g., What are the common side effects of Ozempic?"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit_button = st.button("Get Answer", key="submit")
            
            if submit_button and user_input:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>Your Question:</b><br>{user_input}
                </div>
                """, unsafe_allow_html=True)
                
                # Create a placeholder for the streaming response
                response_placeholder = st.empty()
                
                # Process the query with streaming
                response = bot.process_streaming_query(user_input, response_placeholder)
                
                if response["status"] == "success":
                    st.session_state.chat_history.append({
                        "query": user_input,
                        "response": response
                    })
                else:
                    st.error(response["message"])
        
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### Previous Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>Your Question:</b><br>{chat['query']}
                    </div>
                    <div class="chat-message bot-message">
                        <div class="category-tag">{chat['response']['query_category'].upper()}</div><br>
                        <b>Response:</b><br>{chat['response']['response']}
                        <div class="sources-section">
                            <b>Sources:</b><br>{chat['response']['sources']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
