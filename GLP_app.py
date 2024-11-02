import streamlit as st
# from openai import OpenAI  # Commented out but kept for future reference
import requests
from typing import Dict, Any, Optional

class GLP1Bot:
    def __init__(self):
        """Initialize the GLP1Bot with PPLX client and system prompt"""
        if 'pplx' not in st.secrets:
            raise ValueError("PPLX API key not found in secrets")
            
        self.pplx_api_key = st.secrets["pplx"]["api_key"]
        self.pplx_model = st.secrets["pplx"].get("model", "medical-pplx")  
        
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json"
        }
        
        # Updated system prompt to be more specific to GLP-1 medications
        self.pplx_system_prompt = """
You are a medical information assistant specialized exclusively in GLP-1 medications (such as Ozempic, Wegovy, Mounjaro, and similar GLP-1 receptor agonists). 
Only provide information specifically about GLP-1 medications and their direct effects. If a query is not related to GLP-1 medications, politely redirect the conversation back to GLP-1 topics.

Format your response with:
1. An empathetic opening acknowledging the patient's situation, specifically related to GLP-1 medication use
2. Clear, factual medical information about GLP-1 medications based on the user query
3. An encouraging closing that reinforces their GLP-1 medication journey

If any question goes beyond the scope of GLP-1 medications, respond with: 
"I can only provide information about GLP-1 medications. Please consult your healthcare provider for information about other treatments or conditions."
"""

    def get_pplx_response(self, query: str) -> Optional[str]:
        """Get response from PPLX API"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            st.error(f"Error communicating with PPLX: {str(e)}")
            return None

    def format_response(self, response: str) -> str:
        """Format the response with GLP-1 specific safety disclaimer"""
        if not response:
            return "I apologize, but I couldn't generate a response about GLP-1 medications at this time. Please try again."
            
    def categorize_query(self, query: str) -> str:
        """Categorize the user query specifically for GLP-1 medications"""
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

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query using only PPLX"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question about GLP-1 medications."
                }
            
            # Get response from PPLX
            with st.spinner('🔍 Retrieving information about GLP-1 medications...'):
                pplx_response = self.get_pplx_response(user_query)
            
            if not pplx_response:
                return {
                    "status": "error",
                    "message": "Failed to retrieve information about GLP-1 medications."
                }
            
            # Format final response
            query_category = self.categorize_query(user_query)
            formatted_response = self.format_response(pplx_response)
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "response": formatted_response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing GLP-1 medication query: {str(e)}"
            }

# [Rest of the code remains unchanged - keeping all the styling and UI elements]
def set_page_style():
    """Set page style using custom CSS"""
    st.markdown("""
    <style>
        [... existing styles ...]
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    try:
        st.set_page_config(
            page_title="GLP-1 Medication Assistant",
            page_icon="💊",
            layout="wide"
        )
        
        set_page_style()
        
        # Check for API key
        if 'pplx' not in st.secrets:
            st.error('Required PPLX API key not found. Please configure the PPLX API key in your secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        <div class="info-box">
        Get accurate information about GLP-1 medications, their usage, benefits, and side effects.
        This assistant provides specialized medical information about GLP-1 medications from a comprehensive medical knowledge base.
        
        <em>Please note: This assistant provides general information only. Always consult your healthcare provider for medical advice.</em>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize bot
        bot = GLP1Bot()
        
        # Create session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Main chat interface
        with st.container():
            user_input = st.text_input(
                "Ask your question about GLP-1 medications:",
                key="user_input",
                placeholder="e.g., What are the common side effects of GLP-1 medications?"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit_button = st.button("Get Answer", key="submit")
            
            if submit_button:
                if user_input:
                    response = bot.process_query(user_input)
                    
                    if response["status"] == "success":
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": user_input,
                            "response": response
                        })
                        
                        # Display current response
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <b>Your Question:</b><br>{user_input}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="chat-message bot-message">
                            <div class="category-tag">{response["query_category"].upper()}</div><br>
                            <b>Response:</b><br>{response["response"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(response["message"])
                else:
                    st.warning("Please enter a question about GLP-1 medications.")
        
        # Display chat history
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
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
