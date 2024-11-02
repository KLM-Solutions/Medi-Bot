import streamlit as st
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
        
        self.pplx_system_prompt = """
You are a medical information assistant specialized in GLP-1 medications. Provide detailed, evidence-based information with an empathetic tone based on the user query for the GLP-1 drugs.

Format your response with:
1. An empathetic opening acknowledging the patient's situation
2. Clear medical information based on the user query 
3. A encouraging closing that reinforces their healthcare journey
Focus on medical accuracy while maintaining a compassionate tone throughout.

Only provide information about GLP-1 medications and related treatment aspects. If the query is not related to GLP-1 medications, politely redirect the user to ask about GLP-1 medications.
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
        """Format the response with safety disclaimer"""
        if not response:
            return "I apologize, but I couldn't generate a response at this time. Please try again."
            
        disclaimer = "\n\n*Note: This information is for educational purposes only. Always consult your healthcare provider for medical advice.*"
        return response + disclaimer

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through PPLX"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            # Get response from PPLX
            with st.spinner('🔍 Retrieving information about GLP-1 medications...'):
                pplx_response = self.get_pplx_response(user_query)
            
            if not pplx_response:
                return {
                    "status": "error",
                    "message": "Failed to retrieve information from knowledge base."
                }
            
            # Format final response
            formatted_response = self.format_response(pplx_response)
            
            return {
                "status": "success",
                "original_query": user_query,
                "response": formatted_response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

def set_page_style():
    """Set page style using custom CSS"""
    st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput>div>div>input {
            background-color: white;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.8rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
        }
        .bot-message {
            background-color: #f5f5f5;
            border-left: 4px solid #43a047;
        }
        .stAlert {
            background-color: #ff5252;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .disclaimer {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff9800;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
        .info-box {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .processing-status {
            color: #1976d2;
            font-style: italic;
            margin: 0.5rem 0;
        }
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
            st.error('Required PPLX API key not found. Please configure PPLX API key in your secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        <div class="info-box">
        Get accurate information about GLP-1 medications, their usage, benefits, and side effects.
        This assistant uses specialized medical knowledge to provide detailed responses about GLP-1 medications.
        
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
                            <b>Response:</b><br>{response["response"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(response["message"])
                else:
                    st.warning("Please enter a question.")
        
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
                        <b>Response:</b><br>{chat['response']['response']}
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
