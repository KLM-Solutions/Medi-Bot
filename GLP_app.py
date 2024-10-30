import streamlit as st
from openai import OpenAI
import requests
from typing import Dict, Any, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GLP1Bot:
    def __init__(self):
        """Initialize the GLP1Bot with both PPLX and OpenAI clients and system prompts"""
        if 'openai' not in st.secrets or 'pplx' not in st.secrets:
            raise ValueError("API keys not found in secrets")
            
        self.openai_client = OpenAI(
            api_key=st.secrets["openai"]["api_key"]
        )
        self.pplx_api_key = st.secrets["pplx"]["api_key"]
        self.pplx_model = st.secrets["pplx"].get("model", "medical-pplx")
        
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json"
        }
        self.pplx_system_prompt = """
You are a medical information assistant specialized in GLP-1 medications. Provide detailed, evidence-based information with an empathetic tone.
Cover important aspects such as:
- Mechanism of action
- Proper usage and administration
- Expected outcomes and timeframes
- Potential side effects and management
- Drug interactions and contraindications
- Storage requirements
- Lifestyle modifications for optimal results
Format your response with:
1. An empathetic opening acknowledging the patient's situation
2. Clear medical information using supportive language
3. A encouraging closing that reinforces their healthcare journey
Focus on medical accuracy while maintaining a compassionate tone throughout.
"""
        self.gpt_validation_prompt = """
You are a medical content validator. Review and enhance the following information about GLP-1 medications.
Ensure the response is:
1. Medically accurate and evidence-based
2. Well-structured with clear sections
3. Includes appropriate medical disclaimers
4. Easy to understand for patients
5. Comprehensive yet concise
6. Properly formatted with headers and bullet points
7. Written with empathy and understanding
8. Concludes with supportive guidance
Add any missing critical information and correct any inaccuracies.
Maintain a professional yet approachable tone, emphasizing both expertise and emotional support.
"""

   def get_pplx_response(self, query: str) -> Optional[str]:
        """Get initial response from PPLX API with timing"""
        start_time = time.time()
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
            result = response.json()["choices"][0]["message"]["content"]
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"PPLX Response Time: {elapsed_time:.2f} seconds")
            st.sidebar.write(f"🕒 PPLX Response Time: {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"PPLX Error ({elapsed_time:.2f}s): {str(e)}")
            st.error(f"Error communicating with PPLX: {str(e)}")
            return None

    def validate_with_gpt(self, pplx_response: str, original_query: str) -> Optional[str]:
        """Validate and enhance PPLX response using GPT with timing"""
        start_time = time.time()
        try:
            validation_prompt = f"""
            Original query: {original_query}
            
            PPLX Response to validate:
            {pplx_response}
            
            Please validate and enhance this response according to medical standards and best practices.
            Ensure all information is accurate and properly structured.
            """
            
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.gpt_validation_prompt},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result = completion.choices[0].message.content
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"GPT Validation Time: {elapsed_time:.2f} seconds")
            st.sidebar.write(f"🕒 GPT Validation Time: {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"GPT Error ({elapsed_time:.2f}s): {str(e)}")
            st.error(f"Error validating with GPT: {str(e)}")
            return None

    def format_response(self, response: str) -> str:
        """Format the response with safety disclaimer"""
        if not response:
            return "I apologize, but I couldn't generate a response at this time. Please try again."
            
        safety_disclaimer = """
        
        IMPORTANT MEDICAL DISCLAIMER:
        - This information is for educational purposes only
        - Consult your healthcare provider for personalized medical advice
        - Follow your prescribed treatment plan
        - Report any side effects to your healthcare provider
        - Individual results may vary
        - Never modify your medication regimen without professional guidance
        """
        
        if "disclaimer" not in response.lower():
            response += safety_disclaimer
            
        return response

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

     def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through both PPLX and GPT with total time tracking"""
        total_start_time = time.time()
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            # Add timing information to the sidebar
            st.sidebar.markdown("### Response Time Metrics")
            
            # Step 1: Get initial response from PPLX
            with st.spinner('🔍 Retrieving information from medical knowledge base...'):
                pplx_response = self.get_pplx_response(user_query)
            
            if not pplx_response:
                return {
                    "status": "error",
                    "message": "Failed to retrieve information from knowledge base."
                }
            
            # Step 2: Validate and enhance with GPT
            with st.spinner('✅ Validating and enhancing information...'):
                validated_response = self.validate_with_gpt(pplx_response, user_query)
            
            if not validated_response:
                return {
                    "status": "error",
                    "message": "Failed to validate information."
                }
            
            # Calculate and display total processing time
            total_end_time = time.time()
            total_elapsed_time = total_end_time - total_start_time
            logger.info(f"Total Processing Time: {total_elapsed_time:.2f} seconds")
            st.sidebar.write(f"⏱️ Total Processing Time: {total_elapsed_time:.2f} seconds")
            
            # Format final response
            query_category = self.categorize_query(user_query)
            formatted_response = self.format_response(validated_response)
            
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "pplx_response": pplx_response,
                "response": formatted_response,
                "timing": {
                    "total_time": total_elapsed_time
                }
            }
            
        except Exception as e:
            total_end_time = time.time()
            total_elapsed_time = total_end_time - total_start_time
            logger.error(f"Total Error ({total_elapsed_time:.2f}s): {str(e)}")
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
        .category-tag {
            background-color: #2196f3;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
            display: inline-block;
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
        .metrics-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1976d2;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
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
        
        # Check for API keys
        if 'openai' not in st.secrets or 'pplx' not in st.secrets:
            st.error('Required API keys not found. Please configure both OpenAI and PPLX API keys in your secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        <div class="info-box">
        Get accurate, validated information about GLP-1 medications, their usage, benefits, and side effects.
        This assistant uses a two-stage process:
        1. Retrieves specialized medical information (PPLX API)
        2. Validates and enhances the information for accuracy (GPT API)
        
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
                submit_button = st.button("Get Answer", key="submit", use_container_width=True)
            with col2:
                if st.button("Clear History", key="clear", use_container_width=True):
                    st.session_state.chat_history = []
                    st.experimental_rerun()
            
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
                        
                        # Display timing metrics in an organized way
                        with st.expander("View Processing Times"):
                            timing = response["timing"]
                            cols = st.columns(3)
                            cols[0].metric("PPLX API Time", f"{timing['pplx_time']:.2f}s")
                            cols[1].metric("GPT API Time", f"{timing['gpt_time']:.2f}s")
                            cols[2].metric("Total Time", f"{timing['total_time']:.2f}s")
                    else:
                        st.error(response["message"])
               
