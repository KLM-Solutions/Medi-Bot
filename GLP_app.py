import streamlit as st
from openai import OpenAI
import requests
from typing import Dict, Any, Optional

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
        
        # Prompt for reframing questions
        self.reframe_prompt = """You are a medical query optimizer. Your task is to:
        1. Reframe the user's question to be more specific and medically precise
        2. Ensure the question captures all relevant medical context
        3. Structure the question to elicit a comprehensive medical response
        4. Maintain the original intent while using proper medical terminology
        
        Reframe the following question while keeping it concise and focused."""

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

    def reframe_question(self, original_query: str) -> str:
        """Reframe the user's question using GPT-4o-mini"""
        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.reframe_prompt},
                    {"role": "user", "content": original_query}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error reframing question: {str(e)}")
            return original_query

    def get_pplx_response(self, reframed_query: str) -> Optional[str]:
        """Get response from PPLX API"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": reframed_query}
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

    def validate_response(self, reframed_query: str, pplx_response: str) -> tuple[bool, str]:
        """Validate the response against the reframed question using GPT-4o-mini"""
        try:
            validation_input = f"""
            Reframed Question: {reframed_query}
            
            Response to validate:
            {pplx_response}
            """
            
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.validation_prompt},
                    {"role": "user", "content": validation_input}
                ],
                temperature=0.1
            )
            
            validation_result = completion.choices[0].message.content
            is_valid = validation_result.startswith("VALID")
            
            if is_valid:
                return True, validation_result[6:].strip()  # Remove "VALID" prefix
            return False, validation_result
            
        except Exception as e:
            st.error(f"Error validating response: {str(e)}")
            return False, str(e)

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
        
        return response + safety_disclaimer

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the new streamlined workflow"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            
            # Step 1: Reframe the question
            with st.spinner('🤔 Reframing your question...'):
                reframed_query = self.reframe_question(user_query)
                if not reframed_query:
                    return {
                        "status": "error",
                        "message": "Failed to reframe the question."
                    }
            
            # Step 2: Get PPLX response
            with st.spinner('🔍 Getting medical information...'):
                pplx_response = self.get_pplx_response(reframed_query)
                if not pplx_response:
                    return {
                        "status": "error",
                        "message": "Failed to retrieve medical information."
                    }
            
            # Step 3: Validate response
            with st.spinner('✅ Validating response...'):
                is_valid, validated_response = self.validate_response(reframed_query, pplx_response)
                if not is_valid:
                    return {
                        "status": "error",
                        "message": f"Response validation failed: {validated_response}"
                    }
            
            # Step 4: Format and return response
            formatted_response = self.format_response(validated_response)
            
            return {
                "status": "success",
                "original_query": user_query,
                "reframed_query": reframed_query,
                "response": formatted_response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

def main():
    """Main application function"""
    try:
        st.set_page_config(
            page_title="GLP-1 Medication Assistant",
            page_icon="💊",
            layout="wide"
        )
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        Get accurate, validated information about GLP-1 medications, their usage, benefits, and side effects.
        
        *Please note: This assistant provides general information only. Always consult your healthcare provider for medical advice.*
        """)
        
        # Initialize bot
        bot = GLP1Bot()
        
        # Create session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Main chat interface
        user_input = st.text_input(
            "Ask your question about GLP-1 medications:",
            key="user_input",
            placeholder="e.g., What are the common side effects of GLP-1 medications?"
        )
        
        if st.button("Get Answer", key="submit"):
            if user_input:
                response = bot.process_query(user_input)
                
                if response["status"] == "success":
                    # Display reframed question
                    st.markdown("#### How I understood your question:")
                    st.info(response["reframed_query"])
                    
                    # Display response
                    st.markdown("#### Response:")
                    st.write(response["response"])
                    
                    # Add to chat history
                    st.session_state.chat_history.append(response)
                else:
                    st.error(response["message"])
            else:
                st.warning("Please enter a question.")
        
        # Clear history button
        if st.button("Clear History") and st.session_state.chat_history:
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### Previous Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                with st.expander(f"Question {len(st.session_state.chat_history) - i}"):
                    st.markdown("**Original Question:**")
                    st.write(chat["original_query"])
                    st.markdown("**Reframed Question:**")
                    st.info(chat["reframed_query"])
                    st.markdown("**Response:**")
                    st.write(chat["response"])

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
