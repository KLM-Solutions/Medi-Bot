import streamlit as st
from openai import OpenAI
from typing import Dict, Any, Optional

class GLP1Bot:
    def __init__(self):
        """Initialize the GLP1Bot with OpenAI client and system prompt"""
        if 'openai' not in st.secrets:
            raise ValueError("OpenAI API key not found in secrets")
            
        self.client = OpenAI(
            api_key=st.secrets["openai"]["api_key"]
        )
        
        self.system_prompt = """You are a medical information assistant specialized in GLP-1 medications. 
        Provide accurate, evidence-based information about:
        - Proper usage and administration
        - Benefits and expected outcomes
        - Side effects and management
        - Storage and handling
        - Drug interactions
        - Lifestyle modifications
        
        Always include appropriate medical disclaimers and remind users to consult healthcare providers.
        Format responses clearly with relevant sections and bullet points.
        If you're unsure about any information, acknowledge the limitation and advise consulting a healthcare provider."""
    
    def get_response(self, query: str) -> Optional[str]:
        """Get response from OpenAI API"""
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error communicating with OpenAI: {str(e)}")
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
        """
        
        if "disclaimer" not in response.lower():
            response += safety_disclaimer
            
        return response
    
    def categorize_query(self, query: str) -> str:
        """Categorize the user query"""
        categories = {
            "dosage": ["dose", "dosage", "how to take", "when to take", "injection"],
            "side_effects": ["side effect", "adverse", "reaction", "problem"],
            "benefits": ["benefit", "advantage", "help", "work", "effect"],
            "storage": ["store", "storage", "keep", "refrigerate"],
            "lifestyle": ["diet", "exercise", "lifestyle", "food", "alcohol"]
        }
        
        query_lower = query.lower()
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general"

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and return structured response"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
                
            query_category = self.categorize_query(user_query)
            response = self.get_response(user_query)
            formatted_response = self.format_response(response)
            
            return {
                "status": "success",
                "query_category": query_category,
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
        if 'openai' not in st.secrets:
            st.error('OpenAI API key not found. Please configure it in your secrets.')
            st.stop()
        
        st.title("💊 GLP-1 Medication Information Assistant")
        st.markdown("""
        Get accurate information about GLP-1 medications, their usage, benefits, and side effects.
        
        *Please note: This assistant provides general information only. Always consult your healthcare provider for medical advice.*
        """)
        
        # Initialize bot
        bot = GLP1Bot()
        
        # Create session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        with st.container():
            user_input = st.text_input(
                "Ask your question about GLP-1 medications:",
                key="user_input",
                placeholder="e.g., What are the common side effects of GLP-1 medications?"
            )
            
            if st.button("Get Answer", key="submit"):
                if user_input:
                    with st.spinner('Processing your question...'):
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
                        <div class="category-tag">{chat['response']['query_category'].upper()}</div><br>
                        <b>Response:</b><br>{chat['response']['response']}
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
